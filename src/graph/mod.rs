//! The execution engine. A *run* pursues one goal on its own branch in its
//! own git worktree, moving through phases:
//!
//! ```text
//!   plan ──► act ⇄ (tool) ──finish──► verify ──pass──► review ──approve──► done
//!             ▲                          │fail            │revise
//!             └──────────────────────────┴────────────────┘
//! ```
//!
//! Every step (one LLM turn plus its tool call, one verification, one
//! review, one injected user message) is checkpointed: a git commit of the
//! worktree plus a SQLite row carrying the full loop state. That makes
//! Pause trivial (stop between steps), Resume and crash recovery "continue
//! from the last row", and Rewind "reset the worktree to row N's commit and
//! drop later rows".
//!
//! Routing: the actor model (normally local) does the work; after repeated
//! failures the next turn is escalated to a stronger model; when the cloud
//! budget is exhausted, cloud roles fall back to a local actor.

pub mod external;
pub mod orchestra;
pub mod prompts;
pub mod protocol;
pub mod tools;

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use serde_json::json;
use tokio::sync::broadcast;

use crate::bootstrap::Workspace;
use crate::config::{Config, ModelRef, RemoteAuth};
use crate::error::{Error, Result};
use crate::fsutil::clip;
use crate::git::GitRepo;
use crate::providers::{CallOutcome, CompletionRequest, Message, ProviderId};
use crate::storage::{NewStep, Phase, Run, RunMode, RunSpec, RunState, RunStatus, Step, StepKind, now_unix};
use tools::{MAX_STORED_OUTPUT, ToolBox, run_shell};

const EVENT_CHANNEL_CAPACITY: usize = 512;

/// Pushed to every subscriber (dashboard SSE, CLI) as things happen.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    Run { run: Run },
    Step { step: Step },
    Rewound { run_id: String, seq: i64 },
    Deleted { run_id: String },
    Log { run_id: String, message: String },
}

enum Stop {
    Done,
    Paused(Option<String>),
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Inner>,
}

struct Inner {
    ws: Arc<Workspace>,
    /// Pause flags of runs whose loop is currently executing.
    active: Mutex<HashMap<String, Arc<AtomicBool>>>,
    events: broadcast::Sender<Event>,
    /// Serializes step appends, so an inject arriving mid-run cannot race
    /// the loop for the next sequence number or a git commit.
    append_lock: tokio::sync::Mutex<()>,
    /// Held for the engine's lifetime: one executing process per workspace.
    _lock: std::fs::File,
}

impl Engine {
    /// Takes the workspace's engine lock (so the server and a CLI `run`
    /// cannot drive the same runs at once) and marks runs left `running` by
    /// a previous process as interrupted.
    pub async fn new(ws: Arc<Workspace>) -> Result<Self> {
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(ws.state_dir.join("engine.lock"))?;
        lock.try_lock().map_err(|_| {
            Error::Conflict(
                "another orchopork process (the server or a CLI run) is executing runs in this workspace".into(),
            )
        })?;
        let n = ws.store.mark_interrupted_runs().await?;
        if n > 0 {
            tracing::warn!("{n} run(s) were interrupted by a previous exit; resume them to continue");
        }
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Ok(Self {
            inner: Arc::new(Inner {
                ws,
                active: Mutex::new(HashMap::new()),
                events,
                append_lock: tokio::sync::Mutex::new(()),
                _lock: lock,
            }),
        })
    }

    pub fn workspace(&self) -> &Arc<Workspace> {
        &self.inner.ws
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.inner.events.subscribe()
    }

    pub fn is_active(&self, run_id: &str) -> bool {
        self.inner.active.lock().unwrap().contains_key(run_id)
    }

    pub async fn list_runs(&self) -> Result<Vec<Run>> {
        self.inner.ws.store.list_runs().await
    }

    pub async fn run_detail(&self, run_id: &str) -> Result<(Run, Vec<Step>)> {
        let store = &self.inner.ws.store;
        Ok((store.get_run(run_id).await?, store.steps(run_id).await?))
    }

    /// The run plus only the steps after `seq`.
    pub async fn run_detail_after(&self, run_id: &str, seq: i64) -> Result<(Run, Vec<Step>)> {
        let store = &self.inner.ws.store;
        Ok((store.get_run(run_id).await?, store.steps_after(run_id, seq).await?))
    }

    /// Creates the run's branch + worktree from the workspace's current
    /// HEAD and starts it. Uncommitted changes in your checkout are not
    /// part of the run.
    pub async fn create_run(&self, goal: &str, verify_command: Option<&str>, spec: RunSpec) -> Result<Run> {
        let ws = &self.inner.ws;
        let goal = goal.trim();
        if goal.is_empty() {
            return Err(Error::InvalidRequest("goal is empty".into()));
        }
        match spec.mode {
            RunMode::Classic if ws.config().routing.actor.is_none() => {
                return Err(Error::InvalidRequest("choose an actor model in settings first".into()));
            }
            RunMode::Solo if spec.model.is_none() => {
                return Err(Error::InvalidRequest("choose a model for this chat".into()));
            }
            RunMode::Orchestra => {
                spec.validate_agents()?;
                if spec.model.is_none() && spec.agents.iter().any(|a| a.model.is_none()) {
                    return Err(Error::InvalidRequest("choose a lead model for the orchestra".into()));
                }
            }
            _ => {}
        }
        if !ws.repo.is_repo().await {
            return Err(Error::InvalidRequest(
                "the workspace is not a git repository (initialize it in setup or with `orchopork init --git-init`)"
                    .into(),
            ));
        }
        let verify = verify_command.map(str::trim).filter(|v| !v.is_empty());
        let run = {
            let _g = self.inner.append_lock.lock().await;
            let base = ws.repo.ensure_initial_commit().await?;
            let id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
            let branch = format!("orchopork/{id}");
            let worktree = ws.state_dir.join("worktrees").join(&id);
            ws.repo.worktree_add(&worktree, &branch, &base).await?;
            ws.store.insert_run(&id, goal, &branch, &worktree.to_string_lossy(), &base, verify, &spec).await?
        };
        self.inner.emit(Event::Run { run: run.clone() });
        self.start(&run.id).await
    }

    /// Starts (or resumes) a run's loop in the background.
    pub async fn start(&self, run_id: &str) -> Result<Run> {
        let store = &self.inner.ws.store;
        store.get_run(run_id).await?;
        if let Some(head) = store.head_step(run_id).await?
            && head.state.phase == Phase::Done
        {
            return Err(Error::Conflict("this run is done; inject a follow-up instruction to continue it".into()));
        }
        let flag = Arc::new(AtomicBool::new(false));
        {
            let mut active = self.inner.active.lock().unwrap();
            if active.contains_key(run_id) {
                return Err(Error::Conflict("run is already executing".into()));
            }
            active.insert(run_id.to_string(), flag.clone());
        }
        let run = match store.set_run_status(run_id, RunStatus::Running, None).await {
            Ok(r) => r,
            Err(e) => {
                self.inner.active.lock().unwrap().remove(run_id);
                return Err(e);
            }
        };
        self.inner.emit(Event::Run { run: run.clone() });
        let inner = self.inner.clone();
        let id = run_id.to_string();
        tokio::spawn(async move { inner.drive(id, flag).await });
        Ok(run)
    }

    /// Asks the loop to stop after the step in progress.
    pub async fn pause(&self, run_id: &str) -> Result<Run> {
        let flag = self.inner.active.lock().unwrap().get(run_id).cloned();
        let store = &self.inner.ws.store;
        match flag {
            Some(f) => {
                f.store(true, Ordering::SeqCst);
                self.inner.log(run_id, "pausing after the current step");
                store.get_run(run_id).await
            }
            None => {
                let run = store.get_run(run_id).await?;
                if run.status == RunStatus::Running {
                    // Stale status with no live loop behind it.
                    let run = store.set_run_status(run_id, RunStatus::Paused, None).await?;
                    self.inner.emit(Event::Run { run: run.clone() });
                    return Ok(run);
                }
                Ok(run)
            }
        }
    }

    /// Adds a user message the actor sees on its next turn. Works while the
    /// run executes, while paused, and on a finished run (which re-opens it
    /// for a follow-up; resume to continue).
    pub async fn inject(&self, run_id: &str, text: &str, agent: Option<&str>) -> Result<Step> {
        let text = text.trim();
        if text.is_empty() {
            return Err(Error::InvalidRequest("message is empty".into()));
        }
        let ws = &self.inner.ws;
        let _g = self.inner.append_lock.lock().await;
        let run = ws.store.get_run(run_id).await?;
        let mut state = ws.store.head_step(run_id).await?.map(|s| s.state).unwrap_or_default();
        let mut meta = json!({});
        if run.spec.mode == RunMode::Orchestra {
            let root = run.spec.root().map(|a| a.id.clone()).unwrap_or_default();
            let target = agent.filter(|a| run.spec.agent(a).is_some()).map(str::to_string).unwrap_or(root);
            meta["agent"] = json!(target);
            // Reopening the target (and its chain of command, if it already
            // reported) is `orchestra_turn`'s job: it re-checks every turn,
            // so this works whether the run is active or idle. Here we only
            // need to get a finished run out of `Done` so the loop runs at
            // all.
            if state.phase == Phase::Done {
                state.phase = Phase::Act;
            }
        } else if state.phase == Phase::Done {
            state.phase = Phase::Act;
            state.reviews = 0;
        }
        // Human guidance resets the failure streak.
        state.failures = 0;
        state.escalate = false;
        // No commit: the loop may be mid-tool in this worktree. The step
        // points at the last committed state.
        let commit = GitRepo::new(&run.worktree).head().await?.unwrap_or(run.base_commit.clone());
        let step = ws
            .store
            .append_step(
                run_id,
                NewStep {
                    kind: StepKind::Inject,
                    output: text,
                    observation: None,
                    meta,
                    state: &state,
                    git_commit: &commit,
                    cost_usd: 0.0,
                },
            )
            .await?;
        self.inner.emit(Event::Step { step: step.clone() });
        Ok(step)
    }

    /// Resets the run to just after step `seq` (or to its starting point
    /// for `seq < 0`). The pre-rewind tree is committed and kept under
    /// `refs/orchopork/rewound/<run>/<time>`, so nothing is unrecoverable.
    pub async fn rewind(&self, run_id: &str, seq: i64) -> Result<Run> {
        if self.is_active(run_id) {
            return Err(Error::Conflict("pause the run before rewinding it".into()));
        }
        let ws = &self.inner.ws;
        let _g = self.inner.append_lock.lock().await;
        let run = ws.store.get_run(run_id).await?;
        let target = if seq < 0 { run.base_commit.clone() } else { ws.store.step(run_id, seq).await?.git_commit };
        let wt = GitRepo::new(&run.worktree);
        let safety = wt.commit_all("orchopork: pre-rewind safety").await?;
        wt.update_ref(&format!("refs/orchopork/rewound/{run_id}/{}", now_unix()), &safety).await?;
        wt.reset_hard(&target).await?;
        ws.store.truncate_steps_after(run_id, seq.max(-1)).await?;
        let run = ws.store.set_run_status(run_id, RunStatus::Paused, None).await?;
        self.inner.emit(Event::Rewound { run_id: run_id.to_string(), seq });
        self.inner.emit(Event::Run { run: run.clone() });
        Ok(run)
    }

    /// Removes the run, its worktree and its branch. Spend stays recorded.
    pub async fn delete(&self, run_id: &str) -> Result<()> {
        if self.is_active(run_id) {
            return Err(Error::Conflict("pause the run before deleting it".into()));
        }
        let ws = &self.inner.ws;
        let run = ws.store.get_run(run_id).await?;
        ws.repo.worktree_remove(Path::new(&run.worktree), &run.branch).await?;
        ws.store.delete_run(run_id).await?;
        self.inner.emit(Event::Deleted { run_id: run_id.to_string() });
        Ok(())
    }

    /// (stat, patch) of everything the run changed relative to its base.
    pub async fn diff(&self, run_id: &str) -> Result<(String, String)> {
        let run = self.inner.ws.store.get_run(run_id).await?;
        GitRepo::new(&run.worktree).diff_from(&run.base_commit).await
    }

    /// Pushes the run's branch to `origin`.
    pub async fn push(&self, run_id: &str) -> Result<String> {
        let ws = &self.inner.ws;
        let run = ws.store.get_run(run_id).await?;
        if ws.repo.remote_url("origin").await.is_none() {
            return Err(Error::InvalidRequest("the repository has no `origin` remote".into()));
        }
        let token = match ws.config().remote_auth {
            RemoteAuth::Pat => Some(ws.secrets.get(crate::secrets::GITHUB).ok_or_else(|| {
                Error::InvalidRequest("remote auth is set to PAT but no GitHub token is stored".into())
            })?),
            _ => None,
        };
        GitRepo::new(&run.worktree).push(&run.branch, token.as_deref()).await
    }
}

impl Inner {
    fn emit(&self, e: Event) {
        let _ = self.events.send(e); // no subscribers is fine
    }

    fn log(&self, run_id: &str, message: impl Into<String>) {
        let message = message.into();
        tracing::info!(run = run_id, "{message}");
        self.emit(Event::Log { run_id: run_id.to_string(), message });
    }

    async fn drive(self: Arc<Self>, run_id: String, pause: Arc<AtomicBool>) {
        let (status, error) = match self.drive_loop(&run_id, &pause).await {
            Ok(Stop::Done) => (RunStatus::Done, None),
            Ok(Stop::Paused(msg)) => (RunStatus::Paused, msg),
            Err(e) => (RunStatus::Paused, Some(e.to_string())),
        };
        // Status before releasing the slot: a resume that sneaks in between
        // is refused ("already executing") rather than overwritten.
        match self.ws.store.set_run_status(&run_id, status, error.as_deref()).await {
            Ok(run) => self.emit(Event::Run { run }),
            Err(e) => tracing::error!(run = run_id, "failed to record run status: {e}"),
        }
        self.active.lock().unwrap().remove(&run_id);
    }

    async fn drive_loop(&self, run_id: &str, pause: &AtomicBool) -> Result<Stop> {
        let mut taken = 0u32;
        loop {
            if pause.load(Ordering::SeqCst) {
                return Ok(Stop::Paused(None));
            }
            let run = self.ws.store.get_run(run_id).await?;
            let cfg = effective_config(self.ws.config(), &run.spec);
            if taken >= cfg.limits.steps_per_session {
                return Ok(Stop::Paused(Some(format!(
                    "paused after {taken} steps (limits.steps_per_session); review progress and resume to continue"
                ))));
            }
            let steps = self.ws.store.steps(run_id).await?;
            let mut state = steps.last().map(|s| s.state.clone()).unwrap_or_default();
            // Solo and orchestra runs skip the planner, and so does a classic
            // run whose planner would be an external agent (it plans itself).
            let planner_is_external = cfg
                .routing
                .planner
                .as_ref()
                .or(cfg.routing.actor.as_ref())
                .is_some_and(|m| m.provider == ProviderId::Acp);
            if steps.is_empty() && (run.spec.mode != RunMode::Classic || planner_is_external) {
                state.phase = Phase::Act;
            }
            if run.spec.mode == RunMode::Orchestra {
                if state.phase == Phase::Done {
                    return Ok(Stop::Done);
                }
                taken += 1;
                if let Some(stop) = self.orchestra_turn(&cfg, &run, &steps, state).await? {
                    return Ok(stop);
                }
                continue;
            }
            let stop = match state.phase {
                Phase::Plan => self.plan(&cfg, &run, &steps, state).await?,
                Phase::Act => self.act(&cfg, &run, &steps, state).await?,
                Phase::Verify => self.verify(&cfg, &run, state).await?,
                Phase::Review => self.review(&cfg, &run, &steps, state).await?,
                Phase::Done => return Ok(Stop::Done),
            };
            taken += 1;
            if let Some(stop) = stop {
                return Ok(stop);
            }
        }
    }

    /// One LLM call. A cloud call refused by the budget guard falls back to
    /// the actor model when that one is local.
    async fn call(
        &self,
        cfg: &Config,
        run: &Run,
        model: &ModelRef,
        system: String,
        messages: Vec<Message>,
    ) -> Result<CallOutcome> {
        self.call_req(cfg, run, model, CompletionRequest { system, messages, ..Default::default() }).await
    }

    /// `call` with a volatile tail and cache options (see `CompletionRequest`).
    async fn call_req(&self, cfg: &Config, run: &Run, model: &ModelRef, req: CompletionRequest) -> Result<CallOutcome> {
        let req = CompletionRequest { max_tokens: cfg.limits.max_output_tokens, ..req };
        match self.ws.gateway.complete(model, &req, &run.id).await {
            Err(Error::Budget(msg)) => {
                match cfg.routing.actor.as_ref().filter(|a| a.provider.is_local() && *a != model) {
                    Some(local) => {
                        self.log(&run.id, format!("{msg}; falling back to local model {local}"));
                        self.ws.gateway.complete(local, &req, &run.id).await
                    }
                    None => Err(Error::Budget(msg)),
                }
            }
            other => other,
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn append(
        &self,
        run: &Run,
        kind: StepKind,
        output: &str,
        observation: Option<&str>,
        meta: serde_json::Value,
        state: &RunState,
        cost_usd: f64,
    ) -> Result<()> {
        let _g = self.append_lock.lock().await;
        let label = meta["tool"].as_str().map(|t| format!(" {t}")).unwrap_or_default();
        let commit =
            GitRepo::new(&run.worktree).commit_all(&format!("orchopork {}: {}{label}", run.id, kind.as_str())).await?;
        let observation = observation.map(|o| clip(o, MAX_STORED_OUTPUT));
        let step = self
            .ws
            .store
            .append_step(
                &run.id,
                NewStep {
                    kind,
                    output,
                    observation: observation.as_deref(),
                    meta,
                    state,
                    git_commit: &commit,
                    cost_usd,
                },
            )
            .await?;
        self.emit(Event::Step { step });
        Ok(())
    }

    async fn plan(&self, cfg: &Config, run: &Run, steps: &[Step], state: RunState) -> Result<Option<Stop>> {
        let model = cfg.routing.planner.clone().or_else(|| cfg.routing.actor.clone()).ok_or_else(no_actor)?;
        let wt = GitRepo::new(&run.worktree);
        let files = wt.ls_files().await?;
        let mut docs = Vec::new();
        for name in ["README.md", "PROJECT_CONTEXT.md", "STATE.md"] {
            if let Ok(body) = tokio::fs::read_to_string(Path::new(&run.worktree).join(name)).await {
                docs.push((name.to_string(), body));
            }
        }
        let notes: Vec<&str> = steps.iter().filter(|s| s.kind == StepKind::Inject).map(|s| s.output.as_str()).collect();
        let system = self.ws.skills.compose(prompts::PLANNER)?.system;
        let input = prompts::planner_input(&run.goal, &notes, &files, &docs);
        let out = self.call(cfg, run, &model, system, vec![Message::user(input)]).await?;
        let next = RunState { phase: Phase::Act, ..state };
        self.append(run, StepKind::Plan, &out.text, None, call_meta(&out), &next, out.cost_usd).await?;
        Ok(None)
    }

    async fn act(&self, cfg: &Config, run: &Run, steps: &[Step], state: RunState) -> Result<Option<Stop>> {
        let actor = cfg.routing.actor.clone().ok_or_else(no_actor)?;
        let (model, escalated) = match (&cfg.routing.escalation, state.escalate) {
            (Some(e), true) => (e.clone(), true),
            _ => (actor, false),
        };
        if model.provider == ProviderId::Acp {
            return self.act_external(cfg, run, steps, state, &model).await;
        }
        let skills = self.ws.skills.compose("")?;
        let toolbox = ToolBox::new(Path::new(&run.worktree), &cfg.limits, skills.tools.clone())?;
        let system =
            format!("{}\n\n{}", prompts::actor_system(&toolbox.describe()), skills.system).trim_end().to_string();
        // Files as of the run's start (stable, cacheable) plus what changed
        // since then (volatile, sent after the cache breakpoint).
        let wt = GitRepo::new(&run.worktree);
        let files = wt.ls_files_at(&run.base_commit).await?;
        let tail = prompts::changes_note(&wt.changed_names(&run.base_commit).await?);
        let plan = steps.iter().rev().find(|s| s.kind == StepKind::Plan).map(|s| s.output.as_str()).unwrap_or("");
        let messages = prompts::actor_messages(
            &run.goal,
            plan,
            &files,
            steps,
            cfg.context_chars(&model),
            cfg.limits.tool_output_chars,
        );
        if escalated {
            self.log(&run.id, format!("escalating this turn to {model} after {} failed turns", state.failures));
        }
        let req = CompletionRequest { system, messages, tail: Some(tail), ..Default::default() };
        let out = self.call_req(cfg, run, &model, req).await?;

        let mut meta = call_meta(&out);
        meta["escalated"] = json!(escalated);
        let mut next = state.clone();
        let violations = skills.check_output(&out.text);
        let (observation, ok) = if !violations.is_empty() {
            let list: Vec<String> = violations.iter().map(|v| format!("{} (reply {})", v.skill, v.detail)).collect();
            (
                format!(
                    "Your reply was rejected by validator skills: {}. Redo the step without that.",
                    list.join("; ")
                ),
                false,
            )
        } else {
            match protocol::parse_action(&out.text) {
                Err(e) => {
                    let cut = if out.truncated {
                        " Your reply hit the output limit and was cut off; write large files in smaller parts (write_file a skeleton, then edit_file)."
                    } else {
                        ""
                    };
                    let msg = format!(
                        "Could not parse an action: {e}.{cut} Reply with a short note and exactly one JSON object, e.g. {{\"tool\": \"read_file\", \"args\": {{\"path\": \"README.md\"}}}}."
                    );
                    (msg, false)
                }
                Ok(a) if a.tool == "finish" => {
                    meta["tool"] = json!("finish");
                    next.phase = if !verify_commands(run, &skills).is_empty() {
                        Phase::Verify
                    } else if cfg.routing.critic.is_some() {
                        Phase::Review
                    } else {
                        Phase::Done
                    };
                    let msg = match next.phase {
                        Phase::Verify => "Finish requested; running verification.",
                        Phase::Review => "Finish requested; sending the change to review.",
                        _ => "Finished.",
                    };
                    (msg.to_string(), true)
                }
                Ok(a) => {
                    meta["tool"] = json!(a.tool);
                    let r = toolbox.execute(&a).await;
                    (r.output, r.ok)
                }
            }
        };
        meta["ok"] = json!(ok);
        if ok {
            next.failures = 0;
            next.escalate = false;
        } else {
            next.failures += 1;
            next.escalate = cfg.routing.escalation.is_some() && next.failures >= cfg.limits.escalate_after;
        }
        self.append(run, StepKind::Act, &out.text, Some(&observation), meta, &next, out.cost_usd).await?;
        if next.failures >= cfg.limits.max_failures {
            return Ok(Some(Stop::Paused(Some(format!(
                "{} consecutive failed turns; inject guidance or pick a stronger actor model, then resume",
                next.failures
            )))));
        }
        Ok(None)
    }

    async fn verify(&self, cfg: &Config, run: &Run, state: RunState) -> Result<Option<Stop>> {
        let (all_ok, summary, details) = self.run_verification(cfg, run).await?;
        let next = RunState {
            phase: match (all_ok, cfg.routing.critic.is_some()) {
                (false, _) => Phase::Act,
                (true, true) => Phase::Review,
                (true, false) => Phase::Done,
            },
            failures: 0,
            escalate: false,
            ..state
        };
        let meta = json!({ "ok": all_ok });
        self.append(run, StepKind::Verify, &summary, Some(&details), meta, &next, 0.0).await?;
        Ok(None)
    }

    /// Runs every verification command: (all passed, summary, details).
    async fn run_verification(&self, cfg: &Config, run: &Run) -> Result<(bool, String, String)> {
        let skills = self.ws.skills.compose("")?;
        let timeout = Duration::from_secs(cfg.limits.command_timeout_secs.max(1));
        let mut summary = Vec::new();
        let mut details = String::new();
        let mut all_ok = true;
        let commands = verify_commands(run, &skills);
        let per_command = MAX_STORED_OUTPUT / commands.len().max(1);
        for (label, cmd) in &commands {
            let o = run_shell(Path::new(&run.worktree), cmd, &[], timeout).await;
            let passed = o.success();
            all_ok &= passed;
            let status = match (o.timed_out, o.exit_code) {
                (true, _) => "timed out".to_string(),
                (_, Some(c)) => format!("exit {c}"),
                _ => "killed".to_string(),
            };
            summary.push(format!("{} [{label}] {cmd} ({status})", if passed { "PASS" } else { "FAIL" }));
            details.push_str(&format!("$ {cmd}\n[{status}]\n{}\n\n", clip(&o.output, per_command)));
        }
        if commands.is_empty() {
            summary.push("no verification configured".into());
        }
        Ok((all_ok, summary.join("\n"), details.trim_end().to_string()))
    }

    async fn review(&self, cfg: &Config, run: &Run, steps: &[Step], state: RunState) -> Result<Option<Stop>> {
        let Some(model) = cfg.routing.critic.clone() else {
            let next = RunState { phase: Phase::Done, ..state };
            let meta = json!({ "verdict": "approve" });
            self.append(run, StepKind::Review, "No critic configured; accepted.", None, meta, &next, 0.0).await?;
            return Ok(None);
        };
        let (stat, patch) = GitRepo::new(&run.worktree).diff_from(&run.base_commit).await?;
        let plan = steps.iter().rev().find(|s| s.kind == StepKind::Plan).map(|s| s.output.as_str()).unwrap_or("");
        let summary = steps
            .iter()
            .rev()
            .find(|s| s.kind == StepKind::Act && s.meta["tool"] == "finish")
            .map(|s| s.output.as_str())
            .unwrap_or("");
        let verification =
            steps.iter().rev().find(|s| s.kind == StepKind::Verify).and_then(|s| s.observation.as_deref());
        let input =
            prompts::critic_input(&run.goal, plan, summary, verification, &stat, &patch, cfg.context_chars(&model));
        let system = self.ws.skills.compose(prompts::CRITIC)?.system;
        let out = self.call(cfg, run, &model, system, vec![Message::user(input)]).await?;
        let (approve, feedback) = protocol::parse_verdict(&out.text);
        let mut next = state.clone();
        if approve {
            next.phase = Phase::Done;
        } else {
            next.phase = Phase::Act;
            next.reviews += 1;
            next.failures = 0;
        }
        let mut meta = call_meta(&out);
        meta["verdict"] = json!(if approve { "approve" } else { "revise" });
        let output = match (feedback.is_empty(), approve) {
            (false, _) => feedback,
            (true, true) => "Approved.".to_string(),
            (true, false) => out.text.clone(),
        };
        self.append(run, StepKind::Review, &output, None, meta, &next, out.cost_usd).await?;
        if !approve && next.reviews >= cfg.limits.max_review_rounds {
            return Ok(Some(Stop::Paused(Some(format!(
                "the reviewer still requests changes after {} rounds; inspect the diff, inject guidance and resume",
                next.reviews
            )))));
        }
        Ok(None)
    }
}

/// The workspace config as a given run sees it: a solo run uses its own
/// model for everything and has no planner or critic.
fn effective_config(mut cfg: Config, spec: &RunSpec) -> Config {
    // External agents cannot plan or review through the gateway.
    for slot in [&mut cfg.routing.planner, &mut cfg.routing.critic] {
        if slot.as_ref().is_some_and(|m| m.provider == ProviderId::Acp) {
            *slot = None;
        }
    }
    match spec.mode {
        RunMode::Classic => {}
        RunMode::Solo => {
            cfg.routing.actor = spec.model.clone();
            cfg.routing.planner = None;
            cfg.routing.critic = None;
        }
        RunMode::Orchestra => {
            // Budget fallback in `call` targets the local default, if any.
            cfg.routing.critic = None;
            cfg.routing.planner = None;
        }
    }
    cfg
}

fn no_actor() -> Error {
    Error::InvalidRequest("no actor model configured".into())
}

fn call_meta(out: &CallOutcome) -> serde_json::Value {
    json!({
        "model": out.model.to_string(),
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.completion_tokens,
        "cache_read_tokens": out.cache_read_tokens,
        "cache_write_tokens": out.cache_write_tokens,
        "truncated": out.truncated,
    })
}

/// (label, command) pairs gating `finish`: the run's own verify command,
/// then every enabled validator skill's command.
fn verify_commands(run: &Run, skills: &crate::skills::ComposedPrompt) -> Vec<(String, String)> {
    run.verify_command.iter().map(|c| ("run".to_string(), c.clone())).chain(skills.verify_commands()).collect()
}

#[cfg(test)]
mod tests;
