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

pub mod approvals;
pub mod ask;
pub mod browser;
pub mod screen;
pub mod external;
pub mod flow;
mod parallel;
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
use crate::storage::{Autonomy, NewStep, Phase, Run, RunMode, RunSpec, RunState, RunStatus, Step, StepKind, now_unix};
use approvals::{Approvals, Decision};
use protocol::Action;
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
    /// A tool call waits for the user (`approval`), or no longer does (`None`).
    Approval { run_id: String, approval: Option<approvals::Approval> },
    /// A model's reply while it is being written (the latest part of it).
    Live { run_id: String, agent: Option<String>, text: String, thinking: bool },
}

enum Stop {
    Done,
    Paused(Option<String>),
    /// An agent asked the user something (see `ask`); the status line.
    Waiting(String),
}

tokio::task_local! {
    /// Set while one of a Dispatcher's branches takes its turns: steps then
    /// checkpoint the branch's worktree and state (see `parallel`).
    static BRANCH: Arc<parallel::BranchCtx>;
}

/// Whether the current turn belongs to a Dispatcher's branch.
fn in_branch() -> bool {
    BRANCH.try_with(|_| ()).is_ok()
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Inner>,
}

struct Inner {
    /// Itself, for work spawned as tasks of its own (parallel branches).
    this: std::sync::Weak<Inner>,
    ws: Arc<Workspace>,
    /// Pause flags of runs whose loop is currently executing.
    active: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// The tasks driving active runs, so a run can be killed mid-step.
    tasks: Mutex<HashMap<String, tokio::task::AbortHandle>>,
    events: broadcast::Sender<Event>,
    /// Serializes step appends, so an inject arriving mid-run cannot race
    /// the loop for the next sequence number or a git commit.
    append_lock: tokio::sync::Mutex<()>,
    approvals: Arc<Approvals>,
    /// Per run: one approval request at a time (parallel branches may ask
    /// together; the dashboard shows one).
    approval_gates: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// When each run's agents last finished a step (`run/agent`, and
    /// `run/` for the run itself), so every step records how long it took.
    clocks: Mutex<HashMap<String, std::time::Instant>>,
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
            inner: Arc::new_cyclic(|this| Inner {
                this: this.clone(),
                ws,
                active: Mutex::new(HashMap::new()),
                tasks: Mutex::new(HashMap::new()),
                approvals: Arc::new(Approvals::new(events.clone())),
                events,
                append_lock: tokio::sync::Mutex::new(()),
                approval_gates: Mutex::new(HashMap::new()),
                clocks: Mutex::new(HashMap::new()),
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
        self.create_run_at(goal, verify_command, spec, None).await
    }

    /// `create_run` starting from `base` (a commit, e.g. another run's
    /// latest state) instead of the repository's HEAD.
    pub async fn create_run_at(
        &self,
        goal: &str,
        verify_command: Option<&str>,
        mut spec: RunSpec,
        base: Option<&str>,
    ) -> Result<Run> {
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
                let mut g =
                    spec.graph.take().ok_or_else(|| Error::InvalidRequest("the orchestra has no flow graph".into()))?;
                g.normalize();
                flow::check_orchestra(&spec, &g, verify_command)?;
                crate::storage::flowgraph::auto_layout(&mut g);
                spec.graph = Some(g);
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
            let head = ws.repo.ensure_initial_commit().await?;
            let base = base.map(str::to_string).unwrap_or(head);
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
        // Held while spawning: a loop that ends at once removes its handle
        // only after it was stored, so no stale handle is left behind.
        let mut tasks = self.inner.tasks.lock().unwrap();
        let handle = tokio::spawn(async move { inner.drive(id, flag).await });
        tasks.insert(run_id.to_string(), handle.abort_handle());
        drop(tasks);
        Ok(run)
    }

    /// Stops a run immediately, in the middle of whatever it is doing: the
    /// model call or tool in flight is dropped (child processes die with
    /// it), background processes are stopped and a pending approval is
    /// withdrawn. Everything up to the last completed step is kept.
    pub async fn kill(&self, run_id: &str) -> Result<Run> {
        let store = &self.inner.ws.store;
        let run = store.get_run(run_id).await?;
        let was_active = match self.inner.active.lock().unwrap().get(run_id) {
            Some(flag) => {
                flag.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        };
        // Nothing to kill: a finished (or already stopped) run keeps its state.
        if !was_active && run.status != RunStatus::Running {
            return Ok(run);
        }
        if let Some(h) = self.inner.tasks.lock().unwrap().remove(run_id) {
            h.abort();
        }
        self.inner.active.lock().unwrap().remove(run_id);
        self.inner.approvals.forget(run_id);
        self.inner.approvals.withdraw(run_id);
        if let Ok(root) = crate::fsutil::canonical(Path::new(&run.worktree)) {
            tools::processes::stop_all(&root);
        }
        let run = store.set_run_status(run_id, RunStatus::Paused, Some("killed by the user")).await?;
        self.inner.emit(Event::Run { run: run.clone() });
        self.inner.log(run_id, "killed");
        Ok(run)
    }

    /// Stops a run for good: pauses it (after the step in progress, if it
    /// is executing) and records `why` as its note.
    pub async fn cancel(&self, run_id: &str, why: &str) -> Result<()> {
        let store = self.inner.ws.store.clone();
        let run = store.get_run(run_id).await?;
        if self.is_active(run_id) {
            self.pause(run_id).await?;
            let (engine, id, why) = (self.clone(), run_id.to_string(), why.to_string());
            tokio::spawn(async move {
                for _ in 0..3600 {
                    if !engine.is_active(&id) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                if let Ok(run) = engine.inner.ws.store.set_run_status(&id, RunStatus::Paused, Some(&why)).await {
                    engine.inner.emit(Event::Run { run });
                }
            });
        } else if run.status != RunStatus::Done {
            let run = store.set_run_status(run_id, RunStatus::Paused, Some(why)).await?;
            self.inner.emit(Event::Run { run });
        }
        Ok(())
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
        self.inject_meta(run_id, text, agent, json!({})).await
    }

    /// Answers the question the run is waiting on (step `seq`) and
    /// continues the run.
    pub async fn answer(&self, run_id: &str, seq: i64, text: &str) -> Result<Run> {
        let (_, steps) = self.run_detail(run_id).await?;
        let q = ask::waiting_on(&steps)
            .filter(|q| q.seq == seq)
            .ok_or_else(|| Error::Conflict("that question was already answered or is no longer open".into()))?;
        let agent = q.meta["agent"].as_str().map(str::to_string);
        self.inject_meta(run_id, text, agent.as_deref(), json!({ "answer_to": seq })).await?;
        if self.is_active(run_id) {
            return self.inner.ws.store.get_run(run_id).await;
        }
        self.start(run_id).await
    }

    /// `inject`, with `extra` merged into the step's metadata.
    async fn inject_meta(&self, run_id: &str, text: &str, agent: Option<&str>, extra: serde_json::Value) -> Result<Step> {
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
            let g = flow::current_graph(&run, &state).ok_or_else(|| {
                Error::Conflict("this run uses the removed hierarchical orchestra and is read-only".into())
            })?;
            let lead = g.lead().map(|n| n.id.clone()).unwrap_or_default();
            let target = agent
                .filter(|a| g.node(a).is_some_and(|n| n.is_worker()))
                .map(str::to_string)
                .unwrap_or(lead);
            meta["agent"] = json!(target);
            // Reopening the target (and everything after it, if it already
            // finished) is `flow_turn`'s job: it re-checks every turn, so
            // this works whether the run is active or idle. Here we only
            // need to get a finished run out of `Done` so the loop runs at
            // all.
            if state.phase == Phase::Done {
                state.phase = Phase::Act;
            }
        } else if state.phase == Phase::Done {
            state.phase = Phase::Act;
            state.reviews = 0;
        }
        for (k, v) in extra.as_object().into_iter().flatten() {
            meta[k] = v.clone();
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

    /// The tool call this run is waiting on the user for, if any.
    pub fn pending_approval(&self, run_id: &str) -> Option<approvals::Approval> {
        self.inner.approvals.pending(run_id)
    }

    /// Allows or denies the pending call `id`. `always` also allows every
    /// later call of the same tool in this run without asking.
    pub fn decide(&self, run_id: &str, id: &str, allow: bool, always: bool, note: &str) -> Result<()> {
        self.inner.approvals.decide(run_id, id, allow, always, note)?;
        self.inner.log(run_id, if allow { "approved; running the call" } else { "declined by the user" });
        Ok(())
    }

    /// Changes how much the run may do without asking; takes effect on its
    /// next tool call (and releases a pending one the new mode allows).
    pub async fn set_autonomy(&self, run_id: &str, autonomy: Autonomy) -> Result<Run> {
        let store = &self.inner.ws.store;
        let run = store.get_run(run_id).await?;
        let spec = RunSpec { autonomy, ..run.spec.clone() };
        store.set_run_spec(run_id, &spec).await?;
        self.inner.approvals.reconsider(run_id, autonomy);
        let run = store.get_run(run_id).await?;
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
        if let Ok(root) = crate::fsutil::canonical(Path::new(&run.worktree)) {
            tools::processes::stop_all(&root);
        }
        self.inner.approvals.forget(run_id);
        self.inner.approval_gates.lock().unwrap().remove(run_id);
        parallel::remove_branch_worktrees(ws, &run).await;
        ws.repo.worktree_remove(Path::new(&run.worktree), &run.branch).await?;
        ws.store.delete_run(run_id).await?;
        self.inner.emit(Event::Deleted { run_id: run_id.to_string() });
        Ok(())
    }

    /// (stat, patch) of what one step changed: its checkpoint against the
    /// one before it.
    pub async fn step_diff(&self, run_id: &str, seq: i64) -> Result<(String, String)> {
        let ws = &self.inner.ws;
        let run = ws.store.get_run(run_id).await?;
        let steps = ws.store.steps(run_id).await?;
        let i = steps.iter().position(|s| s.seq == seq).ok_or_else(|| Error::NotFound(format!("step {seq}")))?;
        let b = &steps[i].meta["branch"];
        if let Some(after) = b["commit"].as_str() {
            // A parallel branch's step: against the branch's previous one.
            let before = steps[..i]
                .iter()
                .rev()
                .find(|s| s.meta["branch"]["head"] == b["head"])
                .and_then(|s| s.meta["branch"]["commit"].as_str())
                .or(b["fork"].as_str())
                .unwrap_or(&run.base_commit)
                .to_string();
            return GitRepo::new(&run.worktree).diff_between(&before, after).await;
        }
        let before = if i == 0 { run.base_commit.clone() } else { steps[i - 1].git_commit.clone() };
        GitRepo::new(&run.worktree).diff_between(&before, &steps[i].git_commit).await
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

    /// Merges the run's branch into the workspace's active checkout.
    pub async fn merge(&self, run_id: &str) -> Result<String> {
        let ws = &self.inner.ws;
        let run = ws.store.get_run(run_id).await?;
        let msg = format!("orchopork: merge run {} ({})", run.id, run.goal);
        ws.repo.merge(&run.branch, &msg).await
    }
}

impl Inner {
    pub(super) fn emit(&self, e: Event) {
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
            Ok(Stop::Waiting(msg)) => (RunStatus::Waiting, Some(msg)),
            Err(e) => (RunStatus::Paused, Some(e.to_string())),
        };
        // Status before releasing the slot: a resume that sneaks in between
        // is refused ("already executing") rather than overwritten.
        match self.ws.store.set_run_status(&run_id, status, error.as_deref()).await {
            Ok(run) => self.emit(Event::Run { run }),
            Err(e) => tracing::error!(run = run_id, "failed to record run status: {e}"),
        }
        self.active.lock().unwrap().remove(&run_id);
        self.tasks.lock().unwrap().remove(&run_id);
    }

    async fn drive_loop(&self, run_id: &str, pause: &AtomicBool) -> Result<Stop> {
        let mut taken = 0u32;
        // Time spent paused is not part of any step.
        {
            let prefix = format!("{run_id}/");
            let mut clocks = self.clocks.lock().unwrap();
            clocks.retain(|k, _| !k.starts_with(&prefix));
            clocks.insert(prefix, std::time::Instant::now());
        }
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
                if let Some(stop) = self.flow_turn(&cfg, &run, &steps, state).await? {
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
        // How long the step took: since this agent's previous step (or the
        // run's start or resume). Messages from the user take no time.
        let mut meta = meta;
        if kind != StepKind::Inject {
            let now = std::time::Instant::now();
            let agent = format!("{}/{}", run.id, meta["agent"].as_str().unwrap_or(""));
            let mut clocks = self.clocks.lock().unwrap();
            let since = clocks.get(&agent).or_else(|| clocks.get(&format!("{}/", run.id))).copied();
            if let Some(t) = since
                && meta.get("ms").is_none()
            {
                meta["ms"] = json!(now.duration_since(t).as_millis() as u64);
            }
            clocks.insert(agent, now);
        }
        let commit =
            GitRepo::new(&run.worktree).commit_all(&format!("orchopork {}: {}{label}", run.id, kind.as_str())).await?;
        // A branch's step checkpoints the branch worktree and the branch's
        // state inside the run's; the step itself points at the run's
        // worktree, which the branches do not touch.
        let branch = BRANCH.try_with(Arc::clone).ok();
        let (state, commit) = match &branch {
            Some(b) => {
                let mut run_state = b.shared.lock().await;
                if let Some(br) = run_state.flow.parallel.as_mut().and_then(|p| p.branch_mut(&b.head)) {
                    br.commit = commit.clone();
                    br.state = Some(Box::new(state.clone()));
                }
                meta["branch"] = json!({ "head": b.head, "commit": commit, "fork": b.fork });
                (run_state.clone(), b.main_commit.clone())
            }
            None => (state.clone(), commit),
        };
        let (state, commit) = (&state, commit);
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

    /// The conversation for one turn of `key`'s agent, within `budget`
    /// chars. Turns that no longer fit are folded into the agent's memory
    /// summary (kept in `state`, so it checkpoints and rewinds with the
    /// run) instead of being dropped; if summarizing fails they are dropped
    /// as before. Returns the messages and what summarizing cost.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn fit(
        &self,
        cfg: &Config,
        run: &Run,
        key: &str,
        model: &ModelRef,
        pinned: prompts::Pinned,
        groups: Vec<prompts::Group>,
        budget: usize,
        state: &mut RunState,
    ) -> (Vec<Message>, f64) {
        let mut cost = 0.0;
        // Three folds at most: one normally suffices, more cover a
        // summary that itself grew too large.
        for _ in 0..3 {
            match prompts::window(pinned.clone(), groups.clone(), budget, state.memory.get(key)) {
                prompts::Window::Fits(m) => return (m, cost),
                prompts::Window::Fold { fold, upto } => {
                    let prev = state.memory.get(key).map(|m| m.text.clone()).unwrap_or_default();
                    let limit = (budget / 8).clamp(1_500, 12_000);
                    match self.summarize(cfg, run, model, &prev, &fold, limit).await {
                        Ok((text, c)) => {
                            cost += c;
                            self.log(&run.id, format!("{}: older turns folded into memory", if key.is_empty() { "agent" } else { key }));
                            state.memory.insert(key.to_string(), crate::storage::Memory { upto, text });
                        }
                        Err(e) => {
                            tracing::warn!(run = run.id, "could not summarize old turns ({e}); dropping them instead");
                            break;
                        }
                    }
                }
            }
        }
        let groups = groups.into_iter().filter(|(seq, _)| state.memory.get(key).is_none_or(|m| *seq > m.upto));
        (prompts::windowed(pinned, groups.map(|(_, g)| g).collect(), budget), cost)
    }

    /// Merges `prev` and `fold` into a summary of at most `limit` chars,
    /// asking for a shorter one if the first is too long.
    async fn summarize(
        &self,
        cfg: &Config,
        run: &Run,
        model: &ModelRef,
        prev: &str,
        fold: &[Vec<Message>],
        limit: usize,
    ) -> Result<(String, f64)> {
        let input = format!(
            "# Previous summary
{}

# New transcript to fold in
{}

Write the updated summary in at most {limit} characters.",
            if prev.trim().is_empty() { "(none)" } else { prev.trim() },
            prompts::render_groups(fold, 3_000)
        );
        let out = self.call(cfg, run, model, prompts::SUMMARIZER.into(), vec![Message::user(input)]).await?;
        let mut text = out.text.trim().to_string();
        let mut cost = out.cost_usd;
        if text.len() > limit {
            let again = format!("Shorten this summary to at most {limit} characters, keeping the most important facts:

{text}");
            let out = self.call(cfg, run, model, prompts::SUMMARIZER.into(), vec![Message::user(again)]).await?;
            cost += out.cost_usd;
            text = out.text.trim().to_string();
        }
        if text.is_empty() {
            return Err(Error::Provider("the summary came back empty".into()));
        }
        Ok((clip(&text, limit + 200), cost))
    }

    /// Records a user change (e.g. a flow edit) as a step without running
    /// anything; the worktree is committed as it is.
    pub(super) async fn record_edit(&self, run: &Run, text: &str, meta: serde_json::Value, state: &RunState) -> Result<()> {
        self.append(run, StepKind::Inject, text, None, meta, state, 0.0).await
    }

    /// Streams a reply to the dashboard as `Live` events, at most five a
    /// second, each with the latest part of what was written so far.
    pub(super) fn live_sink(&self, run_id: &str, agent: Option<&str>) -> crate::providers::Sink {
        struct Buf {
            text: String,
            thought: String,
            thinking: bool,
            last: std::time::Instant,
        }
        let buf = Arc::new(Mutex::new(Buf {
            text: String::new(),
            thought: String::new(),
            thinking: false,
            last: std::time::Instant::now() - Duration::from_secs(1),
        }));
        let (events, run_id, agent) = (self.events.clone(), run_id.to_string(), agent.map(str::to_string));
        crate::providers::Sink(Arc::new(move |piece: &str, thinking: bool| {
            let mut b = buf.lock().unwrap();
            if thinking { b.thought.push_str(piece) } else { b.text.push_str(piece) }
            b.thinking = thinking;
            if b.last.elapsed() < Duration::from_millis(200) {
                return;
            }
            b.last = std::time::Instant::now();
            let whole = if b.text.is_empty() { &b.thought } else { &b.text };
            let start = whole.len().saturating_sub(4_000);
            let start = (start..whole.len()).find(|&i| whole.is_char_boundary(i)).unwrap_or(whole.len());
            let _ = events.send(Event::Live {
                run_id: run_id.clone(),
                agent: agent.clone(),
                text: whole[start..].to_string(),
                thinking: b.text.is_empty(),
            });
        }))
    }

    fn pause_flag(&self, run_id: &str) -> Arc<AtomicBool> {
        self.active.lock().unwrap().get(run_id).cloned().unwrap_or_else(|| Arc::new(AtomicBool::new(false)))
    }

    /// Runs one tool call, first asking the user when the run's autonomy
    /// requires it. `ok` is `None` for a call the user declined (or that a
    /// pause interrupted): not the model's failure, so it does not count
    /// towards escalation.
    pub(super) async fn gated(
        &self,
        run: &Run,
        agent: Option<&str>,
        toolbox: &ToolBox,
        a: &Action,
    ) -> (String, Option<bool>) {
        let class = toolbox.class(&a.tool);
        // Fresh from the store: the user may have switched modes mid-run.
        let autonomy = self.ws.store.get_run(&run.id).await.map(|r| r.spec.autonomy).unwrap_or(run.spec.autonomy);
        if self.approvals.needed(&run.id, &a.tool, class, autonomy) {
            let gate = self.approval_gates.lock().unwrap().entry(run.id.clone()).or_default().clone();
            let _one = gate.lock().await;
            self.log(&run.id, format!("waiting for your approval: {}", a.tool));
            let flag = self.pause_flag(&run.id);
            let d = self
                .approvals
                .ask(&run.id, agent, &a.tool, serde_json::Value::Object(a.args.clone()), class, a.tool.clone(), &flag)
                .await;
            if !matches!(d, Decision::Allow) {
                return (approvals::refused_observation(&a.tool, &d), None);
            }
        }
        let r = toolbox.execute(a).await;
        (r.output, Some(r.ok))
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

    async fn act(&self, cfg: &Config, run: &Run, steps: &[Step], mut state: RunState) -> Result<Option<Stop>> {
        let actor = cfg.routing.actor.clone().ok_or_else(no_actor)?;
        let (model, escalated) = match (&cfg.routing.escalation, state.escalate) {
            (Some(e), true) => (e.clone(), true),
            _ => (actor, false),
        };
        if model.provider == ProviderId::Acp {
            return self.act_external(cfg, run, steps, state, &model).await;
        }
        let skills = self.ws.skills.compose("")?;
        let toolbox = ToolBox::new(Path::new(&run.worktree), &cfg.limits, skills.tools.clone())?
            .with_vision(self.ws.gateway.supports_vision(&model).await);
        let system =
            format!("{}\n\n{}", prompts::actor_system(&toolbox.describe_with_finish()), skills.system).trim_end().to_string();
        // Files as of the run's start (stable, cacheable) plus what changed
        // since then (volatile, sent after the cache breakpoint).
        let wt = GitRepo::new(&run.worktree);
        let files = wt.ls_files_at(&run.base_commit).await?;
        let tail = prompts::changes_note(&wt.changed_names(&run.base_commit).await?);
        let plan = steps.iter().rev().find(|s| s.kind == StepKind::Plan).map(|s| s.output.as_str()).unwrap_or("");
        let (messages, memory_cost) = self
            .fit(
                cfg,
                run,
                "",
                &model,
                prompts::actor_pinned(&run.goal, plan, &files),
                prompts::actor_groups(steps, cfg.limits.tool_output_chars),
                cfg.context_chars(&model),
                &mut state,
            )
            .await;
        if escalated {
            self.log(&run.id, format!("escalating this turn to {model} after {} failed turns", state.failures));
        }
        let messages = attach_latest_image(messages, steps, None, &toolbox);
        let req = CompletionRequest {
            system,
            messages,
            tail: Some(tail),
            tools: toolbox.schemas_with_finish(),
            sink: Some(self.live_sink(&run.id, None)),
            ..Default::default()
        };
        let out = self.call_req(cfg, run, &model, req).await?;

        let mut meta = call_meta(&out);
        meta["escalated"] = json!(escalated);
        let mut next = state.clone();
        let mut asked = None;
        let violations = skills.check_output(&out.text);
        let (observation, ok) = if !violations.is_empty() {
            let list: Vec<String> = violations.iter().map(|v| format!("{} (reply {})", v.skill, v.detail)).collect();
            (
                format!(
                    "Your reply was rejected by validator skills: {}. Redo the step without that.",
                    list.join("; ")
                ),
                Some(false),
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
                    (msg, Some(false))
                }
                Ok(a) if a.tool == "ask" => {
                    meta["tool"] = json!("ask");
                    match ask::from_args(&a.args) {
                        Some(q) => {
                            for (k, v) in q.meta().as_object().into_iter().flatten() {
                                meta[k] = v.clone();
                            }
                            asked = Some(q.status("The agent"));
                            ("Waiting for the user's answer.".to_string(), Some(true))
                        }
                        None => ("ask needs a non-empty \"question\"".to_string(), Some(false)),
                    }
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
                    (msg.to_string(), Some(true))
                }
                Ok(a) => {
                    meta["tool"] = json!(a.tool);
                    if let Some(why) = repeated_call(steps, None, &a) {
                        meta["repeat"] = json!(true);
                        (why, Some(false))
                    } else {
                        let (obs, ok) = self.gated(run, None, &toolbox, &a).await;
                        if ok.is_none() {
                            meta["declined"] = json!(true);
                        }
                        note_images(&mut meta, image_of(&a, &obs, ok));
                        note_lines(&mut meta, &obs, ok);
                        let stuck = stuck_hint(steps, None, &obs, ok, "web_search the exact error message");
                        // Being stuck is a failure even when the call itself
                        // worked: it moves the run to escalation or a pause.
                        let ok = if stuck.is_empty() { ok } else { Some(false) };
                        (format!("{obs}{stuck}"), ok)
                    }
                }
            }
        };
        count_outcome(cfg, &mut meta, &mut next, ok);
        let cost = out.cost_usd + memory_cost;
        self.append(run, StepKind::Act, &out.text, Some(&observation), meta, &next, cost).await?;
        if let Some(status) = asked {
            return Ok(Some(Stop::Waiting(status)));
        }
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

/// The images a tool call produced or asked to see (repo-relative paths).
pub(super) fn image_of(a: &Action, obs: &str, ok: Option<bool>) -> Vec<String> {
    if ok != Some(true) {
        return vec![];
    }
    match a.tool.as_str() {
        "view_image" => tools::image_args(a),
        "browser" => a.str_arg("screenshot").map(str::to_string).into_iter().collect(),
        "screenshot_page" | "screenshot_screen" | "screenshot_sequence" => {
            // Every saved path in the result, in order.
            let head = obs.split('\n').next().unwrap_or("");
            let mut out = Vec::new();
            let mut rest = head;
            while let Some(start) = rest.find(tools::SHOTS_DIR) {
                let Some(len) = rest[start..].find(".png") else { break };
                out.push(rest[start..start + len + 4].to_string());
                rest = &rest[start + len + 4..];
            }
            out
        }
        _ => vec![],
    }
}

/// A canonical key for a tool call: its name and arguments.
pub(super) fn call_key(a: &Action) -> String {
    let sorted: std::collections::BTreeMap<&String, &serde_json::Value> = a.args.iter().collect();
    format!("{}:{}", a.tool, serde_json::to_string(&sorted).unwrap_or_default())
}

/// `agent`'s act steps (all act steps when `agent` is `None`).
fn acts_of<'a>(steps: &'a [Step], agent: Option<&'a str>) -> impl DoubleEndedIterator<Item = &'a Step> + 'a {
    steps.iter().filter(move |s| s.kind == StepKind::Act && agent.is_none_or(|a| s.meta["agent"] == a))
}

/// When `agent` already made exactly this call and nothing in the worktree
/// changed since, the answer would be the same: say so instead of running it.
pub(super) fn repeated_call(steps: &[Step], agent: Option<&str>, a: &Action) -> Option<String> {
    if matches!(a.tool.as_str(), "finish" | "process_output" | "notebook_read") {
        return None;
    }
    let key = call_key(a);
    let same = |s: &&Step| protocol::parse_action(&s.output).is_ok_and(|p| call_key(&p) == key);
    let prev = acts_of(steps, agent).rev().find(same)?;
    let head = steps.last().map(|s| s.git_commit.as_str()).unwrap_or("");
    if prev.git_commit != head {
        return None; // files changed since: the result may differ
    }
    // Other agents, processes and user messages change the world without
    // touching files (e.g. a dev server was started): then it is no repeat.
    let changed = steps.iter().filter(|s| s.seq > prev.seq).any(|s| {
        s.kind != StepKind::Act
            || agent.is_some_and(|me| s.meta["agent"] != me)
            || matches!(s.meta["tool"].as_str(), Some("start_process" | "stop_process" | "run_command"))
    });
    if changed {
        return None;
    }
    // Quote the result of the last call that really ran, not an earlier
    // "Not run" notice (those would nest).
    let original = acts_of(steps, agent).rev().filter(same).find(|s| s.meta["repeat"] != true).unwrap_or(prev);
    let times = acts_of(steps, agent).filter(same).count() + 1;
    Some(format!(
        "Not run (attempt {times} of the same call): nothing has changed since you last made it, so the result would be the same:\n{}\n\n\
         Do something different: change the code, start what is missing, or get help (consult a teammate, or web_search the error).",
        clip(original.observation.as_deref().unwrap_or(""), 800)
    ))
}

/// What identifies an error in a result: the failure itself, or the page
/// errors a browser check reported.
pub(super) fn error_sig(obs: &str, ok: Option<bool>) -> Option<String> {
    let page: Vec<&str> = obs.lines().filter(|l| l.contains("Uncaught") || l.contains("Failed to load resource")).collect();
    if !page.is_empty() {
        return Some(page.join("\n").chars().take(300).collect());
    }
    if ok == Some(false) {
        return Some(obs.lines().find(|l| !l.trim().is_empty() && !l.starts_with('$')).unwrap_or(obs).chars().take(300).collect());
    }
    None
}

/// When this error already happened twice or more to `agent`, a strong hint
/// to stop retrying and get help (`help` names who or what can help).
pub(super) fn stuck_hint(steps: &[Step], agent: Option<&str>, obs: &str, ok: Option<bool>, help: &str) -> String {
    let Some(sig) = error_sig(obs, ok) else { return String::new() };
    let seen = acts_of(steps, agent)
        .filter(|s| error_sig(s.observation.as_deref().unwrap_or(""), s.meta["ok"].as_bool()).as_deref() == Some(sig.as_str()))
        .count();
    if seen < 2 {
        return String::new();
    }
    format!(
        "\n\nSTUCK: this same error has now come back {} times. Your approach is not working and guessing again wastes turns. \
         Get help before the next attempt: {help}. Then change your approach based on what you learn.",
        seen + 1
    )
}

/// How many times `agent` made exactly this call and got exactly this
/// (failing) result before.
pub(super) fn identical_failures(steps: &[Step], agent: Option<&str>, a: &Action, obs: &str) -> usize {
    let key = call_key(a);
    let sig = error_sig(obs, Some(false));
    acts_of(steps, agent)
        .filter(|s| {
            protocol::parse_action(&s.output).is_ok_and(|p| call_key(&p) == key)
                && error_sig(s.observation.as_deref().unwrap_or(""), Some(false)) == sig
        })
        .count()
}

/// Records the lines a file tool added and removed.
pub(super) fn note_lines(meta: &mut serde_json::Value, obs: &str, ok: Option<bool>) {
    if ok == Some(true)
        && let Some((a, r)) = tools::lines_of(obs)
    {
        meta["lines"] = json!({ "added": a, "removed": r });
    }
}

/// Records the images of a step for the dashboard (`image`: the first).
pub(super) fn note_images(meta: &mut serde_json::Value, images: Vec<String>) {
    if let Some(first) = images.first() {
        meta["image"] = json!(first);
        meta["images"] = json!(images);
    }
}

/// Most images shown with one result.
const MAX_SHOWN_IMAGES: usize = 6;

/// Whether a step showed its images to the agent that made it: `view_image`
/// always, screenshots when the agent can see.
pub(super) fn shows_images(tool: &str, sees: bool) -> bool {
    tool == "view_image" || (sees && matches!(tool, "screenshot_page" | "screenshot_sequence" | "screenshot_screen" | "browser"))
}

/// When `agent`'s last step showed images (see `shows_images`), sends them
/// with the last message (only then: images are expensive to resend).
pub(super) fn attach_latest_image(mut messages: Vec<Message>, steps: &[Step], agent: Option<&str>, toolbox: &ToolBox) -> Vec<Message> {
    let last = steps.iter().rev().find(|s| s.kind == StepKind::Act && agent.is_none_or(|a| s.meta["agent"] == a));
    if let Some(s) = last
        && shows_images(s.meta["tool"].as_str().unwrap_or(""), toolbox.sees())
        && let Some(m) = messages.last_mut()
    {
        let paths: Vec<&str> = match s.meta["images"].as_array() {
            Some(v) => v.iter().filter_map(|p| p.as_str()).collect(),
            None => s.meta["image"].as_str().into_iter().collect(),
        };
        for p in paths.into_iter().take(MAX_SHOWN_IMAGES) {
            if let Ok(img) = toolbox.load_image(p) {
                m.images.push(img);
            }
        }
    }
    messages
}

/// Records a turn's outcome: success resets the failure streak, failure
/// extends it (and may escalate), a declined call leaves it alone.
fn count_outcome(cfg: &Config, meta: &mut serde_json::Value, next: &mut RunState, ok: Option<bool>) {
    meta["ok"] = json!(ok.unwrap_or(true));
    match ok {
        Some(true) => {
            next.failures = 0;
            next.escalate = false;
        }
        Some(false) => {
            next.failures += 1;
            next.escalate = cfg.routing.escalation.is_some() && next.failures >= cfg.limits.escalate_after;
        }
        None => {}
    }
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
