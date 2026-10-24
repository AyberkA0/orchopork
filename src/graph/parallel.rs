//! The Dispatcher: branches of agents that work at the same time.
//!
//! When the flow reaches a Dispatcher, each of its branches gets a worktree
//! forked from the run's and a loop of its own, and the branches' agents take
//! their turns concurrently. Their steps go into the run's history like any
//! other (tagged with the branch), and each branch's progress lives in the
//! run state (`FlowState::parallel`), so a pause, a restart or a rewind picks
//! the branches up where they were. The Dispatcher's policy decides when the
//! flow moves on; branches still running then are cancelled. Finished
//! branches are merged into the run's worktree, the Dispatcher's model
//! resolving conflicts (code), or only reported (research).

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::Poll;

use serde_json::{Value, json};

use super::flow::{Review, Side, name_of, pending_review, report_text, tool};
use super::{BRANCH, Inner, Stop};
use crate::bootstrap::Workspace;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::fsutil::clip;
use crate::git::GitRepo;
use crate::storage::{
    BranchRun, BranchStatus, EdgeContext, FlowGraph, FlowNode, FlowState, ParallelPolicy, ParallelRun, ParallelWork, Phase,
    Run, RunState, Step, StepKind,
};

/// Turns the Dispatcher gets to resolve one branch's merge conflicts.
const RESOLVE_TURNS: usize = 16;
/// What resolving conflicts may use.
const RESOLVE_TOOLS: &[&str] =
    &["list_files", "list_dir", "read_file", "search", "write_file", "edit_file", "run_command", "process_output"];
/// Longest branch report shown to the Dispatcher's model.
const REPORT_CHARS: usize = 4_000;

/// What a branch's turns need to checkpoint it (see `Inner::append`).
pub(super) struct BranchCtx {
    pub(super) head: String,
    /// The run's state, with every branch's progress in `flow.parallel`.
    pub(super) shared: Arc<tokio::sync::Mutex<RunState>>,
    /// The run worktree's commit while the branches work (it does not move).
    pub(super) main_commit: String,
    /// Where the branches started.
    pub(super) fork: String,
}

/// How a branch's loop ended.
enum BranchEnd {
    Finished(String),
    Failed(String),
    /// The run was paused (or a branch hit the per-session step limit).
    Paused(Option<String>),
    Error(Error),
}

enum Event {
    Branch(usize, String, BranchEnd),
    Judged(Result<Side>),
}

type JudgeFuture<'a> = Pin<Box<dyn Future<Output = Result<Side>> + Send + 'a>>;

/// A branch's loop, as a task of its own: it must go on while the
/// Dispatcher awaits something (an append, a lock) the branch may hold.
/// Dropped means cancelled.
struct BranchTask(tokio::task::JoinHandle<(String, BranchEnd)>);

impl Drop for BranchTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl BranchTask {
    /// Cancels the branch and waits until it has stopped (so nothing of
    /// it runs on while the Dispatcher merges and cleans up).
    async fn cancel(mut self) {
        self.0.abort();
        let _ = (&mut self.0).await;
    }
}

fn place(ws: &Workspace, run: &Run, head: &str) -> (PathBuf, String) {
    (ws.state_dir.join("worktrees").join(format!("{}-{head}", run.id)), format!("orchopork/{}-{head}", run.id))
}

/// Removes every branch worktree (and git branch) of `run`.
pub(super) async fn remove_branch_worktrees(ws: &Workspace, run: &Run) {
    let dir = ws.state_dir.join("worktrees");
    let prefix = format!("{}-", run.id);
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(head) = name.strip_prefix(&prefix) {
            let (path, branch) = place(ws, run, head);
            stop_processes(&path);
            if let Err(err) = ws.repo.worktree_remove(&path, &branch).await {
                tracing::warn!(run = run.id, "could not remove branch worktree {}: {err}", path.display());
            }
        }
    }
}

fn stop_processes(path: &Path) {
    if let Ok(root) = crate::fsutil::canonical(path) {
        super::tools::processes::stop_all(&root);
    }
}

/// A branch's name for people: its team's, else its agents'.
fn branch_label(g: &FlowGraph, b: &BranchRun) -> String {
    let mut nodes = b.nodes.clone();
    nodes.sort();
    for gr in &g.groups {
        let mut m = gr.members.clone();
        m.sort();
        if m == nodes {
            return gr.name.clone();
        }
    }
    b.nodes.iter().map(|id| name_of(g, id)).collect::<Vec<_>>().join(" → ")
}

fn status_word(s: BranchStatus) -> &'static str {
    match s {
        BranchStatus::Running => "running",
        BranchStatus::Finished => "finished",
        BranchStatus::Failed => "failed",
        BranchStatus::Cancelled => "cancelled",
    }
}

impl Inner {
    /// The Dispatcher `id`'s turn: starts (or resumes) its branches, runs
    /// them until its policy decides, merges and hands the result on.
    pub(super) async fn parallel_turn(
        &self,
        cfg: &Config,
        run: &Run,
        mut state: RunState,
        g: &FlowGraph,
        id: &str,
    ) -> Result<Option<Stop>> {
        let me = g.node(id).cloned().ok_or_else(|| Error::InvalidRequest(format!("unknown node {id}")))?;
        let settings = me.parallel_settings();
        let chains: Vec<(String, Vec<String>)> = g.branch_heads(id).into_iter().map(|h| {
            let c = g.branch_chain(&h);
            (h, c)
        }).collect();
        if chains.is_empty() {
            return Ok(Some(Stop::Paused(Some(format!("{} has no branches; connect agents to it, then resume", me.name)))));
        }
        let main = GitRepo::new(&run.worktree);
        let same = state.flow.parallel.as_ref().is_some_and(|p| {
            p.node == id
                && p.branches.len() == chains.len()
                && p.branches.iter().zip(&chains).all(|(b, (h, c))| b.head == *h && b.nodes == *c)
        });
        if same {
            self.restore_branches(state.flow.parallel.as_ref().expect("same")).await?;
        } else {
            if state.flow.parallel.take().is_some() {
                remove_branch_worktrees(&self.ws, run).await;
            }
            self.dispatch(run, &mut state, g, &me, &chains).await?;
        }
        let par = state.flow.parallel.clone().expect("dispatched");
        let main_commit = main.head().await?.unwrap_or_else(|| par.fork.clone());
        let shared = Arc::new(tokio::sync::Mutex::new(state));

        // The branches, concurrently, each a task of its own.
        let inner = self.this.upgrade().ok_or_else(|| Error::Conflict("the engine is shutting down".into()))?;
        let limit = cfg.limits.steps_per_session;
        let mut futs: Vec<BranchTask> = Vec::new();
        for b in par.branches.iter().filter(|b| b.status == BranchStatus::Running) {
            let mut brun = run.clone();
            brun.worktree = b.worktree.clone();
            brun.base_commit = par.fork.clone();
            let ctx = Arc::new(BranchCtx {
                head: b.head.clone(),
                shared: shared.clone(),
                main_commit: main_commit.clone(),
                fork: par.fork.clone(),
            });
            let (head, shared, inner, cfg, g) = (b.head.clone(), shared.clone(), inner.clone(), cfg.clone(), g.clone());
            futs.push(BranchTask(tokio::spawn(BRANCH.scope(ctx, async move {
                let end = inner.branch_loop(&cfg, brun, &g, &head, shared, limit).await;
                (head, end)
            }))));
        }
        let names: Vec<String> = par.branches.iter().map(|b| branch_label(g, b)).collect();
        self.log(&run.id, format!("{}: {} branches in parallel ({})", me.name, par.branches.len(), names.join("; ")));

        let mut judging: Option<JudgeFuture<'_>> = None;
        let mut rejudge = false;
        let mut judged_final = false;
        let mut reason = String::new();
        let pause = self.pause_flag(&run.id);
        let merge: Vec<String> = loop {
            let (finished, running) = {
                let st = shared.lock().await;
                let p = st.flow.parallel.as_ref().expect("parallel");
                let f: Vec<String> = p.branches.iter().filter(|b| b.status == BranchStatus::Finished).map(|b| b.head.clone()).collect();
                (f, p.branches.iter().any(|b| b.status == BranchStatus::Running))
            };
            if futs.is_empty() && judging.is_none() {
                // Nothing more will come in.
                if settings.policy == ParallelPolicy::Manager && !judged_final && !finished.is_empty() && !running {
                    judged_final = true;
                    let brief = self.brief(g, &shared, true).await;
                    judging = Some(Box::pin(self.judge(cfg, run, g, &me, brief)));
                    continue;
                }
                break finished;
            }
            let ev = std::future::poll_fn(|cx| {
                if let Some(j) = judging.as_mut()
                    && let Poll::Ready(r) = j.as_mut().poll(cx)
                {
                    return Poll::Ready(Event::Judged(r));
                }
                for (i, f) in futs.iter_mut().enumerate() {
                    if let Poll::Ready(r) = Pin::new(&mut f.0).poll(cx) {
                        let (head, end) = r.unwrap_or_else(|e| (String::new(), BranchEnd::Error(Error::Provider(format!("a parallel branch crashed: {e}")))));
                        return Poll::Ready(Event::Branch(i, head, end));
                    }
                }
                Poll::Pending
            })
            .await;
            match ev {
                Event::Branch(i, head, end) => {
                    // Finished: nothing to abort (and dropping a finished
                    // task's handle is harmless).
                    drop(futs.remove(i));
                    let (status, report) = match end {
                        BranchEnd::Finished(r) => (BranchStatus::Finished, r),
                        BranchEnd::Failed(why) => (BranchStatus::Failed, why),
                        // The other branches stop too, each after the turn
                        // it is taking (like a paused run); resuming
                        // continues all of them.
                        BranchEnd::Paused(msg) => {
                            drop(judging);
                            for t in futs {
                                if pause.load(Ordering::SeqCst) {
                                    let _ = (&mut { t }.0).await;
                                } else {
                                    t.cancel().await;
                                }
                            }
                            return Ok(Some(Stop::Paused(msg)));
                        }
                        BranchEnd::Error(e) => return Err(e),
                    };
                    let snapshot = {
                        let mut st = shared.lock().await;
                        if let Some(b) = st.flow.parallel.as_mut().and_then(|p| p.branch_mut(&head)) {
                            b.status = status;
                            b.report = report.clone();
                            if status == BranchStatus::Failed
                                && let Some(bs) = b.state.as_mut()
                            {
                                bs.failures = 0;
                                bs.escalate = false;
                            }
                        }
                        st.clone()
                    };
                    let label = snapshot.flow.parallel.as_ref().and_then(|p| p.branch(&head)).map(|b| branch_label(g, b)).unwrap_or_default();
                    let text = format!("Branch {label} {}:\n{report}", status_word(status));
                    let meta = json!({ "agent": me.id, "tool": "branch_report", "ok": true,
                                       "branch_report": { "head": head, "status": status, "label": label } });
                    self.append(run, StepKind::Act, &text, None, meta, &snapshot, 0.0).await?;
                    self.log(&run.id, format!("{}: branch {label} {}", me.name, status_word(status)));
                    let any_finished = snapshot
                        .flow
                        .parallel
                        .as_ref()
                        .is_some_and(|p| p.branches.iter().any(|b| b.status == BranchStatus::Finished));
                    match settings.policy {
                        ParallelPolicy::FirstSuccess if status == BranchStatus::Finished => {
                            reason = format!("{label} finished first.");
                            break vec![head];
                        }
                        // A report to judge (with nothing finished there is
                        // nothing to go on with yet: wait).
                        ParallelPolicy::Manager if any_finished => {
                            if judging.is_some() {
                                rejudge = true;
                            } else {
                                judged_final = futs.is_empty();
                                let brief = self.brief(g, &shared, futs.is_empty()).await;
                                judging = Some(Box::pin(self.judge(cfg, run, g, &me, brief)));
                            }
                        }
                        _ => {}
                    }
                }
                Event::Judged(r) => {
                    judging = None;
                    let side = match r {
                        Ok(s) => s,
                        Err(_) if pause.load(Ordering::SeqCst) => return Ok(Some(Stop::Paused(None))),
                        Err(e) => return Err(e),
                    };
                    let decision = side.args.get("decision").and_then(Value::as_str).unwrap_or("wait").trim().to_lowercase();
                    let why = side.args.get("reason").and_then(Value::as_str).unwrap_or("").trim().to_string();
                    let mut pick: Vec<String> = side
                        .args
                        .get("merge")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .filter(|h| finished.contains(h))
                        .collect();
                    // Everything has ended: waiting is not an option. No
                    // list means every finished branch.
                    let go_on = decision.starts_with("continue") || futs.is_empty();
                    if go_on && pick.is_empty() {
                        pick = finished.clone();
                    }
                    let snapshot = shared.lock().await.clone();
                    let text = format!("{} decided to {}{}", me.name, if go_on { "continue" } else { "wait" }, if why.is_empty() { ".".to_string() } else { format!(": {why}") });
                    let meta = json!({ "agent": me.id, "tool": "decide", "ok": true,
                                       "decision": { "continue": go_on, "merge": pick, "reason": why, "turns": side.turns } });
                    self.append(run, StepKind::Act, &text, None, meta, &snapshot, side.cost).await?;
                    if go_on {
                        reason = why;
                        break pick;
                    }
                    if rejudge {
                        rejudge = false;
                        let brief = self.brief(g, &shared, futs.is_empty()).await;
                        judging = Some(Box::pin(self.judge(cfg, run, g, &me, brief)));
                    }
                }
            }
        };
        // Branches still running are cancelled.
        drop(judging);
        for t in futs {
            t.cancel().await;
        }
        let mut state = {
            let mut st = shared.lock().await;
            if let Some(p) = st.flow.parallel.as_mut() {
                for b in p.branches.iter_mut().filter(|b| b.status == BranchStatus::Running) {
                    b.status = BranchStatus::Cancelled;
                    stop_processes(Path::new(&b.worktree));
                }
            }
            st.clone()
        };
        let par = state.flow.parallel.clone().expect("parallel");
        if merge.is_empty() {
            // Nothing to go on with: let the user decide. Resuming retries
            // the branches that failed or were cancelled.
            for b in state.flow.parallel.as_mut().expect("parallel").branches.iter_mut() {
                if b.status != BranchStatus::Finished {
                    b.status = BranchStatus::Running;
                }
            }
            let why: Vec<String> = par
                .branches
                .iter()
                .filter(|b| b.status != BranchStatus::Finished)
                .map(|b| format!("{}: {}", branch_label(g, b), clip(b.report.trim(), 300)))
                .collect();
            let text = format!("{}: no branch finished.\n{}", me.name, why.join("\n"));
            let meta = json!({ "agent": me.id, "tool": "branch_report", "ok": false });
            self.append(run, StepKind::Act, &text, None, meta, &state, 0.0).await?;
            return Ok(Some(Stop::Paused(Some(format!(
                "none of {}'s branches finished ({}); message their agents or change their models, then resume to retry them",
                me.name,
                why.join("; ")
            )))));
        }
        self.join(cfg, run, g, &me, &mut state, &par, &merge, &reason).await
    }

    /// Starts the branches: a worktree for each, forked from the run's.
    async fn dispatch(&self, run: &Run, state: &mut RunState, g: &FlowGraph, me: &FlowNode, chains: &[(String, Vec<String>)]) -> Result<()> {
        let main = GitRepo::new(&run.worktree);
        let fork = main.head().await?.unwrap_or_else(|| run.base_commit.clone());
        let mut branches = Vec::new();
        for (head, nodes) in chains {
            let (path, git_branch) = place(&self.ws, run, head);
            self.ws.repo.worktree_remove(&path, &git_branch).await?;
            self.ws.repo.worktree_add(&path, &git_branch, &fork).await?;
            let local = RunState {
                phase: Phase::Act,
                failures: 0,
                escalate: false,
                flow: FlowState { done: state.flow.done.clone(), ..Default::default() },
                memory: state.memory.iter().filter(|(k, _)| nodes.contains(k)).map(|(k, v)| (k.clone(), v.clone())).collect(),
                ..state.clone()
            };
            branches.push(BranchRun {
                head: head.clone(),
                nodes: nodes.clone(),
                worktree: path.to_string_lossy().to_string(),
                git_branch,
                commit: fork.clone(),
                status: BranchStatus::Running,
                report: String::new(),
                state: Some(Box::new(local)),
            });
        }
        let steps = self.ws.store.steps(&run.id).await?;
        // What came into the Dispatcher goes on to every branch.
        let input: Vec<String> = g
            .preds(&me.id)
            .iter()
            .filter_map(|p| {
                let s = steps.iter().rev().find(|s| s.kind == StepKind::Act && s.meta["agent"] == *p && s.meta["tool"] == "finish")?;
                let ctx = g.edge(p, &me.id).map(|e| e.context).unwrap_or_default();
                Some(format!("From {}:\n{}", name_of(g, p), report_text(s, ctx)))
            })
            .collect();
        let par = ParallelRun { node: me.id.clone(), fork, branches };
        let labels: Vec<String> = par.branches.iter().map(|b| branch_label(g, b)).collect();
        let summary = format!(
            "{}{}",
            if me.task.trim().is_empty() { String::new() } else { format!("Dispatcher's note: {}\n\n", me.task.trim()) },
            if input.is_empty() { "(no report came in)".to_string() } else { input.join("\n\n") }
        );
        let meta = json!({ "agent": me.id, "tool": "dispatch", "ok": true, "summary": summary,
                           "dispatch": { "branches": par.branches.iter().map(|b| json!({ "head": b.head, "nodes": b.nodes })).collect::<Vec<_>>() } });
        state.flow.parallel = Some(par);
        let text = format!("{} started {} parallel branches: {}.", me.name, labels.len(), labels.join("; "));
        self.append(run, StepKind::Act, &text, None, meta, state, 0.0).await
    }

    /// After a pause, restart or rewind: every unfinished branch's worktree
    /// exists and matches its last checkpoint.
    async fn restore_branches(&self, par: &ParallelRun) -> Result<()> {
        for b in par.branches.iter().filter(|b| b.status == BranchStatus::Running) {
            let path = Path::new(&b.worktree);
            if path.exists() {
                GitRepo::new(path).abort_merge().await;
                GitRepo::new(path).reset_hard(&b.commit).await?;
            } else {
                self.ws.repo.worktree_remove(path, &b.git_branch).await?;
                self.ws.repo.worktree_add(path, &b.git_branch, &b.commit).await?;
            }
        }
        Ok(())
    }

    /// One branch: its agents' turns, one after another, until the last
    /// one finishes.
    async fn branch_loop(
        &self,
        cfg: &Config,
        run: Run,
        g: &FlowGraph,
        head: &str,
        shared: Arc<tokio::sync::Mutex<RunState>>,
        limit: u32,
    ) -> BranchEnd {
        let pause = self.pause_flag(&run.id);
        let nodes = g.branch_chain(head);
        let last = nodes.last().cloned().unwrap_or_default();
        let mut taken = 0u32;
        loop {
            if pause.load(Ordering::SeqCst) {
                return BranchEnd::Paused(None);
            }
            if taken >= limit {
                return BranchEnd::Paused(Some(format!(
                    "a parallel branch took {taken} steps (limits.steps_per_session); review progress and resume to continue"
                )));
            }
            let steps = match self.ws.store.steps(&run.id).await {
                Ok(s) => s,
                Err(e) => return BranchEnd::Error(e),
            };
            let local = {
                let st = shared.lock().await;
                st.flow.parallel.as_ref().and_then(|p| p.branch(head)).and_then(|b| b.state.as_deref().cloned()).unwrap_or_default()
            };
            let review: Option<Review> =
                pending_review(g, &steps, &local.flow.done).filter(|r| nodes.contains(&r.reviewer) && nodes.contains(&r.subject));
            let target = match &review {
                Some(r) => r.reviewer.clone(),
                None => match nodes.iter().find(|n| !local.flow.done.contains(n)) {
                    Some(n) => n.clone(),
                    None => return BranchEnd::Finished(branch_report(g, &steps, &last)),
                },
            };
            taken += 1;
            match self.node_turn(cfg, &run, &steps, local, g, &target, review).await {
                Ok(None | Some(Stop::Done)) => {}
                Ok(Some(Stop::Paused(msg))) => {
                    if pause.load(Ordering::SeqCst) {
                        return BranchEnd::Paused(msg);
                    }
                    return BranchEnd::Failed(msg.unwrap_or_else(|| format!("{} stopped", name_of(g, &target))));
                }
                // Agents in a branch are not offered `ask`, so this only
                // happens if one asks anyway: the branch cannot wait.
                Ok(Some(Stop::Waiting(msg))) => return BranchEnd::Failed(msg),
                Err(_) if pause.load(Ordering::SeqCst) => return BranchEnd::Paused(None),
                Err(e) => return BranchEnd::Error(e),
            }
        }
    }

    /// The branches' state for the Dispatcher's model.
    async fn brief(&self, g: &FlowGraph, shared: &tokio::sync::Mutex<RunState>, ended: bool) -> String {
        let st = shared.lock().await;
        let Some(p) = st.flow.parallel.as_ref() else { return String::new() };
        let mut s = String::from("# Branches\n");
        for b in &p.branches {
            s.push_str(&format!("\n## {} (id {}): {}\n", branch_label(g, b), b.head, status_word(b.status)));
            match b.status {
                BranchStatus::Running => {
                    let done = b.state.as_ref().map(|x| x.flow.done.clone()).unwrap_or_default();
                    let now = b.nodes.iter().find(|n| !done.contains(n)).map(|n| name_of(g, n)).unwrap_or_default();
                    s.push_str(&format!("Still working (now: {now}).\n"));
                }
                _ => s.push_str(&format!("{}\n", clip(b.report.trim(), REPORT_CHARS))),
            }
        }
        if ended {
            s.push_str("\nEvery branch has ended. Decide which finished branches to merge and continue.\n");
        }
        s
    }

    /// The Dispatcher's model decides: continue (with which finished
    /// branches) or wait for more reports.
    async fn judge(&self, cfg: &Config, run: &Run, g: &FlowGraph, me: &FlowNode, brief: String) -> Result<Side> {
        self.log(&run.id, format!("{} is reading the branch reports", me.name));
        let ask = format!(
            "You are the Dispatcher of parallel branches of agents. Judge their reports against the mission and your \
             instructions ({}).\n\n{brief}\n\nCall decide with decision \"continue\" to go on now with the finished branches \
             you list in merge (their work is merged; branches still running are cancelled), or \"wait\" to wait for more \
             reports. Continue as soon as the finished work is enough; do not wait for branches you would not use.",
            if me.task.trim().is_empty() { "none given" } else { me.task.trim() }
        );
        let finish = tool(
            "decide",
            "Your decision on the parallel branches.",
            json!({
                "decision": { "type": "string", "enum": ["continue", "wait"] },
                "merge": { "type": "array", "items": { "type": "string" }, "description": "ids of the finished branches whose work to keep" },
                "reason": { "type": "string" }
            }),
            &["decision", "reason"],
        );
        self.side_session(cfg, run, g, me, &ask, &[], finish, None, vec![]).await
    }

    /// Merges (code) or collects (research) the chosen branches, reports to
    /// the next node and cleans up the branch worktrees.
    #[allow(clippy::too_many_arguments)]
    async fn join(
        &self,
        cfg: &Config,
        run: &Run,
        g: &FlowGraph,
        me: &FlowNode,
        state: &mut RunState,
        par: &ParallelRun,
        merge: &[String],
        reason: &str,
    ) -> Result<Option<Stop>> {
        let settings = me.parallel_settings();
        let main = GitRepo::new(&run.worktree);
        let pause = self.pause_flag(&run.id);
        let mut lines = Vec::new();
        let mut cost = 0.0;
        let mut merged = Vec::new();
        for b in &par.branches {
            let label = branch_label(g, b);
            let kept = merge.contains(&b.head);
            let what = match (b.status, kept, settings.work) {
                (BranchStatus::Finished, true, ParallelWork::Code) => {
                    match self.merge_branch(cfg, run, g, me, b, &label).await {
                        Ok((note, c)) => {
                            cost += c;
                            merged.push(b.head.clone());
                            note
                        }
                        Err(_) if pause.load(Ordering::SeqCst) => return Ok(Some(Stop::Paused(None))),
                        Err(e) => format!("NOT merged: {e}"),
                    }
                }
                (BranchStatus::Finished, true, ParallelWork::Research) => {
                    merged.push(b.head.clone());
                    "used (research: file changes discarded)".into()
                }
                (BranchStatus::Finished, false, _) => "finished, not used".into(),
                (s, _, _) => status_word(s).into(),
            };
            let report = if b.report.trim().is_empty() { String::new() } else { format!("\n{}", clip(b.report.trim(), REPORT_CHARS)) };
            lines.push(format!("## {label}: {what}{report}"));
        }
        let policy = match settings.policy {
            ParallelPolicy::WaitAll => "waited for every branch",
            ParallelPolicy::FirstSuccess => "went on with the first branch to finish",
            ParallelPolicy::Manager => "decided from the reports",
        };
        let summary = format!(
            "{} ran {} branches in parallel and {policy}.{}\n\n{}",
            me.name,
            par.branches.len(),
            if reason.trim().is_empty() { String::new() } else { format!(" {}", reason.trim()) },
            lines.join("\n\n")
        );
        let (stat, patch) = main.diff_from(&par.fork).await.unwrap_or_default();
        let mut next = state.clone();
        next.flow.parallel = None;
        if !next.flow.done.contains(&me.id) {
            next.flow.done.push(me.id.clone());
        }
        for b in par.branches.iter().filter(|b| b.status == BranchStatus::Finished) {
            for n in &b.nodes {
                if !next.flow.done.contains(n) {
                    next.flow.done.push(n.clone());
                }
            }
            if let Some(bs) = &b.state {
                for (k, v) in &bs.memory {
                    next.memory.insert(k.clone(), v.clone());
                }
            }
        }
        next.failures = 0;
        next.escalate = false;
        let meta = json!({
            "agent": me.id, "tool": "finish", "ok": true, "summary": summary,
            "changes": { "stat": stat.trim(), "patch": clip(&patch, cfg.limits.tool_output_chars) },
            "parallel": { "merged": merged, "branches": par.branches.iter().map(|b| json!({ "head": b.head, "status": b.status })).collect::<Vec<_>>() },
        });
        self.append(run, StepKind::Act, &summary, None, meta, &next, cost).await?;
        remove_branch_worktrees(&self.ws, run).await;
        self.log(&run.id, format!("{} handed the combined result on", me.name));
        Ok(None)
    }

    /// Merges one branch into the run's worktree; conflicts go to the
    /// Dispatcher's model. Returns a note for the report and the cost.
    async fn merge_branch(&self, cfg: &Config, run: &Run, g: &FlowGraph, me: &FlowNode, b: &BranchRun, label: &str) -> Result<(String, f64)> {
        let main = GitRepo::new(&run.worktree);
        let msg = format!("orchopork {}: merge branch {label}", run.id);
        let files = match main.merge_or_conflicts(&b.git_branch, &msg).await? {
            Ok(()) => return Ok(("merged".into(), 0.0)),
            Err(files) => files,
        };
        self.log(&run.id, format!("{} is resolving merge conflicts of {label} in {}", me.name, files.join(", ")));
        let ask = format!(
            "Merging the parallel branch \"{label}\" into the project conflicted in: {}.\n\nThe branch's report:\n{}\n\n\
             Resolve every conflict: read each file, combine both sides so that the work of both branches is kept (when they \
             truly contradict, follow the mission), and remove every conflict marker (<<<<<<<, =======, >>>>>>>). Check that \
             the result still builds or runs if you can. Then call resolved with a short summary.",
            files.join(", "),
            clip(b.report.trim(), REPORT_CHARS)
        );
        let finish = tool("resolved", "The conflicts are resolved.", json!({ "summary": { "type": "string" } }), &["summary"]);
        let side = match self.side_session_n(cfg, run, g, me, &ask, RESOLVE_TOOLS, finish, None, vec![], RESOLVE_TURNS).await {
            Ok(s) => s,
            Err(e) => {
                main.abort_merge().await;
                return Err(e);
            }
        };
        let root = Path::new(&run.worktree);
        let left: Vec<&String> = files
            .iter()
            .filter(|f| {
                std::fs::read_to_string(root.join(f.as_str())).is_ok_and(|t| {
                    t.lines().any(|l| l.starts_with("<<<<<<< ") || l.starts_with(">>>>>>> ") || l == "=======")
                })
            })
            .collect();
        if !left.is_empty() {
            main.abort_merge().await;
            return Err(Error::Git(format!(
                "conflict markers were left in {}",
                left.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            )));
        }
        main.commit_all(&msg).await?;
        let summary = side.args.get("summary").or_else(|| side.args.get("text")).and_then(Value::as_str).unwrap_or("").trim().to_string();
        Ok((format!("merged; conflicts in {} resolved by {}{}", files.join(", "), me.name, if summary.is_empty() { String::new() } else { format!(": {summary}") }), side.cost))
    }
}

/// What a finished branch hands back: its last agent's report.
fn branch_report(g: &FlowGraph, steps: &[Step], last: &str) -> String {
    match steps.iter().rev().find(|s| s.kind == StepKind::Act && s.meta["agent"] == last && s.meta["tool"] == "finish") {
        Some(s) => format!("{}: {}", name_of(g, last), report_text(s, EdgeContext::Report)),
        None => format!("{} finished without a report", name_of(g, last)),
    }
}
