//! End-to-end engine tests against scripted providers: real git worktrees,
//! real SQLite, real tools; only the LLM is fake.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use super::*;
use crate::providers::{Completion, Provider, ProviderId};
use serde_json::json;

struct Scripted {
    id: ProviderId,
    replies: Mutex<VecDeque<String>>,
    seen: Mutex<Vec<CompletionRequest>>,
    /// Sleeps before returning the reply for the call at this 1-based
    /// index, so a test can act (e.g. inject) while that call is in
    /// flight and the run is still active.
    stall: Mutex<HashMap<usize, Duration>>,
}

impl Scripted {
    fn new(id: ProviderId, replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            id,
            replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            seen: Mutex::new(vec![]),
            stall: Mutex::new(HashMap::new()),
        })
    }

    #[allow(dead_code)]
    fn stall_call(self: &Arc<Self>, call: usize, d: Duration) {
        self.stall.lock().unwrap().insert(call, d);
    }
}

struct Handle(Arc<Scripted>);

#[async_trait::async_trait]
impl Provider for Handle {
    fn id(&self) -> ProviderId {
        self.0.id
    }
    async fn complete(&self, _model: &str, req: &CompletionRequest) -> Result<Completion> {
        let call = {
            let mut seen = self.0.seen.lock().unwrap();
            seen.push(req.clone());
            seen.len()
        };
        let stall = self.0.stall.lock().unwrap().get(&call).copied();
        if let Some(d) = stall {
            tokio::time::sleep(d).await;
        }
        if req.system.starts_with("You maintain the working memory") {
            return Ok(Completion { text: "SUMMARY: wrote some files".into(), ..Default::default() });
        }
        let text = self.0.replies.lock().unwrap().pop_front().unwrap_or_else(|| FINISH.to_string());
        Ok(Completion { text, prompt_tokens: 100, completion_tokens: 50, ..Default::default() })
    }
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![])
    }
}

const FINISH: &str = r#"Done. {"tool": "finish", "args": {"summary": "wrote hello.txt"}}"#;
const WRITE: &str = r#"Creating the file. {"tool": "write_file", "args": {"path": "hello.txt", "content": "hello\n"}}"#;

fn local(model: &str) -> ModelRef {
    ModelRef::new(ProviderId::Ollama, model)
}

fn cloud() -> ModelRef {
    ModelRef::new(ProviderId::Claude, "claude-haiku-4-5")
}

async fn setup(f: impl FnOnce(&mut Config)) -> (tempfile::TempDir, Engine) {
    let d = tempfile::tempdir().unwrap();
    let repo = GitRepo::new(d.path());
    repo.init().await.unwrap();
    tokio::fs::write(d.path().join("README.md"), "# demo\n").await.unwrap();
    repo.commit_all("init").await.unwrap();
    let ws = Workspace::open(d.path()).await.unwrap();
    ws.update_config(|c| {
        c.onboarded = true;
        c.routing.actor = Some(local("actor"));
        f(c);
    })
    .unwrap();
    // Replace real providers with scripted ones after config is final.
    ws.gateway.unregister_all();
    (d, Engine::new(ws).await.unwrap())
}

fn register(engine: &Engine, s: &Arc<Scripted>) {
    engine.workspace().gateway.register(Box::new(Handle(s.clone())));
}

async fn wait_idle(engine: &Engine, id: &str) -> Run {
    for _ in 0..500 {
        if !engine.is_active(id) {
            return engine.run_detail(id).await.unwrap().0;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("run {id} did not stop");
}

fn kinds(steps: &[Step]) -> Vec<&'static str> {
    steps.iter().map(|s| s.kind.as_str()).collect()
}

#[tokio::test]
async fn happy_path_plans_acts_verifies_reviews_and_leaves_main_checkout_alone() {
    let (d, engine) = setup(|c| c.routing.critic = Some(cloud())).await;
    let actor = Scripted::new(ProviderId::Ollama, &["1. write hello.txt\nVerification: file exists", WRITE, FINISH]);
    let critic = Scripted::new(ProviderId::Claude, &[r#"{"verdict": "approve", "feedback": ""}"#]);
    register(&engine, &actor);
    register(&engine, &critic);

    let run = engine.create_run("create hello.txt", Some("test -f hello.txt"), RunSpec::default()).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();

    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(kinds(&steps), ["plan", "act", "act", "verify", "review"]);
    assert_eq!(steps[3].meta["ok"], true);
    assert!(Path::new(&run.worktree).join("hello.txt").exists());
    assert!(!d.path().join("hello.txt").exists(), "the user's checkout must not change");
    assert!(run.cost_usd > 0.0, "the cloud critic call is metered");

    // Each step is its own commit; the diff shows the run's change.
    let (stat, _) = engine.diff(&run.id).await.unwrap();
    assert!(stat.contains("hello.txt"));
    let first_actor_turn = &actor.seen.lock().unwrap()[1];
    assert_eq!(first_actor_turn.messages.len(), 1, "fresh transcript is just the pinned goal message");
    assert!(first_actor_turn.system.contains("write_file"));
}

#[tokio::test]
async fn a_question_waits_for_the_user_and_the_answer_continues_the_run() {
    let (_d, engine) = setup(|_| {}).await;
    let ask = r#"Need a decision. {"tool": "ask", "args": {"question": "Which greeting?", "options": ["hello", "hi"]}}"#;
    let actor = Scripted::new(ProviderId::Ollama, &["plan", ask, WRITE, FINISH]);
    register(&engine, &actor);

    let run = engine.create_run("create hello.txt", None, RunSpec::default()).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Waiting, "error: {:?}", run.error);
    assert_eq!(run.error.as_deref(), Some("The agent asks: Which greeting?"));
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let q = ask::waiting_on(&steps).expect("an open question");
    assert_eq!(q.meta["options"], json!(["hello", "hi"]));
    assert!(steps.iter().all(|s| s.kind == StepKind::Inject || s.meta.get("ms").is_some()), "every step is timed");

    assert!(engine.answer(&run.id, q.seq + 1, "hello").await.is_err(), "only the open question can be answered");
    engine.answer(&run.id, q.seq, "hello").await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert!(ask::waiting_on(&steps).is_none());
    let saw = actor.seen.lock().unwrap()[2].messages.iter().any(|m| m.content.contains("Answer from the user:\nhello"));
    assert!(saw, "the agent must see the answer");
}

#[tokio::test]
async fn failed_verification_sends_the_actor_back_to_work() {
    let (_d, engine) = setup(|_| {}).await;
    let fix = r#"{"tool": "write_file", "args": {"path": "ok", "content": ""}}"#;
    let actor = Scripted::new(ProviderId::Ollama, &["plan", FINISH, fix, FINISH]);
    register(&engine, &actor);

    let run = engine.create_run("make `ok` exist", Some("test -f ok"), RunSpec::default()).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(kinds(&steps), ["plan", "act", "verify", "act", "act", "verify"]);
    assert_eq!(steps[2].meta["ok"], false);
    let saw_failure =
        actor.seen.lock().unwrap()[2].messages.iter().any(|m| m.content.contains("FAIL [run] test -f ok"));
    assert!(saw_failure, "the actor must see why verification failed");
}

#[tokio::test]
async fn repeated_failures_escalate_then_pause() {
    let (_d, engine) = setup(|c| {
        c.routing.escalation = Some(cloud());
        c.limits.escalate_after = 2;
        c.limits.max_failures = 3;
    })
    .await;
    let actor = Scripted::new(ProviderId::Ollama, &["plan", "no json here", "still none"]);
    let strong = Scripted::new(ProviderId::Claude, &["neither do I"]);
    register(&engine, &actor);
    register(&engine, &strong);

    let run = engine.create_run("anything", None, RunSpec::default()).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Paused);
    assert!(run.error.as_deref().unwrap().contains("3 consecutive failed turns"));
    assert_eq!(steps[3].meta["escalated"], true);
    assert_eq!(steps[3].meta["model"], "claude:claude-haiku-4-5");
    assert_eq!(strong.seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn budget_exhaustion_falls_back_to_the_local_actor() {
    let (_d, engine) = setup(|c| {
        c.monthly_cap_usd = 0.0;
        c.routing.planner = Some(cloud());
    })
    .await;
    let actor = Scripted::new(ProviderId::Ollama, &["local plan", FINISH]);
    let planner = Scripted::new(ProviderId::Claude, &["cloud plan"]);
    register(&engine, &actor);
    register(&engine, &planner);

    let run = engine.create_run("anything", None, RunSpec::default()).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps[0].output, "local plan");
    assert!(planner.seen.lock().unwrap().is_empty(), "no cloud call may leave the process over budget");
    assert_eq!(run.cost_usd, 0.0);
}

#[tokio::test]
async fn inject_reopens_a_done_run_and_rewind_restores_the_worktree() {
    let (_d, engine) = setup(|_| {}).await;
    let actor = Scripted::new(ProviderId::Ollama, &["plan", WRITE, FINISH]);
    register(&engine, &actor);
    let run = engine.create_run("create hello.txt", None, RunSpec::default()).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Done);
    assert!(matches!(engine.start(&run.id).await, Err(Error::Conflict(_))));

    engine.inject(&run.id, "also add bye.txt", None).await.unwrap();
    let bye = r#"{"tool": "write_file", "args": {"path": "bye.txt", "content": "bye"}}"#;
    actor.replies.lock().unwrap().extend([bye.to_string(), FINISH.to_string()]);
    engine.start(&run.id).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done);
    assert_eq!(kinds(&steps), ["plan", "act", "act", "inject", "act", "act"]);
    let wt = Path::new(&run.worktree);
    assert!(wt.join("bye.txt").exists());

    // Back to right after the first write: bye.txt gone, hello.txt kept.
    let run = engine.rewind(&run.id, 1).await.unwrap();
    assert_eq!((run.status, run.step_count), (RunStatus::Paused, 2));
    assert!(wt.join("hello.txt").exists() && !wt.join("bye.txt").exists());
    // And all the way back to the start.
    engine.rewind(&run.id, -1).await.unwrap();
    assert!(!wt.join("hello.txt").exists());
    assert_eq!(engine.run_detail(&run.id).await.unwrap().1.len(), 0);

    engine.delete(&run.id).await.unwrap();
    assert!(!wt.exists());
}

#[tokio::test]
async fn pause_stops_between_steps_and_a_second_engine_is_refused() {
    let (d, engine) = setup(|c| c.limits.steps_per_session = 2).await;
    let actor = Scripted::new(ProviderId::Ollama, &["plan", WRITE, WRITE, FINISH]);
    register(&engine, &actor);
    let run = engine.create_run("x", None, RunSpec::default()).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!((run.status, run.step_count), (RunStatus::Paused, 2));
    assert!(run.error.unwrap().contains("steps_per_session"));

    let ws2 = Workspace::open(d.path()).await.unwrap();
    assert!(matches!(Engine::new(ws2).await, Err(Error::Conflict(_))));
}

#[tokio::test]
async fn solo_runs_use_their_own_model_and_skip_planning() {
    let (_d, engine) = setup(|_| {}).await;
    let chosen = Scripted::new(ProviderId::Claude, &[WRITE, FINISH]);
    register(&engine, &chosen);
    let spec = RunSpec { autonomy: Default::default(), graph: None, worker_model: None, mode: RunMode::Solo, model: Some(cloud()), agents: vec![] };
    let run = engine.create_run("create hello.txt", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(kinds(&steps), ["act", "act"]);
    assert_eq!(steps[0].meta["model"], "claude:claude-haiku-4-5");
}

use crate::storage::{EdgeCaps, FlowEdge, FlowGraph, FlowNode, GoalJudge, NodeKind};

fn fnode(id: &str, kind: NodeKind, task: &str) -> FlowNode {
    FlowNode::new(id, kind, &id.to_uppercase(), task)
}

fn fedge(from: &str, to: &str, caps: EdgeCaps) -> FlowEdge {
    FlowEdge { caps, ..FlowEdge::new(from, to) }
}

/// Lead → `agents`… → Goal in a line, the Goal deciding by `check` only.
fn line(agents: &[&str], check: &str) -> FlowGraph {
    let mut g = FlowGraph::default();
    g.nodes.push(fnode("lead", NodeKind::Lead, "plan the work"));
    let mut prev = "lead".to_string();
    for a in agents {
        g.nodes.push(fnode(a, NodeKind::Agent, &format!("{a}'s part")));
        g.edges.push(FlowEdge::new(&prev, a));
        prev = a.to_string();
    }
    let mut goal = fnode("goal", NodeKind::Goal, "");
    let gs = goal.goal.as_mut().unwrap();
    gs.judge = GoalJudge::Command;
    gs.verify_command = Some(check.into());
    g.nodes.push(goal);
    g.edges.push(FlowEdge::new(&prev, "goal"));
    g
}

fn orch(g: FlowGraph) -> RunSpec {
    RunSpec { mode: RunMode::Orchestra, model: Some(local("lead")), graph: Some(g), ..Default::default() }
}

fn fin(summary: &str) -> String {
    format!(r#"{{"tool": "finish", "args": {{"summary": "{summary}"}}}}"#)
}

fn who(steps: &[Step]) -> Vec<&str> {
    steps.iter().map(|s| s.meta["agent"].as_str().unwrap_or("-")).collect()
}

fn ctx(r: &CompletionRequest) -> String {
    r.messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
}

#[tokio::test]
async fn flow_runs_left_to_right_and_the_goal_ends_it() {
    let (_d, engine) = setup(|_| {}).await;
    let replies = [fin("dev: write hello.txt").as_str(), WRITE, &fin("hello.txt written")].map(str::to_string);
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("make hello", None, orch(line(&["dev"], "test -f hello.txt"))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "dev", "dev", "goal"]);
    assert_eq!(steps[3].kind, StepKind::Verify);
    assert_eq!(steps[3].meta["ok"], true);
    assert!(steps[2].meta["changes"]["stat"].as_str().unwrap().contains("hello.txt"));

    let seen = model.seen.lock().unwrap();
    assert!(ctx(&seen[1]).contains("Handoff from LEAD") && ctx(&seen[1]).contains("dev: write hello.txt"));
    assert!(seen[1].tools.iter().any(|t| t.name == "finish") && !seen[1].tools.iter().any(|t| t.name == "delegate"));
    assert!(seen[0].tools.iter().any(|t| t.name == "rerun"), "the Lead coordinates");
    assert!(!seen[1].tools.iter().any(|t| t.name == "rerun"));
    // Consecutive turns of one agent keep a cacheable prefix; status lives in the tail.
    assert_eq!(seen[1].system, seen[2].system);
    assert_eq!(seen[1].messages[..], seen[2].messages[..seen[1].messages.len()]);
    assert!(seen[1].tail.as_deref().unwrap().contains("dev working (you)"));
    let shared = |r: &CompletionRequest| r.messages[0].content[..r.messages[0].cache_split.unwrap()].to_string();
    assert_eq!(shared(&seen[0]), shared(&seen[1]), "every node shares the mission prefix");
}

#[tokio::test]
async fn a_failed_goal_goes_back_to_the_lead_who_reruns_an_agent() {
    let (_d, engine) = setup(|_| {}).await;
    let replies = [
        fin("dev: write files"),
        WRITE.to_string(),
        fin("hello.txt written"),
        // goal check fails: bye.txt missing
        r#"{"tool": "rerun", "args": {"agent": "dev", "instruction": "also add bye.txt"}}"#.to_string(),
        fin("dev reruns"),
        r#"{"tool": "write_file", "args": {"path": "bye.txt", "content": "bye"}}"#.to_string(),
        fin("bye.txt added"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("make files", None, orch(line(&["dev"], "test -f bye.txt"))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "dev", "dev", "goal", "lead", "lead", "dev", "dev", "goal"]);
    assert_eq!((steps[3].meta["ok"].clone(), steps[8].meta["ok"].clone()), (json!(false), json!(true)));
    assert_eq!(steps[8].state.flow.goal_rounds, 1);
    let seen = model.seen.lock().unwrap();
    assert!(ctx(&seen[3]).contains("Goal NOT reached"), "the Lead gets the failed check");
    assert!(ctx(&seen[5]).contains("also add bye.txt"), "the rerun instruction reaches the agent");
}

#[tokio::test]
async fn an_approve_edge_sends_work_back_until_it_is_approved() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev", "rev"], "test -f hello.txt");
    g.edges.iter_mut().find(|e| e.from == "dev").unwrap().caps.approve = true;
    let replies = [
        fin("go"),
        WRITE.to_string(),
        fin("wrote hello"),
        r#"{"tool": "review", "args": {"accept": false, "feedback": "say hello louder"}}"#.to_string(),
        fin("louder now"),
        r#"{"tool": "review", "args": {"accept": true, "feedback": ""}}"#.to_string(), // bounced: nothing checked
        r#"{"tool": "read_file", "args": {"path": "hello.txt"}}"#.to_string(),
        r#"{"tool": "review", "args": {"accept": true, "feedback": ""}}"#.to_string(),
        fin("reviewed"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("hello", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "dev", "dev", "rev", "dev", "rev", "rev", "rev", "rev", "goal"]);
    assert_eq!(steps[3].meta["review"]["accept"], false);
    assert_eq!(steps[5].meta["evidence_nudge"], true, "an approval with nothing checked is sent back once");
    assert!(steps[5].meta.get("review").is_none());
    assert_eq!(steps[7].meta["review"]["accept"], true);
    let seen = model.seen.lock().unwrap();
    assert!(seen[3].tools.iter().any(|t| t.name == "review") && !seen[3].tools.iter().any(|t| t.name == "write_file"),
        "a reviewer decides, read-only");
    assert!(ctx(&seen[4]).contains("sent your work back") && ctx(&seen[4]).contains("say hello louder"));
}

#[tokio::test]
async fn a_report_edge_lets_the_upstream_agent_accept_the_work() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev"], "echo ok");
    g.edges[0].caps.report = true;
    let replies = [
        fin("go"),
        fin("done it"),
        r#"{"tool": "list_files", "args": {}}"#.to_string(),
        r#"{"tool": "review", "args": {"accept": true, "feedback": "fine"}}"#.to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "dev", "lead", "lead", "goal"]);
    assert!(ctx(&model.seen.lock().unwrap()[2]).contains("reports back to you"));
    assert_eq!(steps[3].meta["review"]["accept"], true, "having looked, the accept goes through");
}

#[tokio::test]
async fn agents_consult_each_other_and_share_a_notebook() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["a", "b"], "echo ok");
    g.edges.iter_mut().find(|e| e.from == "a").unwrap().caps = EdgeCaps { consult: true, ..Default::default() };
    g.edges.iter_mut().find(|e| e.from == "a").unwrap().call_limit = 1;
    g.nodes.push(fnode("mem", NodeKind::Notebook, ""));
    g.edges.push(fedge("mem", "a", EdgeCaps::default()));
    g.edges.push(fedge("mem", "b", EdgeCaps::default()));
    let replies = [
        fin("go"),
        r#"{"tool": "consult", "args": {"agent": "b", "question": "which port?"}}"#.to_string(),
        r#"{"tool": "answer", "args": {"text": "use 8080"}}"#.to_string(), // b, inside a's turn
        r#"{"tool": "consult", "args": {"agent": "b", "question": "again?"}}"#.to_string(),
        r#"{"tool": "notebook_write", "args": {"entry": "port is 8080"}}"#.to_string(),
        r#"{"tool": "finish", "args": {"summary": "a done", "notes": "the API lives in api.rs"}}"#.to_string(),
        fin("b done"), // no notes: sent back once
        fin("b done"), // still none: accepted
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("ports", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "a", "a", "a", "a", "b", "b", "goal"]);
    assert_eq!(steps[1].meta["comm"]["response"], "use 8080");
    assert_eq!(steps[1].meta["comm"]["with"], "b");
    assert!(steps[1].observation.as_deref().unwrap().contains("use 8080"));
    assert!(steps[2].observation.as_deref().unwrap().contains("limit"), "{:?}", steps[2].observation);
    assert_eq!(steps[4].meta["notes"][0]["notebook"], "mem", "finish's notes go to the notebook");
    assert_eq!(steps[5].meta["notes_missing"], true, "a finish without notes is sent back once");
    let seen = model.seen.lock().unwrap();
    assert!(seen[2].tools.iter().any(|t| t.name == "answer") && !seen[2].tools.iter().any(|t| t.name == "write_file"));
    let b_ctx = ctx(&seen[6]);
    assert!(b_ctx.contains("Earlier, A asked you"), "b remembers being consulted");
    assert!(b_ctx.contains("Shared note in MEM from A:
port is 8080"), "notes reach b without reading: {b_ctx}");
    assert!(b_ctx.contains("the API lives in api.rs"));
    assert!(b_ctx.contains("Shared notebook"), "b is told it shares a notebook");
    let finish = seen[6].tools.iter().find(|t| t.name == "finish").unwrap();
    assert_eq!(finish.parameters["required"], json!(["summary", "notes"]));
    assert!(!seen[0].tools.iter().any(|t| t.name == "notebook_write"), "only connected agents get the notebook");
}

#[tokio::test]
async fn a_flow_node_asks_the_user_and_carries_on_with_the_answer() {
    let (_d, engine) = setup(|_| {}).await;
    let ask = r#"{"tool": "ask", "args": {"question": "Greeting text?"}}"#;
    let replies = [fin("dev: write hello.txt").as_str(), ask, WRITE, &fin("hello.txt written")].map(str::to_string);
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("make hello", None, orch(line(&["dev"], "test -f hello.txt"))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Waiting, "error: {:?}", run.error);
    assert_eq!(run.error.as_deref(), Some("DEV asks: Greeting text?"));
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let q = ask::waiting_on(&steps).unwrap();
    assert_eq!(q.meta["agent"], "dev");
    assert!(model.seen.lock().unwrap()[1].tools.iter().any(|t| t.name == "ask"));

    engine.answer(&run.id, q.seq, "hello").await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(who(&steps), ["lead", "dev", "dev", "dev", "dev", "goal"], "the answer goes to dev, who goes on");
    assert!(ctx(&model.seen.lock().unwrap()[2]).contains("Answer from the user:\nhello"));
}

/// A flow whose questions go to the Lead first.
fn lead_answers(mut spec: RunSpec) -> RunSpec {
    spec.graph.as_mut().unwrap().questions = crate::storage::flowgraph::QuestionRoute::Lead;
    spec
}

#[tokio::test]
async fn the_lead_answers_a_question_and_the_agent_carries_on() {
    let (_d, engine) = setup(|_| {}).await;
    let ask = r#"{"tool": "ask", "args": {"question": "Greeting text?", "options": ["hello", "hi"]}}"#;
    let answer = r#"{"tool": "answer", "args": {"text": "use hello"}}"#;
    let replies = [fin("dev: write hello.txt").as_str(), ask, answer, WRITE, &fin("hello.txt written")].map(str::to_string);
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("make hello", None, lead_answers(orch(line(&["dev"], "test -f hello.txt")))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Done, "no waiting for the user: {:?}", run.error);
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let q = steps.iter().find(|s| s.meta["tool"] == "ask").unwrap();
    assert_eq!((q.meta["route"].as_str(), q.meta["answered_by"].as_str()), (Some("lead"), Some("lead")));
    assert_eq!(q.meta["answer"], "use hello");
    assert!(q.observation.as_deref().unwrap().contains("use hello"));
    assert!(ask::waiting_on(&steps).is_none());
    let seen = model.seen.lock().unwrap();
    let to_lead = ctx(&seen[2]);
    assert!(to_lead.contains("Greeting text?") && to_lead.contains("- hi") && to_lead.contains(ask::ASK_USER), "{to_lead}");
    assert!(!seen[2].tools.iter().any(|t| t.name == "write_file"), "the Lead answers read-only");
}

#[tokio::test]
async fn the_lead_passes_a_question_it_cannot_decide_to_the_user() {
    let (_d, engine) = setup(|_| {}).await;
    let ask = r#"{"tool": "ask", "args": {"question": "Which brand colour?"}}"#;
    let pass = r#"{"tool": "answer", "args": {"text": "ASK_USER"}}"#;
    let replies = [fin("dev: style it").as_str(), ask, pass].map(str::to_string);
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("style", None, lead_answers(orch(line(&["dev"], "echo ok")))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Waiting, "error: {:?}", run.error);
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let q = ask::waiting_on(&steps).expect("now the user's question");
    assert_eq!((q.meta["route"].as_str(), q.meta["passed_on_by"].as_str()), (Some("user"), Some("lead")));
}

#[tokio::test]
async fn the_user_can_edit_the_flow_while_it_is_idle() {
    let (_d, engine) = setup(|_| {}).await;
    let replies = [fin("go"), fin("dev done"), fin("dev again, as edited"), fin("qa done")];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(line(&["dev"], "echo ok"))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Done);
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();

    // Change dev's task and add a QA agent after it.
    let mut g = flow::current_graph(&run, &steps.last().unwrap().state).unwrap();
    g.nodes.iter_mut().find(|n| n.id == "dev").unwrap().task = "write it in Rust".into();
    g.nodes.push(fnode("qa", NodeKind::Agent, "check it"));
    g.edges.retain(|e| e.from != "dev");
    g.edges.push(FlowEdge::new("dev", "qa"));
    g.edges.push(FlowEdge::new("qa", "goal"));
    let mut bad = g.clone();
    bad.edges.retain(|e| e.to != "goal");
    assert!(engine.edit_flow(&run.id, bad).await.is_err(), "an invalid flow is refused");
    engine.edit_flow(&run.id, g).await.unwrap();
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let edit = steps.last().unwrap();
    assert_eq!(edit.kind, StepKind::Inject);
    assert!(edit.output.contains("changed DEV") && edit.output.contains("added QA"), "{}", edit.output);
    assert_eq!(edit.state.flow.done, ["lead"], "dev and everything after it run again");
    let paused = engine.run_detail(&run.id).await.unwrap().0;
    assert_eq!(paused.status, RunStatus::Paused, "a done run with work to redo shows as paused");

    engine.start(&run.id).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "dev", "goal", "", "dev", "qa", "goal"]);
    assert!(ctx(&model.seen.lock().unwrap()[2]).contains("Your task now: write it in Rust"));
}

#[tokio::test]
async fn goal_criteria_are_judged_one_by_one() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev"], "");
    let gs = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap();
    gs.judge = GoalJudge::Agent;
    gs.verify_command = None;
    gs.max_rounds = 1;
    gs.criteria = "1. hello.txt exists
2. it says goodbye".into();
    let replies = [
        fin("go"),
        WRITE.to_string(),
        fin("wrote hello"),
        r#"{"tool": "read_file", "args": {"path": "hello.txt"}}"#.to_string(),
        r#"{"tool": "goal_verdict", "args": {"achieved": true, "evidence": "looked", "checklist": [
            {"criterion": "hello.txt exists", "met": true, "evidence": "read it"},
            {"criterion": "it says goodbye", "met": false, "evidence": "it says hello"}]}}"#.to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let goal = steps.last().unwrap();
    assert_eq!(goal.meta["ok"], false, "an unmet criterion overrides an overall yes");
    assert!(goal.meta["missing"].as_str().unwrap().contains("it says goodbye (it says hello)"));
    assert_eq!(goal.meta["checklist"].as_array().unwrap().len(), 2);
    assert!(goal.output.contains("✗ it says goodbye"), "{}", goal.output);
    assert_eq!(run.status, RunStatus::Paused);
}

#[tokio::test]
async fn the_lead_can_add_an_agent_when_allowed() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&[], "test -f docs.txt");
    g.nodes[0].lead.as_mut().unwrap().dynamic_agents = true;
    let replies = [
        r#"{"tool": "spawn_agent", "args": {"name": "Doc Writer", "role": "docs", "task": "write docs.txt"}}"#.to_string(),
        fin("handing to docs"),
        r#"{"tool": "write_file", "args": {"path": "docs.txt", "content": "docs"}}"#.to_string(),
        fin("docs written"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("docs", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "lead", "doc_writer", "doc_writer", "goal"]);
    assert!(steps[4].state.flow.patch.added_nodes.iter().any(|n| n.id == "doc_writer" && n.dynamic));
}

/// Spawning, retiring and spawning again under the same name must keep the
/// flow intact (the edge restored on retire and the re-added node must not
/// be swallowed by the earlier removals).
#[tokio::test]
async fn the_lead_can_retire_an_added_agent_and_add_it_again() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&[], "echo ok");
    g.nodes[0].lead.as_mut().unwrap().dynamic_agents = true;
    let spawn = r#"{"tool": "spawn_agent", "args": {"name": "Helper", "task": "help"}}"#.to_string();
    let replies = [
        spawn.clone(),
        r#"{"tool": "retire_agent", "args": {"agent": "helper"}}"#.to_string(),
        spawn,
        fin("handing over"),
        fin("helped"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    for s in &steps[..3] {
        assert_eq!(s.meta["ok"], true, "{}", s.observation.as_deref().unwrap_or(""));
    }
    // The retired id stays taken, so the new agent gets a fresh one.
    assert_eq!(who(&steps), ["lead", "lead", "lead", "lead", "helper_2", "goal"]);
}

#[tokio::test]
async fn adding_an_agent_waits_for_approval_in_semi_auto() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&[], "echo ok");
    g.nodes[0].lead.as_mut().unwrap().dynamic_agents = true;
    let replies = [r#"{"tool": "spawn_agent", "args": {"name": "Helper", "task": "help"}}"#.to_string(), fin("done")];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let spec = RunSpec { autonomy: Autonomy::Semi, ..orch(g) };
    let run = engine.create_run("x", None, spec).await.unwrap();
    let a = wait_approval(&engine, &run.id).await;
    assert_eq!((a.tool.as_str(), a.args["name"].as_str()), ("spawn_agent", Some("Helper")));
    engine.decide(&run.id, &a.id, false, false, "no extra agents").unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "lead", "goal"]);
    assert!(steps[1].state.flow.patch.added_nodes.is_empty());
}

#[tokio::test]
async fn old_turns_are_summarized_instead_of_dropped() {
    let (_d, engine) = setup(|c| c.limits.local_context_chars = 3_000).await;
    let big = |i: usize| format!(r#"{{"tool": "write_file", "args": {{"path": "f{i}.txt", "content": "{}"}}}}"#, "x".repeat(900));
    let replies: Vec<String> = (0..6).map(big).chain([FINISH.to_string()]).collect();
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(local("actor")), ..Default::default() };
    let run = engine.create_run("many files", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let mem = &steps.last().unwrap().state.memory[""];
    assert!(mem.upto >= 0 && mem.text.contains("SUMMARY"), "{mem:?}");
    let seen = model.seen.lock().unwrap();
    let last = seen.iter().rev().find(|r| !r.system.starts_with("You maintain")).unwrap();
    assert!(ctx(last).contains("Your memory of earlier turns") && ctx(last).contains("SUMMARY"));
    assert!(!ctx(last).contains("earlier turns are omitted"), "nothing was dropped");
}

#[tokio::test]
async fn orchestras_need_a_valid_flow() {
    let (_d, engine) = setup(|_| {}).await;
    let legacy = RunSpec { mode: RunMode::Orchestra, model: Some(local("lead")), ..Default::default() };
    assert!(matches!(engine.create_run("x", None, legacy).await, Err(Error::InvalidRequest(_))));
    let mut g = line(&["a"], "echo ok");
    g.edges.retain(|e| e.to != "goal");
    assert!(matches!(engine.create_run("x", None, orch(g)).await, Err(Error::InvalidRequest(_))));
    let mut g = line(&["a"], "echo ok");
    let goal = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap();
    goal.goal.as_mut().unwrap().verify_command = None;
    assert!(matches!(engine.create_run("x", None, orch(g)).await, Err(Error::InvalidRequest(_))));
}

#[tokio::test]
async fn re_orchestrating_cancels_the_run_and_starts_a_new_flow() {
    let (_d, engine) = setup(|_| {}).await;
    let design = r#"{"goal": {"criteria": "notes exist"}, "nodes": [{"id": "lead", "kind": "lead", "name": "Lead", "task": "write notes"}], "edges": []}"#;
    let replies = [
        fin("go"),
        WRITE.to_string(),
        fin("hello written"),
        design.to_string(),
        fin("new plan done"),
        r#"{"tool": "list_files", "args": {}}"#.to_string(),
        r#"{"tool": "goal_verdict", "args": {"achieved": true, "evidence": "files are there", "missing": ""}}"#.to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let old = engine.create_run("first", None, orch(line(&["dev"], "echo ok"))).await.unwrap();
    let old = wait_idle(&engine, &old.id).await;
    assert_eq!(old.status, RunStatus::Done);
    let head = GitRepo::new(&old.worktree).head().await.unwrap().unwrap();
    let new = match engine.reorchestrate(&old.id, "second goal", true).await.unwrap() {
        flow::Reorchestration::Started(r) => r,
        flow::Reorchestration::Draft { .. } => panic!("full autonomy starts right away"),
    };
    assert_eq!(new.base_commit, head, "continues from the old run's code");
    assert!(Path::new(&new.worktree).join("hello.txt").exists());
    let new = wait_idle(&engine, &new.id).await;
    let (_, steps) = engine.run_detail(&new.id).await.unwrap();
    assert_eq!(new.status, RunStatus::Done, "error: {:?}", new.error);
    assert_eq!(who(&steps), ["lead", "goal"]);
    assert_eq!(steps[1].meta["judged"], "lead", "the agent before the Goal judges");
}

#[tokio::test]
async fn re_orchestrating_in_semi_auto_returns_a_draft() {
    let (_d, engine) = setup(|_| {}).await;
    let design = r#"{"nodes": [{"id": "lead", "kind": "lead", "name": "Lead", "task": "t"}, {"id": "dev", "kind": "agent", "name": "Dev", "task": "code"}], "edges": [{"from": "lead", "to": "dev"}, {"from": "dev", "to": "goal"}]}"#;
    let replies = [fin("go"), fin("done"), design.to_string()];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let spec = RunSpec { autonomy: Autonomy::Semi, ..orch(line(&["dev"], "echo ok")) };
    let old = engine.create_run("first", None, spec).await.unwrap();
    let old = wait_idle(&engine, &old.id).await;
    match engine.reorchestrate(&old.id, "second goal", false).await.unwrap() {
        flow::Reorchestration::Draft { spec, base, .. } => {
            assert!(base.is_none(), "a clean start by default");
            assert_eq!(spec.graph.unwrap().order(), ["lead", "dev", "goal"]);
            assert_eq!(spec.autonomy, Autonomy::Semi);
        }
        flow::Reorchestration::Started(_) => panic!("semi-auto shows a draft first"),
    }
    assert_eq!(engine.list_runs().await.unwrap().len(), 1, "nothing new starts before approval");
}

#[tokio::test]
async fn the_goal_judge_must_check_before_saying_yes() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev"], "");
    let gs = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap();
    gs.judge = GoalJudge::Agent;
    gs.verify_command = None;
    let replies = [
        fin("go"),
        WRITE.to_string(),
        fin("hello written"),
        r#"{"tool": "goal_verdict", "args": {"achieved": true, "evidence": "the report says so"}}"#.to_string(),
        r#"{"tool": "read_file", "args": {"path": "hello.txt"}}"#.to_string(),
        r#"{"tool": "goal_verdict", "args": {"achieved": true, "evidence": "hello.txt contains hello"}}"#.to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("hello", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let goal = steps.last().unwrap();
    let tools: Vec<&str> = goal.meta["turns"].as_array().unwrap().iter().filter_map(|t| t["tool"].as_str()).collect();
    assert_eq!(tools, ["rejected", "read_file", "goal_verdict"]);
    assert_eq!(goal.meta["evidence"], "hello.txt contains hello");
    assert!(goal.output.contains("after read_file") && goal.output.contains("hello.txt contains hello"), "{}", goal.output);
    assert!(ctx(&model.seen.lock().unwrap()[4]).contains("You have not checked anything yet"));
}

#[tokio::test]
async fn a_broken_web_page_fails_the_goal_without_asking_anyone() {
    if crate::graph::browser::find_browser().is_none() {
        eprintln!("skipped: no Edge/Chrome here");
        return;
    }
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev"], "");
    let gs = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap();
    gs.judge = GoalJudge::Agent;
    gs.verify_command = None;
    gs.max_rounds = 1;
    // The same bug as a real run: a bare module specifier with no import map.
    let page = r#"{"tool": "write_file", "args": {"path": "index.html", "content": "<script type="module">import * as T from 'three'; document.body.textContent = 'ok';</script>"}}"#;
    let replies = [fin("go"), page.to_string(), fin("page written")];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("a page", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let goal = steps.last().unwrap();
    assert_eq!(goal.kind, StepKind::Verify);
    assert_eq!(goal.meta["ok"], false);
    assert!(goal.meta["missing"].as_str().unwrap().contains("Failed to resolve module specifier"), "{}", goal.meta);
    assert_eq!(run.status, RunStatus::Paused, "one check allowed, so it pauses for the user");
    assert_eq!(model.seen.lock().unwrap().len(), 3, "no judge was asked about a page that does not load");
}

#[tokio::test]
async fn a_page_with_a_missing_texture_fails_the_goal_even_with_a_clean_console() {
    if crate::graph::browser::find_browser().is_none() {
        eprintln!("skipped: no Edge/Chrome here");
        return;
    }
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev"], "");
    let gs = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap();
    gs.judge = GoalJudge::Agent;
    gs.verify_command = None;
    gs.max_rounds = 1;
    // The same bug as a real run: the water's normal map is a 404, nothing is logged.
    let page = r#"{"tool": "write_file", "args": {"path": "index.html", "content": "<canvas></canvas><script>new Image().src = 'textures/water/Water_1_M_Normal.jpg';</script>"}}"#;
    let replies = [fin("go"), page.to_string(), fin("page written")];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("a page", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let goal = steps.last().unwrap();
    assert_eq!(goal.meta["ok"], false);
    let missing = goal.meta["missing"].as_str().unwrap();
    assert!(missing.contains("404") && missing.contains("Water_1_M_Normal.jpg"), "{}", goal.meta);
}

#[tokio::test]
async fn a_blind_judge_of_visual_criteria_hears_from_an_agent_who_can_see() {
    if crate::graph::browser::find_browser().is_none() {
        eprintln!("skipped: no Edge/Chrome here");
        return;
    }
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev", "eyes"], "");
    g.nodes.iter_mut().find(|n| n.id == "eyes").unwrap().model = Some(seeing("eyes"));
    let gs = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap();
    gs.judge = GoalJudge::Judge;
    gs.judge_model = Some(local("judge"));
    gs.verify_command = None;
    gs.max_rounds = 1;
    gs.criteria = "The page shows a blue ocean.".into();
    let page = r#"{"tool": "write_file", "args": {"path": "index.html", "content": "<body style='background:#87ceeb'></body>"}}"#;
    let replies = [
        fin("go"),
        page.to_string(),
        fin("page written"),
        fin("eyes done"),
        r#"{"tool": "describe", "args": {"text": "not met: the page is one flat light blue colour, no ocean"}}"#.to_string(),
        r#"{"tool": "read_file", "args": {"path": "index.html"}}"#.to_string(),
        r#"{"tool": "goal_verdict", "args": {"achieved": false, "checklist": [{"criterion": "blue ocean", "met": false, "evidence": "EYES reported a flat colour"}], "evidence": "per EYES"}}"#.to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("an ocean", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(steps.last().unwrap().meta["ok"], false, "{:?}", steps.last().unwrap().meta);
    let seen = model.seen.lock().unwrap();
    let describe = seen.iter().find(|r| r.tools.iter().any(|t| t.name == "describe")).expect("eyes was asked to describe");
    assert_eq!(describe.messages[0].images.len(), 1, "with the engine's screenshot");
    let judge = seen.iter().find(|r| r.tools.iter().any(|t| t.name == "goal_verdict")).unwrap();
    assert!(ctx(judge).contains("EYES (who can see) saw") && ctx(judge).contains("one flat light blue colour"), "{}", ctx(judge));
}

#[tokio::test]
async fn the_checklist_decides_even_when_the_overall_flag_is_off_or_nested() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev"], "");
    let gs = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap();
    gs.judge = GoalJudge::Agent;
    gs.verify_command = None;
    // Every item met but "achieved" false, and the whole verdict nested in a string.
    let verdict = r#"{"tool": "goal_verdict", "args": {"text": "{\"achieved\": false, \"evidence\": \"ran it\", \"checklist\": [{\"criterion\": \"hello.txt exists\", \"met\": true, \"evidence\": \"read it\"}]}"}}"#;
    let replies = [
        fin("go"),
        WRITE.to_string(),
        fin("wrote"),
        r#"{"tool": "read_file", "args": {"path": "hello.txt"}}"#.to_string(),
        verdict.to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let goal = steps.last().unwrap();
    assert_eq!(goal.meta["ok"], true);
    assert_eq!(goal.meta["checklist"][0]["met"], true);
    assert!(goal.output.contains("✓ hello.txt exists"), "{}", goal.output);
}

/// A 1x1 PNG committed to the repository, so runs have an image to show.
async fn with_image(d: &Path) {
    const DOT: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0x0d, 0x49, 0x48, 0x44, 0x52, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0,
        0, 0x1f, 0x15, 0xc4, 0x89, 0, 0, 0, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0, 0x1f, 0, 5,
        0, 1, 0xff, 0x89, 0x99, 0x3d, 0x1d, 0, 0, 0, 0, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];
    std::fs::write(d.join("dot.png"), DOT).unwrap();
    GitRepo::new(d).commit_all("image").await.unwrap();
}

fn seeing(model: &str) -> ModelRef {
    let mut m = local(model);
    m.options.insert("vision".into(), "true".into());
    m
}

#[tokio::test]
async fn a_text_agent_consults_a_vision_agent_about_a_screenshot() {
    let (d, engine) = setup(|_| {}).await;
    with_image(d.path()).await;
    let mut g = line(&["dev", "eyes"], "echo ok");
    g.edges.iter_mut().find(|e| e.from == "dev").unwrap().caps = EdgeCaps { consult: true, ..Default::default() };
    g.nodes.iter_mut().find(|n| n.id == "eyes").unwrap().model = Some(seeing("eyes"));
    let replies = [
        fin("go"),
        r#"{"tool": "consult", "args": {"agent": "eyes", "question": "what colour is it?", "images": ["dot.png"]}}"#.to_string(),
        r#"{"tool": "answer", "args": {"text": "a single red dot"}}"#.to_string(), // eyes, seeing the image
        fin("dev done"),
        fin("eyes done"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("look", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps[1].meta["comm"]["response"], "a single red dot");
    assert_eq!(steps[1].meta["comm"]["images"], json!(["dot.png"]));
    let seen = model.seen.lock().unwrap();
    assert!(seen[1].tools.iter().find(|t| t.name == "consult").unwrap().description.contains("images"));
    assert!(seen[1].system.contains("EYES can see images"), "the text agent is told who can see");
    assert_eq!(seen[2].messages[0].images.len(), 1, "the vision agent gets the image");
    assert_eq!(seen[2].messages[0].images[0].media_type, "image/png");
    assert!(!seen[1].tools.iter().any(|t| t.name == "view_image"), "a text model gets no view_image");
    assert!(seen[2].tools.iter().any(|t| t.name == "view_image"), "a vision model does");
}

#[tokio::test]
async fn images_are_not_sent_to_text_only_agents() {
    let (d, engine) = setup(|_| {}).await;
    with_image(d.path()).await;
    let mut g = line(&["dev", "blind"], "echo ok");
    g.edges.iter_mut().find(|e| e.from == "dev").unwrap().caps = EdgeCaps { consult: true, ..Default::default() };
    let replies = [
        fin("go"),
        r#"{"tool": "consult", "args": {"agent": "blind", "question": "what colour?", "images": ["dot.png"]}}"#.to_string(),
        fin("dev done"),
        fin("blind done"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("look", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert!(steps[1].observation.as_deref().unwrap().contains("cannot see images"), "{:?}", steps[1].observation);
    assert_eq!(steps[1].meta["ok"], false);
}

#[tokio::test]
async fn a_vision_agent_looks_at_an_image_itself() {
    let (d, engine) = setup(|_| {}).await;
    with_image(d.path()).await;
    let replies = [r#"{"tool": "view_image", "args": {"path": "dot.png"}}"#.to_string(), FINISH.to_string()];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(seeing("actor")), ..Default::default() };
    let run = engine.create_run("look at dot.png", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps[0].meta["image"], "dot.png");
    let seen = model.seen.lock().unwrap();
    assert!(seen[0].messages.iter().all(|m| m.images.is_empty()));
    assert_eq!(seen[1].messages.last().unwrap().images.len(), 1, "the image comes with the result");
}

#[tokio::test]
async fn a_looking_agent_must_look_before_it_reports() {
    let (d, engine) = setup(|_| {}).await;
    with_image(d.path()).await;
    let mut g = line(&["eyes"], "echo ok");
    let eyes = g.nodes.iter_mut().find(|n| n.id == "eyes").unwrap();
    eyes.model = Some(seeing("eyes"));
    eyes.task = "Take screenshots of the page and look at them.".into();
    let replies = [
        fin("go"),
        fin("The ocean looks great, waves move."), // nothing looked at
        r#"{"tool": "view_image", "args": {"path": "dot.png"}}"#.to_string(),
        fin("I saw one red dot on a clear background."),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("look", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps[1].meta["no_look"], true);
    assert_eq!(steps[1].meta["ok"], false);
    assert!(steps[1].observation.as_deref().unwrap().contains("not looked at any image"));
    assert_eq!(steps[3].meta["summary"], "I saw one red dot on a clear background.");
}

#[tokio::test]
async fn a_looking_agent_that_never_looks_is_marked_unverified_instead_of_stuck() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["eyes"], "echo ok");
    let eyes = g.nodes.iter_mut().find(|n| n.id == "eyes").unwrap();
    eyes.model = Some(seeing("eyes"));
    eyes.task = "Look at the screenshot.".into();
    let replies = [fin("go"), fin("all fine"), fin("all fine here"), fin("everything fine")];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("look", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps[3].meta["summary"], "[UNVERIFIED: EYES did not look at any image] everything fine");
}

#[tokio::test]
async fn a_blind_agent_cannot_claim_it_saw_the_page() {
    let (_d, engine) = setup(|_| {}).await;
    let g = line(&["tester"], "echo ok");
    let replies = [fin("go"), fin("No tests present; verified visually."), fin("Static checks pass; the look was not checked.")];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("check", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps[1].meta["false_claim"], true);
    assert!(steps[1].observation.as_deref().unwrap().contains("no-false-claims"), "{:?}", steps[1].observation);
    assert_eq!(steps[2].meta["summary"], "Static checks pass; the look was not checked.");
    let seen = model.seen.lock().unwrap();
    assert!(seen[1].system.contains("Honesty about evidence"), "the skill's rules are in the prompt");
}

#[tokio::test]
async fn a_vision_agent_compares_several_frames_at_once() {
    let (d, engine) = setup(|_| {}).await;
    with_image(d.path()).await;
    std::fs::copy(d.path().join("dot.png"), d.path().join("dot2.png")).unwrap();
    GitRepo::new(d.path()).commit_all("second frame").await.unwrap();
    let replies = [r#"{"tool": "view_image", "args": {"paths": ["dot.png", "dot2.png"]}}"#.to_string(), FINISH.to_string()];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(seeing("actor")), ..Default::default() };
    let run = engine.create_run("compare frames", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(steps[0].meta["images"], json!(["dot.png", "dot2.png"]));
    assert_eq!(model.seen.lock().unwrap()[1].messages.last().unwrap().images.len(), 2);
}

#[tokio::test]
async fn screen_capture_is_off_unless_enabled() {
    let d = tempfile::tempdir().unwrap();
    GitRepo::new(d.path()).init().await.unwrap();
    let names = |limits: &crate::config::Limits| {
        tools::ToolBox::new(d.path(), limits, vec![]).unwrap().schemas().into_iter().map(|t| t.name).collect::<Vec<_>>()
    };
    let off = names(&Default::default());
    assert!(!off.contains(&"screenshot_screen".to_string()) && off.contains(&"screenshot_page".to_string()));
    let on = names(&crate::config::Limits { allow_screen_capture: true, ..Default::default() });
    assert!(on.contains(&"screenshot_screen".to_string()));
    assert_eq!(tools::ToolBox::new(d.path(), &Default::default(), vec![]).unwrap().class("screenshot_screen"), tools::ToolClass::Exec);
}

#[tokio::test]
async fn page_screenshots_are_saved_outside_the_checkpoints() {
    if crate::graph::browser::find_browser().is_none() {
        eprintln!("skipped: no Edge/Chrome here");
        return;
    }
    let (_d, engine) = setup(|_| {}).await;
    let replies = [
        r#"{"tool": "write_file", "args": {"path": "index.html", "content": "<h1 style='color:red'>hi</h1>"}}"#.to_string(),
        r#"{"tool": "screenshot_page", "args": {"url": "index.html", "name": "home"}}"#.to_string(),
        FINISH.to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(local("actor")), ..Default::default() };
    let run = engine.create_run("shoot", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(steps[1].meta["image"], ".orchopork-shots/home.png", "{:?}", steps[1].observation);
    assert!(Path::new(&run.worktree).join(".orchopork-shots/home.png").is_file());
    let (stat, _) = engine.diff(&run.id).await.unwrap();
    assert!(stat.contains("index.html") && !stat.contains("orchopork-shots"), "{stat}");
}

#[tokio::test]
async fn identical_calls_with_nothing_changed_are_not_run_again() {
    let (_d, engine) = setup(|_| {}).await;
    let read = r#"{"tool": "read_file", "args": {"path": "README.md"}}"#;
    let replies = [read, read, FINISH];
    let model = Scripted::new(ProviderId::Ollama, &replies);
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(local("actor")), ..Default::default() };
    let run = engine.create_run("x", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(steps[0].meta["ok"], true);
    assert_eq!(steps[1].meta["repeat"], true);
    assert!(steps[1].observation.as_deref().unwrap().starts_with("Not run"), "{:?}", steps[1].observation);
    assert!(steps[1].observation.as_deref().unwrap().contains("# demo"), "the earlier result is repeated for it");
}

#[tokio::test]
async fn repeat_notices_do_not_nest_and_process_changes_allow_a_retry() {
    let (_d, engine) = setup(|_| {}).await;
    let read = r#"{"tool": "read_file", "args": {"path": "README.md"}}"#;
    let start = r#"{"tool": "start_process", "args": {"command": "echo up", "wait_secs": 0}}"#;
    let replies = [read, read, read, read, start, read, FINISH];
    let model = Scripted::new(ProviderId::Ollama, &replies);
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(local("actor")), ..Default::default() };
    let run = engine.create_run("x", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let obs = |i: usize| steps[i].observation.clone().unwrap_or_default();
    assert_eq!(obs(3).matches("Not run").count(), 1, "one notice, not nested ones: {}", obs(3));
    assert!(obs(3).contains("attempt 4") && obs(3).contains("# demo"), "{}", obs(3));
    assert_eq!(steps[5].meta.get("repeat"), None, "a process started since: the read runs again");
}

#[tokio::test]
async fn repeating_the_same_blocked_call_hands_the_work_back() {
    let (_d, engine) = setup(|_| {}).await;
    let read = r#"{"tool": "read_file", "args": {"path": "README.md"}}"#.to_string();
    let replies = [
        fin("go"),
        read.clone(),
        read.clone(),
        read.clone(),
        read.clone(),
        r#"{"tool": "rerun", "args": {"agent": "dev", "instruction": "stop re-reading; finish"}}"#.to_string(),
        fin("lead again"),
        fin("dev done"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(line(&["dev"], "echo ok"))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let back = steps.iter().find(|s| s.meta["tool"] == "handback").expect("handed back");
    assert!(back.output.contains("kept repeating"), "{}", back.output);
}

#[tokio::test]
async fn repeated_errors_trigger_help_and_become_team_dead_ends() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev", "helper"], "echo ok");
    g.edges.iter_mut().find(|e| e.from == "dev").unwrap().caps = EdgeCaps { consult: true, ..Default::default() };
    g.nodes.push(fnode("notes", NodeKind::Notebook, ""));
    g.edges.push(fedge("notes", "dev", EdgeCaps::default()));
    let bad_edit = r#"{"tool": "edit_file", "args": {"path": "README.md", "old": "no such text", "new": "x"}}"#.to_string();
    let write = |i: usize| format!(r#"{{"tool": "write_file", "args": {{"path": "f{i}.txt", "content": "{i}"}}}}"#);
    let mut replies = vec![fin("go")];
    for i in 0..4 {
        replies.push(write(i)); // something changes, so the identical edit is run again
        replies.push(bad_edit.clone());
    }
    replies.push(r#"{"tool": "finish", "args": {"summary": "gave up on that edit", "notes": "none"}}"#.to_string());
    replies.push(fin("helper done"));
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let edits: Vec<&Step> = steps.iter().filter(|s| s.meta["tool"] == "edit_file").collect();
    assert_eq!(edits.len(), 4);
    assert!(!edits[1].observation.as_deref().unwrap().contains("STUCK"));
    let third = edits[2].observation.as_deref().unwrap();
    assert!(third.contains("STUCK") && third.contains("consult HELPER"), "{third}");
    assert_eq!(edits[3].meta["dead_end"], true, "the fourth identical failure is a dead end");
    assert!(edits[3].meta["notes"][0]["entry"].as_str().unwrap().contains("Dead end (tried 4 times by DEV"));
    let seen = model.seen.lock().unwrap();
    assert!(ctx(&seen[1]).contains("# Who can help you") && ctx(&seen[1]).contains("HELPER (helper)"), "{}", ctx(&seen[1]));
    assert!(ctx(&seen[1]).contains("Text only: cannot see images"));
}

#[tokio::test]
async fn file_tools_report_lines_added_and_removed() {
    let (_d, engine) = setup(|_| {}).await;
    let replies = [
        r#"{"tool": "write_file", "args": {"path": "a.txt", "content": "one\ntwo\nthree\n"}}"#,
        r#"{"tool": "edit_file", "args": {"path": "a.txt", "old": "two\n", "new": "2\n2b\n"}}"#,
        FINISH,
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies);
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(local("actor")), ..Default::default() };
    let run = engine.create_run("x", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(steps[0].meta["lines"], json!({ "added": 3, "removed": 0 }));
    assert_eq!(steps[1].meta["lines"], json!({ "added": 2, "removed": 1 }));
    assert_eq!(GitRepo::new(&run.worktree).numstat_from(&run.base_commit).await.unwrap(), (4, 0));
}

#[tokio::test]
async fn a_run_can_be_killed_in_the_middle_of_a_step() {
    let (_d, engine) = setup(|_| {}).await;
    let model = Scripted::new(ProviderId::Ollama, &[WRITE, FINISH]);
    model.stall_call(1, Duration::from_secs(30));
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(local("actor")), ..Default::default() };
    let run = engine.create_run("x", None, spec).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(engine.is_active(&run.id));
    let t = std::time::Instant::now();
    let killed = engine.kill(&run.id).await.unwrap();
    assert!(t.elapsed() < Duration::from_secs(2), "no waiting for the step");
    assert!(!engine.is_active(&run.id));
    assert_eq!((killed.status, killed.error.as_deref()), (RunStatus::Paused, Some("killed by the user")));
    assert!(engine.run_detail(&run.id).await.unwrap().1.is_empty(), "the unfinished step left nothing behind");
    engine.start(&run.id).await.unwrap(); // and it can be resumed
    assert!(engine.is_active(&run.id));
    engine.kill(&run.id).await.unwrap();
}

#[tokio::test]
async fn each_step_has_its_own_diff() {
    let (_d, engine) = setup(|_| {}).await;
    let model = Scripted::new(ProviderId::Ollama, &[WRITE, r#"{"tool": "read_file", "args": {"path": "hello.txt"}}"#, FINISH]);
    register(&engine, &model);
    let spec = RunSpec { mode: RunMode::Solo, model: Some(local("actor")), ..Default::default() };
    let run = engine.create_run("x", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let (stat, patch) = engine.step_diff(&run.id, steps[0].seq).await.unwrap();
    assert!(stat.contains("hello.txt") && patch.contains("+hello"), "{stat}");
    let (stat, _) = engine.step_diff(&run.id, steps[1].seq).await.unwrap();
    assert!(stat.trim().is_empty(), "reading changes nothing: {stat}");
}

#[tokio::test]
async fn a_stuck_agent_hands_its_work_back_to_the_lead() {
    let (_d, engine) = setup(|c| c.limits.node_turns = 3).await;
    let replies = [
        fin("go"),
        r#"{"tool": "list_files", "args": {}}"#.to_string(),
        r#"{"tool": "list_dir", "args": {}}"#.to_string(),
        r#"{"tool": "read_file", "args": {"path": "README.md"}}"#.to_string(),
        // three turns used: the Lead takes over
        r#"{"tool": "rerun", "args": {"agent": "dev", "instruction": "stop exploring and just finish"}}"#.to_string(),
        fin("lead again"),
        fin("dev finished as told"),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(line(&["dev"], "echo ok"))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "dev", "dev", "dev", "dev", "lead", "lead", "dev", "goal"]);
    assert_eq!(steps[4].meta["tool"], "handback");
    let seen = model.seen.lock().unwrap();
    assert!(ctx(&seen[4]).contains("hands the problem back") && ctx(&seen[4]).contains("rerun dev"), "{}", ctx(&seen[4]));
    assert!(ctx(&seen[6]).contains("stop exploring and just finish"));
}

#[tokio::test]
async fn preflight_finds_broken_models_and_design_mistakes() {
    let (_d, engine) = setup(|_| {}).await;
    let model = Scripted::new(ProviderId::Ollama, &[]);
    register(&engine, &model);
    let mut g = line(&["eyes", "dev"], "");
    let gs = g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap();
    gs.judge = GoalJudge::Agent;
    gs.verify_command = None;
    gs.criteria = "the ocean looks realistic".into();
    g.nodes.iter_mut().find(|n| n.id == "dev").unwrap().task = "build it, then ask EYES whether it looks right".into();
    g.nodes.iter_mut().find(|n| n.id == "eyes").unwrap().model = Some(cloud()); // Claude is not configured here
    g.nodes.push(fnode("notes", NodeKind::Notebook, ""));
    g.edges.push(fedge("notes", "dev", EdgeCaps::default()));
    let r = engine.preflight(&orch(g), None).await.unwrap();
    assert_eq!(r["ok"], false);
    let errors = r["errors"].to_string();
    assert!(errors.contains("EYES") && errors.contains("cannot be used"), "{errors}");
    assert_eq!(r["nodes"]["dev"]["ok"], true);
    let warnings = r["warnings"].to_string();
    assert!(warnings.contains("no Consult/Suggest link"), "{warnings}");
    assert!(warnings.contains("vague"), "{warnings}");
    assert!(warnings.contains("connected to 1 agent"), "{warnings}");
    assert!(warnings.contains("visual criteria but its judge cannot see"), "{warnings}");
}

#[tokio::test]
async fn lessons_are_proposed_from_failures_and_become_a_project_skill() {
    let (_d, engine) = setup(|_| {}).await;
    let replies = [
        fin("go"),
        fin("dev done"),
        "- When three.js addons are imported, add an import map for \"three\".\n- When a page is blank, open it in the browser tool first.".to_string(),
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let mut g = line(&["dev"], "test -f missing.txt");
    g.nodes.iter_mut().find(|n| n.kind == NodeKind::Goal).unwrap().goal.as_mut().unwrap().max_rounds = 1;
    let run = engine.create_run("x", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Paused, "the goal failed");
    let lessons = engine.suggest_lessons(&run.id).await.unwrap();
    assert_eq!(lessons.len(), 2, "{lessons:?}");
    assert!(lessons[0].starts_with("When three.js addons"));
    assert_eq!(engine.save_lessons(&lessons).unwrap(), 2);
    assert_eq!(engine.save_lessons(&lessons[..1]).unwrap(), 2, "no duplicates");
    let composed = engine.workspace().skills.compose("").unwrap();
    assert!(composed.system.contains("add an import map for \"three\""), "later agents read the lessons");
}

/// Polls until `pred` holds for the run's steps, or panics.
#[allow(dead_code)]
async fn wait_for(engine: &Engine, id: &str, mut pred: impl FnMut(&[Step]) -> bool) {
    for _ in 0..500 {
        let (_, steps) = engine.run_detail(id).await.unwrap();
        if pred(&steps) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("condition not met for run {id}");
}

#[tokio::test]
async fn a_message_to_a_finished_agent_reruns_it_and_what_follows() {
    let (_d, engine) = setup(|_| {}).await;
    let replies = [fin("go"), WRITE.to_string(), fin("hello written"), fin("again, with the note")];
    let model = Scripted::new(ProviderId::Ollama, &replies.iter().map(String::as_str).collect::<Vec<_>>());
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(line(&["dev"], "test -f hello.txt"))).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Done);
    engine.inject(&run.id, "please add a note", Some("dev")).await.unwrap();
    engine.start(&run.id).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "dev", "dev", "goal", "dev", "dev", "goal"]);
    assert!(ctx(&model.seen.lock().unwrap()[3]).contains("please add a note"));
}

/// Minimal ACP agent: writes `<name>.txt` through the client and reports.
#[cfg(unix)]
const FAKE_ACP: &str = r#"
import json, sys
def send(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
name = sys.argv[1]
while True:
    line = sys.stdin.readline()
    if not line: break
    m = json.loads(line)
    meth = m.get("method")
    if meth == "initialize": send({"jsonrpc": "2.0", "id": m["id"], "result": {"protocolVersion": 1}})
    elif meth == "session/new": send({"jsonrpc": "2.0", "id": m["id"], "result": {"sessionId": "s"}})
    elif meth == "session/prompt":
        text = m["params"]["prompt"][0]["text"]
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "s", "update": {"sessionUpdate": "tool_call", "toolCallId": "1", "title": "Write " + name, "kind": "edit", "status": "completed"}}})
        send({"jsonrpc": "2.0", "id": 900, "method": "fs/write_text_file", "params": {"sessionId": "s", "path": name + ".txt", "content": "saw goal: " + str("Mission" in text or "Goal" in text)}})
        sys.stdin.readline()
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "s", "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": name + " report: file written"}}}})
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"stopReason": "end_turn"}})
"#;

#[cfg(unix)]
fn fake_acp(dir: &Path, id: &str) -> crate::agents::ExternalAgent {
    let script = dir.join("fake_acp.py");
    std::fs::write(&script, FAKE_ACP).unwrap();
    crate::agents::ExternalAgent::new(id, &format!("Fake {id}"), "python3", &[&script.to_string_lossy(), id])
}

#[cfg(unix)]
#[tokio::test]
async fn solo_chat_with_an_external_acp_agent() {
    let scripts = tempfile::tempdir().unwrap();
    let agent = fake_acp(scripts.path(), "ext");
    let (_d, engine) = setup(|c| c.external_agents = vec![agent]).await;
    let model = ModelRef::new(ProviderId::Acp, "ext");
    let spec = RunSpec { autonomy: Default::default(), graph: None, worker_model: None, mode: RunMode::Solo, model: Some(model), agents: vec![] };
    let run = engine.create_run("write a file", Some("test -f ext.txt"), spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(kinds(&steps), ["act", "verify"]);
    assert_eq!(steps[0].output, "ext report: file written");
    assert_eq!(steps[0].meta["external"]["tool_calls"][0]["title"], "Write ext");
    let body = std::fs::read_to_string(Path::new(&run.worktree).join("ext.txt")).unwrap();
    assert_eq!(body, "saw goal: True");
}

#[cfg(unix)]
#[tokio::test]
async fn external_agents_are_flow_nodes_too() {
    let scripts = tempfile::tempdir().unwrap();
    let ext = fake_acp(scripts.path(), "coder");
    let (_d, engine) = setup(|c| c.external_agents = vec![ext]).await;
    let lead = Scripted::new(ProviderId::Ollama, &[r#"{"tool": "finish", "args": {"summary": "coder: write coder.txt"}}"#]);
    register(&engine, &lead);
    let mut g = line(&["coder"], "test -f coder.txt");
    g.nodes.iter_mut().find(|n| n.id == "coder").unwrap().model = Some(ModelRef::new(ProviderId::Acp, "coder"));
    let run = engine.create_run("make files", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "coder", "goal"]);
    assert_eq!(steps[1].meta["model"], "acp:coder");
    assert!(Path::new(&run.worktree).join("coder.txt").exists());
}

async fn wait_approval(engine: &Engine, id: &str) -> approvals::Approval {
    for _ in 0..500 {
        if let Some(a) = engine.pending_approval(id) {
            return a;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("run {id} never asked for approval");
}

fn solo(autonomy: Autonomy) -> RunSpec {
    RunSpec { mode: RunMode::Solo, model: Some(local("actor")), autonomy, ..Default::default() }
}

#[tokio::test]
async fn semi_autonomy_edits_freely_but_asks_before_deleting() {
    let (_d, engine) = setup(|_| {}).await;
    let delete = r#"{"tool": "delete_file", "args": {"path": "hello.txt"}}"#;
    let actor = Scripted::new(ProviderId::Ollama, &[WRITE, delete, FINISH]);
    register(&engine, &actor);
    let run = engine.create_run("write then delete", None, solo(Autonomy::Semi)).await.unwrap();

    let a = wait_approval(&engine, &run.id).await;
    assert_eq!((a.tool.as_str(), a.class), ("delete_file", "exec"));
    assert_eq!(a.args["path"], "hello.txt");
    assert!(engine.decide(&run.id, "wrong-id", true, false, "").is_err());
    engine.decide(&run.id, &a.id, false, false, "keep the file").unwrap();

    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].meta["tool"], "write_file", "writes do not ask in semi mode");
    assert_eq!(steps[1].meta["declined"], true);
    assert!(steps[1].observation.as_deref().unwrap().contains("keep the file"));
    assert_eq!(steps[1].state.failures, 0, "a declined call is not the model's failure");
    assert!(Path::new(&run.worktree).join("hello.txt").exists());
    assert!(engine.pending_approval(&run.id).is_none());
}

#[tokio::test]
async fn manual_mode_asks_for_writes_and_switching_to_full_releases_the_call() {
    let (_d, engine) = setup(|_| {}).await;
    let actor = Scripted::new(ProviderId::Ollama, &[WRITE, FINISH]);
    register(&engine, &actor);
    let run = engine.create_run("write a file", None, solo(Autonomy::Manual)).await.unwrap();
    let a = wait_approval(&engine, &run.id).await;
    assert_eq!(a.class, "write");
    engine.set_autonomy(&run.id, Autonomy::Full).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(run.spec.autonomy, Autonomy::Full);
    assert!(Path::new(&run.worktree).join("hello.txt").exists());
}

#[tokio::test]
async fn pausing_while_waiting_does_not_run_the_call() {
    let (_d, engine) = setup(|_| {}).await;
    let actor = Scripted::new(ProviderId::Ollama, &[WRITE]);
    register(&engine, &actor);
    let run = engine.create_run("write a file", None, solo(Autonomy::Manual)).await.unwrap();
    wait_approval(&engine, &run.id).await;
    engine.pause(&run.id).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Paused);
    assert!(steps[0].observation.as_deref().unwrap().contains("paused"));
    assert!(!Path::new(&run.worktree).join("hello.txt").exists());
}

/// A provider speaking native tool calls: its first successful reply is a
/// structured `write_file` call with no text. With `reject_tools`, any
/// request carrying tools fails the way Ollama does for models without a
/// tool template.
struct Native {
    reject_tools: bool,
    seen: Mutex<Vec<CompletionRequest>>,
    answered: Mutex<usize>,
}

struct NativeHandle(Arc<Native>);

#[async_trait::async_trait]
impl Provider for NativeHandle {
    fn id(&self) -> ProviderId {
        ProviderId::Ollama
    }
    async fn complete(&self, _model: &str, req: &CompletionRequest) -> Result<Completion> {
        self.0.seen.lock().unwrap().push(req.clone());
        if self.0.reject_tools && !req.tools.is_empty() {
            return Err(Error::Provider("ollama: HTTP 400: registry.ollama.ai/library/x does not support tools".into()));
        }
        let n = {
            let mut a = self.0.answered.lock().unwrap();
            *a += 1;
            *a
        };
        Ok(if n == 1 {
            Completion {
                tool_call: Some(("write_file".into(), json!({ "path": "n.txt", "content": "native\n" }))),
                ..Default::default()
            }
        } else {
            Completion { text: FINISH.into(), ..Default::default() }
        })
    }
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![])
    }
}

async fn native_run(reject_tools: bool) -> (Arc<Native>, Run, Vec<Step>, tempfile::TempDir) {
    let (d, engine) = setup(|_| {}).await;
    let p = Arc::new(Native { reject_tools, seen: Mutex::new(vec![]), answered: Mutex::new(0) });
    engine.workspace().gateway.register(Box::new(NativeHandle(p.clone())));
    let run = engine.create_run("write n.txt", None, solo(Autonomy::Full)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    (p, run, steps, d)
}

#[tokio::test]
async fn native_tool_calls_drive_the_same_protocol() {
    let (p, run, steps, _d) = native_run(false).await;
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(steps[0].meta["tool"], "write_file");
    assert!(steps[0].output.contains(r#""tool":"write_file""#), "{}", steps[0].output);
    assert_eq!(std::fs::read_to_string(Path::new(&run.worktree).join("n.txt")).unwrap(), "native\n");
    let seen = p.seen.lock().unwrap();
    let names: Vec<&str> = seen[0].tools.iter().map(|t| t.name.as_str()).collect();
    for want in ["read_file", "write_file", "edit_file", "run_command", "start_process", "browser", "web_search", "finish"] {
        assert!(names.contains(&want), "{want} missing from {names:?}");
    }
}

#[tokio::test]
async fn endpoints_without_tool_support_fall_back_to_the_text_protocol() {
    let (p, run, _steps, _d) = native_run(true).await;
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let seen = p.seen.lock().unwrap();
    assert!(!seen[0].tools.is_empty(), "tools are offered first");
    assert!(seen[1..].iter().all(|r| r.tools.is_empty()), "then never again for that model");
    assert!(Path::new(&run.worktree).join("n.txt").exists());
}

// ---------- Dispatcher: parallel branches ----------

/// Replies per agent (read from the prompt's "(id X)"), so branches that
/// run at the same time each get their own script. Slow agents wait before
/// every reply.
struct ByAgent {
    replies: Mutex<HashMap<String, VecDeque<String>>>,
    slow: HashMap<String, Duration>,
    seen: Mutex<Vec<(String, CompletionRequest)>>,
}

struct ByAgentHandle(Arc<ByAgent>);

fn agent_in(req: &CompletionRequest) -> String {
    let text = ctx(req);
    let Some(i) = text.find("(id ") else { return String::new() };
    text[i + 4..].split(')').next().unwrap_or("").to_string()
}

#[async_trait::async_trait]
impl Provider for ByAgentHandle {
    fn id(&self) -> ProviderId {
        ProviderId::Ollama
    }
    async fn complete(&self, _model: &str, req: &CompletionRequest) -> Result<Completion> {
        if req.system.starts_with("You maintain the working memory") {
            return Ok(Completion { text: "SUMMARY: earlier work".into(), ..Default::default() });
        }
        let who = agent_in(req);
        self.0.seen.lock().unwrap().push((who.clone(), req.clone()));
        if let Some(d) = self.0.slow.get(&who) {
            tokio::time::sleep(*d).await;
        }
        let text = self
            .0
            .replies
            .lock()
            .unwrap()
            .get_mut(&who)
            .and_then(VecDeque::pop_front)
            .unwrap_or_else(|| fin(&format!("{who} done")));
        Ok(Completion { text, prompt_tokens: 100, completion_tokens: 50, ..Default::default() })
    }
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![])
    }
}

fn by_agent(engine: &Engine, script: &[(&str, &[&str])], slow: &[(&str, u64)]) -> Arc<ByAgent> {
    let p = Arc::new(ByAgent {
        replies: Mutex::new(script.iter().map(|(a, r)| (a.to_string(), r.iter().map(|s| s.to_string()).collect())).collect()),
        slow: slow.iter().map(|(a, ms)| (a.to_string(), Duration::from_millis(*ms))).collect(),
        seen: Mutex::new(vec![]),
    });
    engine.workspace().gateway.register(Box::new(ByAgentHandle(p.clone())));
    p
}

/// Lead → Dispatcher (a branch per list of agents) → Goal (`check`).
fn fork(policy: crate::storage::ParallelPolicy, work: crate::storage::ParallelWork, branches: &[&[&str]], check: &str) -> FlowGraph {
    let mut g = line(&[], check);
    g.edges.clear();
    let mut par = fnode("par", NodeKind::Parallel, "combine the branches");
    par.parallel = Some(crate::storage::ParallelSettings { policy, work });
    g.nodes.push(par);
    g.edges.push(FlowEdge::new("lead", "par"));
    g.edges.push(FlowEdge::new("par", "goal"));
    for (i, b) in branches.iter().enumerate() {
        for (j, a) in b.iter().enumerate() {
            let mut n = fnode(a, NodeKind::Agent, &format!("{a}'s part"));
            n.pos = Some([300.0 + j as f64 * 300.0, i as f64 * 200.0]);
            g.nodes.push(n);
            if j == 0 {
                g.edges.push(FlowEdge::branch("par", a));
            } else {
                g.edges.push(FlowEdge::new(b[j - 1], a));
            }
        }
    }
    g
}

fn write(path: &str, content: &str) -> String {
    json!({ "tool": "write_file", "args": { "path": path, "content": content } }).to_string()
}

async fn wait_idle_long(engine: &Engine, id: &str) -> Run {
    for _ in 0..6000 {
        if !engine.is_active(id) {
            return engine.run_detail(id).await.unwrap().0;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let (_, steps) = engine.run_detail(id).await.unwrap();
    let trail: Vec<String> = steps.iter().map(|s| format!("{} {} {}", s.seq, s.meta["agent"], s.meta["tool"])).collect();
    panic!("run {id} did not stop; steps: {}", trail.join(", "));
}

fn step_of<'a>(steps: &'a [Step], who: &str, tool: &str) -> Option<&'a Step> {
    steps.iter().find(|s| s.meta["agent"] == who && s.meta["tool"] == tool)
}

#[tokio::test]
async fn a_dispatcher_runs_branches_in_parallel_and_merges_them() {
    use crate::storage::{ParallelPolicy, ParallelWork};
    let (d, engine) = setup(|_| {}).await;
    let g = fork(ParallelPolicy::WaitAll, ParallelWork::Code, &[&["a"], &["b", "b2"]], "test -f a.txt && test -f b.txt && test -f b2.txt");
    let (a_write, b_write, b2_write) = (write("a.txt", "a\n"), write("b.txt", "b\n"), write("b2.txt", "b2\n"));
    // a is slow: the other branch works meanwhile.
    let model = by_agent(
        &engine,
        &[
            ("lead", &[&fin("split it")]),
            ("a", &[&a_write, &fin("a.txt written")]),
            ("b", &[&b_write, &fin("b.txt written")]),
            ("b2", &[&b2_write, &fin("b2.txt written")]),
        ],
        &[("a", 2500)],
    );
    let run = engine.create_run("a and b", None, orch(g)).await.unwrap();
    let run = wait_idle_long(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let seq = |who: &str, tool: &str| step_of(&steps, who, tool).unwrap_or_else(|| panic!("{who} {tool} missing")).seq;
    assert!(seq("b2", "finish") < seq("a", "finish"), "the branches interleave");
    assert!(seq("par", "dispatch") < seq("b", "write_file"));
    assert!(seq("a", "finish") < seq("par", "finish"));
    assert_eq!(steps.last().unwrap().kind, StepKind::Verify);
    // Branch steps checkpoint their branch; the run's worktree got the merge.
    let b_step = step_of(&steps, "b", "write_file").unwrap();
    assert_eq!(b_step.meta["branch"]["head"], "b");
    let wt = Path::new(&run.worktree);
    for f in ["a.txt", "b.txt", "b2.txt"] {
        assert!(wt.join(f).exists(), "{f} merged");
    }
    let par_fin = step_of(&steps, "par", "finish").unwrap();
    assert_eq!(par_fin.meta["parallel"]["merged"].as_array().unwrap().len(), 2);
    assert!(par_fin.state.flow.parallel.is_none());
    // b starts from the Lead's handoff (via the Dispatcher), b2 from b's report.
    {
        let seen = model.seen.lock().unwrap();
        let first = |who: &str| ctx(&seen.iter().find(|(w, _)| w == who).unwrap().1);
        assert!(first("b").contains("started parallel branches") && first("b").contains("split it"), "{}", first("b"));
        assert!(first("b2").contains("Handoff from B") && first("b2").contains("b.txt written"));
        assert!(first("a").contains("parallel branch"), "branch agents know they run in parallel");
    }
    // The branch worktrees are gone.
    let left: Vec<String> =
        std::fs::read_dir(d.path().join(".orchopork/worktrees")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
    assert_eq!(left, std::slice::from_ref(&run.id));
    // A branch step's diff is the branch's change.
    let (stat, _) = engine.step_diff(&run.id, b_step.seq).await.unwrap();
    assert!(stat.contains("b.txt"), "{stat}");
}

#[tokio::test]
async fn the_dispatcher_resolves_merge_conflicts() {
    use crate::storage::{ParallelPolicy, ParallelWork};
    let (_d, engine) = setup(|_| {}).await;
    let g = fork(ParallelPolicy::WaitAll, ParallelWork::Code, &[&["a"], &["b"]], "grep -q A shared.txt && grep -q B shared.txt");
    let (wa, wb) = (write("shared.txt", "A\n"), write("shared.txt", "B\n"));
    let resolve = write("shared.txt", "A\nB\n");
    let done = r#"{"tool": "resolved", "args": {"summary": "kept both lines"}}"#;
    by_agent(&engine, &[("a", &[&wa, &fin("A")]), ("b", &[&wb, &fin("B")]), ("par", &[&resolve, done])], &[("b", 150)]);
    let run = engine.create_run("both", None, orch(g)).await.unwrap();
    let run = wait_idle_long(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(std::fs::read_to_string(Path::new(&run.worktree).join("shared.txt")).unwrap(), "A\nB\n");
    let par_fin = step_of(&steps, "par", "finish").unwrap();
    assert!(par_fin.output.contains("conflicts in shared.txt resolved"), "{}", par_fin.output);
}

#[tokio::test]
async fn first_success_cancels_the_slower_branch() {
    use crate::storage::{ParallelPolicy, ParallelWork};
    let (_d, engine) = setup(|_| {}).await;
    let g = fork(ParallelPolicy::FirstSuccess, ParallelWork::Code, &[&["fast"], &["slow"]], "test -f fast.txt");
    let (wf, wsl) = (write("fast.txt", "f\n"), write("slow.txt", "s\n"));
    by_agent(&engine, &[("fast", &[&wf, &fin("fast done")]), ("slow", &[&wsl, &fin("slow done")])], &[("slow", 1500)]);
    let run = engine.create_run("race", None, orch(g)).await.unwrap();
    let run = wait_idle_long(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert!(step_of(&steps, "slow", "finish").is_none(), "slow was cancelled");
    assert!(!Path::new(&run.worktree).join("slow.txt").exists());
    let par_fin = step_of(&steps, "par", "finish").unwrap();
    assert!(par_fin.output.contains("cancelled"), "{}", par_fin.output);
}

#[tokio::test]
async fn a_managing_dispatcher_waits_then_goes_on() {
    use crate::storage::{ParallelPolicy, ParallelWork};
    let (_d, engine) = setup(|_| {}).await;
    let g = fork(ParallelPolicy::Manager, ParallelWork::Research, &[&["r1"], &["r2"], &["r3"]], "true");
    let wait = r#"{"tool": "decide", "args": {"decision": "wait", "reason": "need more"}}"#;
    let go = r#"{"tool": "decide", "args": {"decision": "continue", "merge": ["r1", "r2"], "reason": "two answers agree"}}"#;
    let scratch = write("scratch.txt", "x");
    let model = by_agent(
        &engine,
        &[("r1", &[&scratch, &fin("answer 42")]), ("r2", &[&fin("also 42")]), ("r3", &[&fin("never")]), ("par", &[wait, go])],
        // r3 is far behind: the Dispatcher goes on without it.
        &[("r1", 50), ("r2", 3000), ("r3", 60_000)],
    );
    let run = engine.create_run("find the answer", None, orch(g)).await.unwrap();
    let run = wait_idle_long(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let decisions: Vec<bool> = steps.iter().filter(|s| s.meta["tool"] == "decide").map(|s| s.meta["decision"]["continue"] == true).collect();
    assert_eq!(decisions, [false, true]);
    assert!(step_of(&steps, "r3", "finish").is_none(), "r3 was cancelled");
    // Research: the branches' files are not merged.
    assert!(!Path::new(&run.worktree).join("scratch.txt").exists());
    {
        let seen = model.seen.lock().unwrap();
        let asks: Vec<String> = seen.iter().filter(|(w, _)| w == "par").map(|(_, r)| ctx(r)).collect();
        assert!(asks[0].contains("answer 42") && asks[0].contains("Still working"), "{}", asks[0]);
    }
    let par_fin = step_of(&steps, "par", "finish").unwrap();
    assert!(par_fin.output.contains("two answers agree") && par_fin.output.contains("research"), "{}", par_fin.output);
}

#[tokio::test]
async fn a_paused_dispatcher_resumes_its_branches_where_they_were() {
    use crate::storage::{ParallelPolicy, ParallelWork};
    let (_d, engine) = setup(|_| {}).await;
    let g = fork(ParallelPolicy::WaitAll, ParallelWork::Code, &[&["a"], &["b"]], "test -f a1.txt && test -f a2.txt && test -f b.txt");
    let (a1, a2, wb) = (write("a1.txt", "1"), write("a2.txt", "2"), write("b.txt", "b"));
    by_agent(&engine, &[("a", &[&a1, &a2, &fin("a done")]), ("b", &[&wb, &fin("b done")])], &[("a", 250)]);
    let run = engine.create_run("pause me", None, orch(g)).await.unwrap();
    // Pause once a has written a1.txt in its branch.
    for _ in 0..500 {
        let (_, steps) = engine.run_detail(&run.id).await.unwrap();
        if step_of(&steps, "a", "write_file").is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    engine.pause(&run.id).await.unwrap();
    let paused = wait_idle_long(&engine, &run.id).await;
    assert_eq!(paused.status, RunStatus::Paused, "error: {:?}", paused.error);
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    let par = steps.last().unwrap().state.flow.parallel.clone().expect("still in parallel");
    let a = par.branch("a").unwrap();
    assert!(!a.state.as_ref().unwrap().flow.done.contains(&"a".to_string()));
    engine.start(&run.id).await.unwrap();
    let run = wait_idle_long(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let a_writes = steps.iter().filter(|s| s.meta["agent"] == "a" && s.meta["tool"] == "write_file").count();
    assert_eq!(a_writes, 2, "a went on where it was, not from the start");
}

const WRITE_A: &str = r#"{"tool": "write_file", "args": {"path": "a.txt", "content": "x"}}"#;

/// The names of the tools a request offered.
fn offered(r: &CompletionRequest) -> Vec<String> {
    r.tools.iter().map(|t| t.name.clone()).collect()
}

#[tokio::test]
async fn a_lead_with_agents_plans_and_the_agents_do_the_work() {
    let (_d, engine) = setup(|_| {}).await;
    let g = line(&["dev"], "test -f a.txt");
    let replies = [WRITE_A, &fin("plan"), WRITE_A, &fin("done")];
    let model = Scripted::new(ProviderId::Ollama, &replies);
    register(&engine, &model);
    let run = engine.create_run("make a.txt", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(who(&steps), ["lead", "lead", "dev", "dev", "goal"]);
    // The Lead tried to write the file itself and was refused.
    assert_eq!(steps[0].meta["ok"], false);
    assert!(steps[0].observation.as_deref().unwrap_or("").contains("not available here"), "{:?}", steps[0].observation);
    let seen = model.seen.lock().unwrap();
    let lead = &seen[0];
    let tools = offered(lead);
    assert!(tools.contains(&"read_file".to_string()) && tools.contains(&"rerun".to_string()), "{tools:?}");
    assert!(!tools.iter().any(|t| ["write_file", "edit_file", "run_command", "delete_file"].contains(&t.as_str())), "{tools:?}");
    assert!(lead.system.contains("ROLE: LEAD") && lead.system.contains("name=\"team-protocol\""), "{}", lead.system);
    assert!(ctx(lead).contains("the work is NOT yours"));
    // The agent after it has its hands free and no Lead playbook.
    let dev = seen.iter().find(|r| offered(r).contains(&"write_file".to_string())).expect("dev got write_file");
    assert!(!dev.system.contains("ROLE: LEAD"));
}

#[tokio::test]
async fn a_lead_with_hands_on_or_without_agents_may_edit() {
    for (agents, hands_on) in [(vec!["dev"], true), (vec![], false)] {
        let (_d, engine) = setup(|_| {}).await;
        let mut g = line(&agents, "test -f a.txt");
        g.nodes[0].lead.as_mut().unwrap().hands_on = hands_on;
        let replies = [WRITE_A, &fin("done"), &fin("checked")];
        let model = Scripted::new(ProviderId::Ollama, &replies);
        register(&engine, &model);
        let run = engine.create_run("make a.txt", None, orch(g)).await.unwrap();
        let run = wait_idle(&engine, &run.id).await;
        let (_, steps) = engine.run_detail(&run.id).await.unwrap();
        assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
        assert_eq!(steps[0].meta["ok"], true, "{agents:?}: {:?}", steps[0].observation);
        assert!(offered(&model.seen.lock().unwrap()[0]).contains(&"write_file".to_string()));
    }
}

#[tokio::test]
async fn an_agents_internet_level_and_playbook_reach_its_prompt_and_tools() {
    let (_d, engine) = setup(|_| {}).await;
    let mut g = line(&["dev"], "echo ok");
    g.nodes[1].archetype = "coder".into();
    g.nodes[1].network = Some(crate::config::NetAccess::Off);
    let fetch = r#"{"tool": "web_fetch", "args": {"url": "https://example.com"}}"#;
    let replies = [&fin("plan") as &str, fetch, &fin("done")];
    let model = Scripted::new(ProviderId::Ollama, &replies);
    register(&engine, &model);
    let run = engine.create_run("x", None, orch(g)).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let seen = model.seen.lock().unwrap();
    let lead = &seen[0];
    assert!(offered(lead).contains(&"web_search".to_string()), "the Lead inherits the workspace's internet");
    let dev = seen.iter().find(|r| r.system.contains("ROLE: CODER")).expect("the coder playbook");
    let tools = offered(dev);
    assert!(!tools.iter().any(|t| ["web_search", "web_fetch", "browser", "screenshot_page"].contains(&t.as_str())), "{tools:?}");
    assert!(ctx(dev).contains("Internet access: none"), "{}", ctx(dev));
    let tried = steps.iter().find(|s| s.meta["tool"] == "web_fetch").expect("the call was made");
    assert_eq!(tried.meta["ok"], false);
    assert!(tried.observation.as_deref().unwrap_or("").contains("internet access"), "{:?}", tried.observation);
}

#[test]
fn a_node_gets_its_playbook_from_its_archetype_or_a_guess() {
    let node = |name: &str, role: &str, archetype: &str| {
        let mut n = FlowNode::new("x", NodeKind::Agent, name, "t");
        n.role = role.into();
        n.archetype = archetype.into();
        n.role_key()
    };
    assert_eq!(node("Tester", "writes and runs tests", ""), "tester");
    assert_eq!(node("Visual tester", "checks how it looks and moves", ""), "visual-tester");
    assert_eq!(node("Reviewer", "reviews for bugs", ""), "reviewer");
    assert_eq!(node("Researcher", "reads and maps code", ""), "researcher");
    assert_eq!(node("Docs writer", "updates documentation", ""), "docs");
    assert_eq!(node("Backend", "implements the API", ""), "coder");
    assert_eq!(node("Release manager", "tags builds", ""), "", "no guess, only the team rules");
    assert_eq!(node("Anything", "whatever", "Refactorer"), "refactorer", "an explicit archetype wins");
    assert_eq!(fnode("lead", NodeKind::Lead, "").role_key(), "lead");
    assert_eq!(fnode("par", NodeKind::Parallel, "").role_key(), "dispatcher");
}

#[test]
fn the_designer_may_set_archetype_and_internet_level() {
    let g = super::flow::parse_flow(
        r#"{"nodes": [{"id": "lead", "kind": "lead", "task": "plan"},
                      {"id": "r", "kind": "agent", "name": "R", "archetype": "Researcher", "network": "docs", "task": "look up the API"},
                      {"id": "c", "kind": "agent", "name": "C", "archetype": "wizard", "network": "everything", "task": "code"}],
            "edges": [{"from": "lead", "to": "r"}, {"from": "r", "to": "c"}, {"from": "c", "to": "goal"}]}"#,
        "g",
    );
    let r = g.node("r").unwrap();
    assert_eq!((r.archetype.as_str(), r.network), ("researcher", Some(crate::config::NetAccess::Docs)));
    let c = g.node("c").unwrap();
    assert_eq!((c.archetype.as_str(), c.network), ("", None), "unknown values are dropped");
}

#[tokio::test]
async fn the_designer_knows_the_workspace_and_its_flow_stays_within_it() {
    use crate::config::NetAccess;
    let (_d, engine) = setup(|c| {
        c.limits.network = NetAccess::Docs;
        c.limits.allow_commands = false;
    })
    .await;
    let skills = &engine.workspace().skills;
    std::fs::write(
        skills.dir().join("role-designer.yaml"),
        "name: role-designer\nversion: '1'\ntype: system_modifier\ndescription: Visual designer playbook.\nroles: [designer]\nprompt_injection: Mind the grid.\n",
    )
    .unwrap();
    skills.reload().unwrap();
    skills.set_enabled("role-designer", true).unwrap();
    let reply = r#"{"goal": {"criteria": "1. x", "verify_command": "npm test", "judge": "command"},
        "nodes": [{"id": "lead", "kind": "lead", "task": "plan", "lead": {"dynamic_agents": true, "max_dynamic": 2}},
                  {"id": "r", "kind": "agent", "name": "Scout", "archetype": "researcher", "network": "full", "task": "read"},
                  {"id": "ux", "kind": "agent", "name": "UX", "archetype": "designer", "task": "layout"},
                  {"id": "c", "kind": "agent", "name": "Backend", "role": "implements the API", "network": "off", "task": "code"}],
        "edges": [{"from": "lead", "to": "r"}, {"from": "r", "to": "ux", "caps": {"consult": true}, "context": "report", "call_limit": 50},
                  {"from": "ux", "to": "c"}, {"from": "c", "to": "goal"}]}"#;
    let model = Scripted::new(ProviderId::Ollama, &[reply]);
    register(&engine, &model);
    let g = engine.propose_flow("build the api", &local("lead")).await.unwrap();
    let seen = model.seen.lock().unwrap();
    let input = ctx(&seen[0]);
    assert!(input.contains("Internet maximum for agents: docs"), "{input}");
    assert!(input.contains("may NOT run shell commands") && input.contains("designer: Visual designer playbook."), "{input}");
    assert!(input.contains("# README.md"), "the project's docs reach the designer");
    assert!(seen[0].system.contains("The Lead investigates and plans only"));

    let r = g.node("r").unwrap();
    assert_eq!((r.archetype.as_str(), r.network), ("researcher", None), "full is above the workspace's docs: shown as the default");
    assert_eq!(g.node("ux").unwrap().archetype, "designer", "a workspace role key is accepted");
    let c = g.node("c").unwrap();
    assert_eq!((c.archetype.as_str(), c.network), ("coder", Some(NetAccess::Off)), "the playbook is guessed and spelled out");
    let e = g.edge("r", "ux").unwrap();
    assert_eq!((e.context, e.call_limit, e.caps.consult), (crate::storage::EdgeContext::Report, 10, true));
    let ls = g.lead_settings();
    assert_eq!((ls.dynamic_agents, ls.max_dynamic), (true, 2));
    let gs = g.goal_settings();
    assert_eq!((gs.verify_command, gs.judge), (None, GoalJudge::Agent), "nothing can run commands here");
}
