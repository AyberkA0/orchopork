//! End-to-end engine tests against scripted providers: real git worktrees,
//! real SQLite, real tools; only the LLM is fake.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::*;
use crate::providers::{Completion, Provider, ProviderId};

struct Scripted {
    id: ProviderId,
    replies: Mutex<VecDeque<String>>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl Scripted {
    fn new(id: ProviderId, replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            id,
            replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            seen: Mutex::new(vec![]),
        })
    }
}

struct Handle(Arc<Scripted>);

#[async_trait::async_trait]
impl Provider for Handle {
    fn id(&self) -> ProviderId {
        self.0.id
    }
    async fn complete(&self, _model: &str, req: &CompletionRequest) -> Result<Completion> {
        self.0.seen.lock().unwrap().push(req.clone());
        let text = self.0.replies.lock().unwrap().pop_front().unwrap_or_else(|| FINISH.to_string());
        Ok(Completion { text, prompt_tokens: 100, completion_tokens: 50, truncated: false })
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
    let spec = RunSpec { mode: RunMode::Solo, model: Some(cloud()), agents: vec![] };
    let run = engine.create_run("create hello.txt", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    assert_eq!(kinds(&steps), ["act", "act"]);
    assert_eq!(steps[0].meta["model"], "claude:claude-haiku-4-5");
}

fn agent(id: &str, parent: Option<&str>, task: &str) -> crate::storage::AgentSpec {
    crate::storage::AgentSpec {
        id: id.into(),
        name: id.to_uppercase(),
        role: "r".into(),
        task: task.into(),
        parent: parent.map(Into::into),
        model: None,
    }
}

#[tokio::test]
async fn orchestra_works_bottom_up_with_delegation_and_reports() {
    let (_d, engine) = setup(|_| {}).await;
    // Order of turns: coder (leaf) writes + finishes, then lead delegates back,
    // coder fixes + finishes, lead finishes.
    let replies = [
        WRITE,
        r#"{"tool": "finish", "args": {"summary": "hello.txt written"}}"#,
        r#"{"tool": "delegate", "args": {"agent": "coder", "instruction": "also add bye.txt"}}"#,
        r#"{"tool": "write_file", "args": {"path": "bye.txt", "content": "bye"}}"#,
        r#"{"tool": "finish", "args": {"summary": "bye.txt added"}}"#,
        r#"{"tool": "finish", "args": {"summary": "mission complete"}}"#,
    ];
    let model = Scripted::new(ProviderId::Ollama, &replies);
    register(&engine, &model);
    let spec = RunSpec {
        mode: RunMode::Orchestra,
        model: Some(local("lead")),
        agents: vec![agent("lead", None, "coordinate"), agent("coder", Some("lead"), "write files")],
    };
    let run = engine.create_run("make files", Some("test -f bye.txt"), spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let who: Vec<&str> = steps.iter().map(|s| s.meta["agent"].as_str().unwrap_or("-")).collect();
    assert_eq!(who, ["coder", "coder", "lead", "coder", "coder", "lead", "lead"]);
    assert_eq!(steps.last().unwrap().kind, StepKind::Verify);

    let seen = model.seen.lock().unwrap();
    assert!(
        seen[2]
            .messages
            .iter()
            .any(|m| m.content.contains("Report from CODER") && m.content.contains("hello.txt written"))
    );
    assert!(seen[2].system.contains("delegate"), "commanders get the delegate tool");
    assert!(!seen[0].system.contains("- delegate"), "leaves do not");
    assert!(
        seen[3]
            .messages
            .iter()
            .any(|m| m.content.contains("Order from LEAD") && m.content.contains("also add bye.txt"))
    );
    assert!(Path::new(&run.worktree).join("bye.txt").exists());
}

#[tokio::test]
async fn orchestra_trees_are_validated_on_create() {
    let (_d, engine) = setup(|_| {}).await;
    let spec = RunSpec {
        mode: RunMode::Orchestra,
        model: Some(local("lead")),
        agents: vec![agent("a", None, "t"), agent("b", None, "t")],
    };
    assert!(matches!(engine.create_run("x", None, spec).await, Err(Error::InvalidRequest(_))));
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
fn fake_acp(dir: &Path, id: &str) -> crate::acp::ExternalAgent {
    let script = dir.join("fake_acp.py");
    std::fs::write(&script, FAKE_ACP).unwrap();
    crate::acp::ExternalAgent {
        id: id.into(),
        name: format!("Fake {id}"),
        command: "python3".into(),
        args: vec![script.to_string_lossy().into(), id.into()],
    }
}

#[cfg(unix)]
#[tokio::test]
async fn solo_chat_with_an_external_acp_agent() {
    let scripts = tempfile::tempdir().unwrap();
    let agent = fake_acp(scripts.path(), "ext");
    let (_d, engine) = setup(|c| c.external_agents = vec![agent]).await;
    let model = ModelRef::new(ProviderId::Acp, "ext");
    let spec = RunSpec { mode: RunMode::Solo, model: Some(model), agents: vec![] };
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
async fn orchestra_mixes_internal_commanders_and_external_agents() {
    let scripts = tempfile::tempdir().unwrap();
    let ext = fake_acp(scripts.path(), "coder");
    let (_d, engine) = setup(|c| c.external_agents = vec![ext]).await;
    let lead = Scripted::new(ProviderId::Ollama, &[r#"{"tool": "finish", "args": {"summary": "team done"}}"#]);
    register(&engine, &lead);
    let mut coder = agent("coder", Some("lead"), "write coder.txt");
    coder.model = Some(ModelRef::new(ProviderId::Acp, "coder"));
    let spec = RunSpec {
        mode: RunMode::Orchestra,
        model: Some(local("lead")),
        agents: vec![agent("lead", None, "coordinate"), coder],
    };
    let run = engine.create_run("make files", None, spec).await.unwrap();
    let run = wait_idle(&engine, &run.id).await;
    let (_, steps) = engine.run_detail(&run.id).await.unwrap();
    assert_eq!(run.status, RunStatus::Done, "error: {:?}", run.error);
    let who: Vec<&str> = steps.iter().map(|s| s.meta["agent"].as_str().unwrap_or("-")).collect();
    assert_eq!(who, ["coder", "lead"]);
    let seen = lead.seen.lock().unwrap();
    assert!(seen[0].messages.iter().any(|m| m.content.contains("coder report: file written")));
    assert!(Path::new(&run.worktree).join("coder.txt").exists());
}
