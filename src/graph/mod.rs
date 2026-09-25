//! Execution graph: agent phases run one at a time, each step checkpointed
//! through `storage::Checkpointer` (SQLite row + git commit) so the
//! transcript, cost and worktree state are always in lockstep.
//!
//! There is no internal run loop: the caller (HTTP handler, CLI, test)
//! drives `step` one call at a time. That is what makes Pause a no-op (just
//! stop calling `step`), Inject a plain checkpoint with no LLM call, and
//! Rewind exactly `Checkpointer::rewind` — no separate state machine needed.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::providers::{CompletionRequest, CompletionResponse, Gateway, Message, ProviderId, Role};
use crate::skills::SkillRegistry;
use crate::storage::{Checkpointer, Snapshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Planner,
    Coder,
    Critic,
    TestRunner,
    ToolExecutor,
}

impl NodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeKind::Planner => "planner",
            NodeKind::Coder => "coder",
            NodeKind::Critic => "critic",
            NodeKind::TestRunner => "test_runner",
            NodeKind::ToolExecutor => "tool_executor",
        }
    }

    /// The instruction each phase starts from, before skills are layered on.
    fn base_prompt(self) -> &'static str {
        match self {
            NodeKind::Planner => "You are the planning phase. Break the task into concrete, ordered steps.",
            NodeKind::Coder => "You are the coding phase. Produce the smallest diff that implements the current step.",
            NodeKind::Critic => {
                "You are the critic phase. Find defects in the previous step's output; do not restate it."
            }
            NodeKind::TestRunner => "You are the test-running phase. Report which checks pass or fail, verbatim.",
            NodeKind::ToolExecutor => {
                "You are the tool-execution phase. Call the requested tool and report its result."
            }
        }
    }
}

/// One node's LLM call plus the checkpoint it produced.
#[derive(Debug, Clone, Serialize)]
pub struct StepOutcome {
    pub snapshot: Snapshot,
    pub response: CompletionResponse,
}

/// Ties the provider gateway, skill composer and checkpointer together for
/// one workspace. Holds no per-thread state itself: the transcript is
/// rebuilt from checkpoint history on every call, so an `Executor` is cheap
/// to construct and safe to share.
///
/// Gateway and SkillRegistry are `Arc`-shared rather than owned: the same
/// instances back the standalone `/api/skills` and `/api/providers/*`
/// endpoints, so a key set or a skill toggled through those routes is
/// visible to the very next `step` call, from any of the three call sites
/// (in-process embedding, HTTP API, CLI) alike.
#[derive(Clone)]
pub struct Executor {
    gateway: Arc<Gateway>,
    skills: Arc<SkillRegistry>,
    checkpointer: Checkpointer,
}

impl Executor {
    pub fn new(gateway: Arc<Gateway>, skills: Arc<SkillRegistry>, checkpointer: Checkpointer) -> Self {
        Self { gateway, skills, checkpointer }
    }

    pub async fn history(&self, thread_id: &str) -> Result<Vec<Snapshot>> {
        self.checkpointer.history(thread_id).await
    }

    pub async fn rewind(&self, thread_id: &str, target: crate::storage::RewindTarget) -> Result<Snapshot> {
        self.checkpointer.rewind(thread_id, target).await
    }

    /// Replays a thread's checkpoint history into the message list a
    /// provider expects. This is the only source of truth for "what has
    /// this thread said so far" — there is no separate in-memory transcript
    /// to fall out of sync with the git/SQLite record.
    pub async fn transcript(&self, thread_id: &str) -> Result<Vec<Message>> {
        self.checkpointer
            .history(thread_id)
            .await?
            .into_iter()
            .map(|s| serde_json::from_str(&s.context_delta).map_err(Error::from))
            .collect()
    }

    /// Records a user message with no LLM call and no cost. This is the
    /// dashboard's "Inject": the next `step` will see it in the transcript.
    pub async fn inject(&self, thread_id: &str, content: &str) -> Result<Snapshot> {
        let delta = Message { role: Role::User, content: content.to_string() };
        self.checkpointer.commit(thread_id, "user_injection", &serde_json::to_value(&delta)?, 0.0).await
    }

    /// Runs exactly one node: composes the skill-augmented system prompt,
    /// calls the provider, checks the result against any enabled validator
    /// skills, then checkpoints it. A validator failure is not
    /// checkpointed — the caller sees the violation and may retry, switch
    /// providers, or surface it to the user, but nothing bad is recorded.
    pub async fn step(
        &self,
        thread_id: &str,
        node: NodeKind,
        provider: ProviderId,
        model: &str,
    ) -> Result<StepOutcome> {
        let messages = self.transcript(thread_id).await?;
        let composed = self.skills.compose(node.base_prompt())?;
        let req = CompletionRequest { model: model.to_string(), system: composed.system.clone(), messages };
        let response = self.gateway.complete(provider, &req).await?;

        let violations = composed.check_output(&response.text);
        if let Some(v) = violations.first() {
            return Err(Error::Skill(format!("{} validator: {}", v.skill, v.detail)));
        }

        let delta = Message { role: Role::Assistant, content: response.text.clone() };
        let snapshot = self
            .checkpointer
            .commit(thread_id, node.as_str(), &serde_json::to_value(&delta)?, response.cost_usd)
            .await?;
        Ok(StepOutcome { snapshot, response })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::GitRepo;
    use crate::storage::Store;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Stub {
        calls: AtomicUsize,
        reply: &'static str,
    }

    #[async_trait::async_trait]
    impl crate::providers::Provider for Stub {
        fn id(&self) -> ProviderId {
            ProviderId::Ollama
        }
        async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResponse {
                text: format!("{} (saw {} prior messages)", self.reply, req.messages.len()),
                prompt_tokens: 1,
                completion_tokens: 1,
                cost_usd: 0.0,
            })
        }
    }

    async fn executor(dir: &std::path::Path) -> Executor {
        let repo = GitRepo::new(dir);
        repo.init().await.unwrap();
        let store = Store::open(&dir.join(".orchopork/state.db")).await.unwrap();
        let checkpointer = Checkpointer::new(store.clone(), repo);
        let gateway = Gateway::new(store, 40.0);
        gateway.register(Box::new(Stub { calls: AtomicUsize::new(0), reply: "plan ready" }));
        let skills =
            SkillRegistry::open(dir.join(".orchopork/skills"), dir.join(".orchopork/skills.enabled.yaml")).unwrap();
        Executor::new(Arc::new(gateway), Arc::new(skills), checkpointer)
    }

    #[tokio::test]
    async fn step_checkpoints_the_response_and_extends_the_transcript() {
        let ws = tempfile::tempdir().unwrap();
        let ex = executor(ws.path()).await;

        let out = ex.step("t1", NodeKind::Planner, ProviderId::Ollama, "m").await.unwrap();
        assert_eq!(out.snapshot.node, "planner");
        assert!(out.response.text.starts_with("plan ready"));

        let transcript = ex.transcript("t1").await.unwrap();
        assert_eq!(transcript.len(), 1);
        assert_eq!(transcript[0].role, Role::Assistant);
    }

    #[tokio::test]
    async fn inject_extends_transcript_without_calling_the_provider_and_step_sees_it() {
        let ws = tempfile::tempdir().unwrap();
        let ex = executor(ws.path()).await;

        ex.inject("t1", "focus on the auth module").await.unwrap();
        let out = ex.step("t1", NodeKind::Coder, ProviderId::Ollama, "m").await.unwrap();
        // The injected message is the one prior message the stub should see.
        assert!(out.response.text.contains("saw 1 prior messages"));

        let transcript = ex.transcript("t1").await.unwrap();
        assert_eq!(transcript.len(), 2);
        assert_eq!(
            (&transcript[0].role, &transcript[0].content),
            (&Role::User, &"focus on the auth module".to_string())
        );
    }

    #[tokio::test]
    async fn validator_violation_blocks_the_checkpoint() {
        let ws = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(ws.path().join(".orchopork/skills")).unwrap();
        std::fs::write(
            ws.path().join(".orchopork/skills/no-todo.yaml"),
            "name: no-todo\nversion: \"1.0.0\"\ntype: validator\ndescription: reject TODOs\nvalidator:\n  forbidden_substrings: [\"plan ready\"]\n",
        )
        .unwrap();
        let ex = executor(ws.path()).await;
        ex.skills.set_enabled("no-todo", true).unwrap();

        let err = ex.step("t1", NodeKind::Planner, ProviderId::Ollama, "m").await.unwrap_err();
        assert!(matches!(err, Error::Skill(_)));
        assert!(ex.transcript("t1").await.unwrap().is_empty());
    }
}
