//! Turns executed by external ACP agents (Claude Code, Gemini CLI, Codex…).
//!
//! An external agent owns a whole task per turn: it edits the worktree with
//! its own tools and ends with a report. orchopork gives it the same
//! context an internal agent would see (flattened into one prompt), streams
//! its progress to the UI, then checkpoints the worktree like any step.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use serde_json::json;

use super::{Inner, Stop, prompts, verify_commands};
use crate::acp::{self, Outcome, Policy, Update};
use crate::config::{Config, ModelRef};
use crate::error::{Error, Result};
use crate::fsutil::clip;
use crate::git::GitRepo;
use crate::providers::{Message, Role};
use crate::storage::{Phase, Run, RunState, Step, StepKind};

const SUFFIX: &str = "\n\n---\nWork directly in the current directory (a git working tree dedicated to this task). \
Make the changes yourself and check them. When you are done, end with a concise report: what you changed, how you \
verified it, and anything left open.";

/// Renders a chat transcript as one prompt for an agent that takes a
/// single message.
pub fn as_prompt(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| match m.role {
            Role::User => m.content.clone(),
            Role::Assistant => format!("[Your earlier reply]\n{}", m.content),
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n")
}

pub fn tool_lines(out: &Outcome) -> String {
    out.tool_calls
        .iter()
        .map(|t| {
            let mark = match t.status.as_str() {
                "completed" => "✓",
                "failed" => "✗",
                _ => "·",
            };
            format!("{mark} {} ({})", t.title, t.kind)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn meta(model: &ModelRef, out: &Outcome) -> serde_json::Value {
    json!({
        "model": model.to_string(),
        "external": { "stop_reason": out.stop_reason, "tool_calls": out.tool_calls, "settings": out.settings },
        "ok": !out.cancelled,
        "tool": if out.cancelled { "external" } else { "finish" },
    })
}

impl Inner {
    /// Runs one external turn. `who` labels progress messages.
    pub(super) async fn external_turn(
        &self,
        cfg: &Config,
        run: &Run,
        model: &ModelRef,
        who: &str,
        prompt: String,
    ) -> Result<Outcome> {
        let agent = cfg.external_agents.iter().find(|a| a.id == model.model).cloned().ok_or_else(|| {
            Error::InvalidRequest(format!("external agent {:?} is not configured (Models & settings)", model.model))
        })?;
        let flag =
            self.active.lock().unwrap().get(&run.id).cloned().unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        self.log(&run.id, format!("{who}: starting {}", agent.name));
        let mut last = Instant::now() - Duration::from_secs(5);
        let run_id = run.id.clone();
        acp::run(
            &agent,
            Path::new(&run.worktree),
            &prompt,
            &model.options,
            Policy { allow_execute: cfg.limits.allow_commands },
            Duration::from_secs(cfg.limits.external_timeout_secs.max(30)),
            &flag,
            |u| {
                let msg = match u {
                    Update::Tool(t) => Some(format!("{who} · {}", t.title)),
                    Update::Plan(p) => Some(format!("{who} plans: {}", clip(&p, 160))),
                    Update::Thinking | Update::Text if last.elapsed() > Duration::from_secs(3) => Some(format!(
                        "{who} is {}…",
                        if matches!(u, Update::Thinking) { "thinking" } else { "writing" }
                    )),
                    _ => None,
                };
                if let Some(m) = msg {
                    last = Instant::now();
                    self.log(&run_id, m);
                }
            },
        )
        .await
    }

    /// Solo/classic act turn when the actor is an external agent.
    pub(super) async fn act_external(
        &self,
        cfg: &Config,
        run: &Run,
        steps: &[Step],
        state: RunState,
        model: &ModelRef,
    ) -> Result<Option<Stop>> {
        let files = GitRepo::new(&run.worktree).ls_files().await?;
        let plan = steps.iter().rev().find(|s| s.kind == StepKind::Plan).map(|s| s.output.as_str()).unwrap_or("");
        let messages = prompts::actor_messages(
            &run.goal,
            plan,
            &files,
            steps,
            cfg.limits.cloud_context_chars,
            cfg.limits.tool_output_chars,
        );
        let prompt = format!("{}{SUFFIX}", as_prompt(&messages));
        let name = self.agent_display(cfg, model);
        let out = self.external_turn(cfg, run, model, &name, prompt).await?;
        let mut next = RunState { failures: 0, escalate: false, ..state };
        if !out.cancelled {
            let skills = self.ws.skills.compose("")?;
            next.phase = if !verify_commands(run, &skills).is_empty() {
                Phase::Verify
            } else if cfg.routing.critic.is_some() {
                Phase::Review
            } else {
                Phase::Done
            };
        }
        let text =
            if out.text.is_empty() { format!("({} finished without a message)", name) } else { out.text.clone() };
        self.append(run, StepKind::Act, &text, Some(&tool_lines(&out)), meta(model, &out), &next, 0.0).await?;
        Ok(out.cancelled.then_some(Stop::Paused(None)))
    }

    pub(super) fn agent_display(&self, cfg: &Config, model: &ModelRef) -> String {
        cfg.external_agents.iter().find(|a| a.id == model.model).map(|a| a.name.clone()).unwrap_or(model.model.clone())
    }
}
