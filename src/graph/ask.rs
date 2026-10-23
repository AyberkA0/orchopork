//! Questions from agents.
//!
//! An internal agent asks with the `ask` tool; an external one (Claude
//! Code, Codex…) ends its reply with a `QUESTION:` line. Either way the
//! question becomes an `act` step with `tool: "ask"`, and then:
//!
//! - routed to the user, the run stops in `waiting` until an answer is
//!   injected (`Engine::answer`), or a message for that agent arrives;
//! - routed to the commander (orchestras, `RunSpec::questions`), the asking
//!   agent's commander answers right away with one call of its own model,
//!   and the agent continues. A commander that cannot decide passes the
//!   question on to the user.
//!
//! The answer is an `inject` step addressed to the asking agent, with
//! `from` naming who answered and `answer_to` the question's step.

use std::time::Instant;

use serde_json::{Value, json};

use super::{Inner, call_meta, orchestra, prompts};
use crate::config::Config;
use crate::error::Result;
use crate::providers::{CompletionRequest, Message, ProviderId};
use crate::storage::{AgentSpec, Run, Step, StepKind};

/// What a commander replies when only the user can answer.
pub const ASK_USER: &str = "ASK_USER";

/// The line external agents are told to end with when they need a decision.
pub const EXTERNAL_HOW: &str = "If you cannot continue without a decision only a person can make (missing information, a \
product choice), stop and end your reply with one line `QUESTION: <your question>`, optionally followed by lines \
`- <option>` with the choices you see. Prefer a sensible default over asking.";

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub text: String,
    pub options: Vec<String>,
}

impl Question {
    pub fn meta(&self) -> Value {
        json!({ "question": self.text, "options": self.options })
    }
}

/// The `QUESTION:` an external agent ended its reply with, if any.
pub fn from_text(text: &str) -> Option<Question> {
    let lines: Vec<&str> = text.lines().collect();
    let at = lines.iter().rposition(|l| {
        let l = l.trim().trim_start_matches(['*', '#', '>', ' ']);
        l.len() > 9 && l[..9].eq_ignore_ascii_case("question:")
    })?;
    let head = lines[at].trim().trim_start_matches(['*', '#', '>', ' ']);
    let mut q = head[9..].trim().trim_matches('*').trim().to_string();
    let mut options = Vec::new();
    for l in &lines[at + 1..] {
        let l = l.trim();
        if let Some(o) = l.strip_prefix("- ").or_else(|| l.strip_prefix("* ")) {
            options.push(o.trim().to_string());
        } else if !l.is_empty() && options.is_empty() {
            // The question went on over a few lines.
            q.push(' ');
            q.push_str(l);
        } else if !l.is_empty() {
            break;
        }
    }
    (!q.is_empty()).then_some(Question { text: q, options })
}

/// The arguments of an `ask` action.
pub fn from_args(args: &serde_json::Map<String, Value>) -> Option<Question> {
    let text = args.get("question").or_else(|| args.get("text")).and_then(Value::as_str)?.trim().to_string();
    let options = match args.get("options") {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(|s| s.trim().to_string()).collect(),
        _ => vec![],
    };
    (!text.is_empty()).then_some(Question { text, options })
}

/// The question the run is waiting on: the latest `ask` routed to the user
/// with no message for its agent after it.
pub fn waiting_on(steps: &[Step]) -> Option<&Step> {
    let q = steps.iter().rev().find(|s| s.kind == StepKind::Act && s.meta["tool"] == "ask")?;
    if q.meta["route"] != "user" {
        return None;
    }
    let agent = q.meta["agent"].as_str();
    let answered = steps
        .iter()
        .any(|s| s.seq > q.seq && s.kind == StepKind::Inject && (agent.is_none() || s.meta["agent"].as_str() == agent));
    (!answered).then_some(q)
}

/// "Name asks: question" for the run's status line.
pub fn status_line(step: &Step, who: &str) -> String {
    format!("{who} asks: {}", step.meta["question"].as_str().unwrap_or(""))
}

impl Inner {
    /// Asks `cmd` (the asking agent's commander) to answer `q` from what it
    /// knows. `Ok(None)` when it passes the question on to the user.
    /// Returns the answer and the metadata of the call.
    pub(super) async fn commander_answer(
        &self,
        cfg: &Config,
        run: &Run,
        steps: &[Step],
        me: &AgentSpec,
        cmd: &AgentSpec,
        q: &Question,
    ) -> Result<Option<(String, Value, f64)>> {
        let t0 = Instant::now();
        let spec = &run.spec;
        let model = orchestra::model_for(cfg, spec, cmd)?;
        let files = crate::git::GitRepo::new(&run.worktree).ls_files_at(&run.base_commit).await?;
        let subs: Vec<&AgentSpec> = spec.children(&cmd.id).collect();
        let root = spec.root().map(|r| r.id.clone()).unwrap_or_default();
        let pin = orchestra::pinned(run, spec, cmd, &subs, &files);
        let groups = orchestra::transcript(spec, cmd, &root, steps, cfg.limits.tool_output_chars);
        let mut options = String::new();
        for o in &q.options {
            options.push_str(&format!("\n- {o}"));
        }
        let ask = format!(
            "# Question from your subordinate {} ({})\n{}{options}\n\nAnswer it yourself, directly and briefly, so they can \
             continue: decide from the mission, your brief and the reports so far. Do not change any files. Only if this \
             truly needs the user (a decision nobody in the team can make), reply with exactly {ASK_USER}.",
            me.name, me.id, q.text
        );
        let (text, mut meta, cost) = if model.provider == ProviderId::Acp {
            let mut msgs = prompts::windowed(pin, groups, cfg.limits.cloud_context_chars);
            msgs.push(Message::user(ask));
            let prompt = super::external::as_prompt(&msgs);
            let out = self.external_turn(cfg, run, &model, &cmd.name, prompt).await?;
            (out.text.clone(), super::external::meta(&model, &out), 0.0)
        } else {
            let mut msgs = prompts::windowed(pin, groups, cfg.context_chars(&model));
            msgs.push(Message::user(ask));
            let system = format!(
                "You are {}, a commander in a team of AI agents working on a software task. Answer your subordinate's question.",
                cmd.name
            );
            let req = CompletionRequest { system, messages: merge(msgs), ..Default::default() };
            let out = self.call_req(cfg, run, &model, req).await?;
            (out.text.clone(), call_meta(&out), out.cost_usd)
        };
        meta["ms"] = json!(t0.elapsed().as_millis() as u64);
        let text = text.trim().to_string();
        if text.is_empty() || text.contains(ASK_USER) {
            return Ok(None);
        }
        Ok(Some((text, meta, cost)))
    }
}

/// Joins consecutive user messages (the transcript may end with one).
fn merge(msgs: Vec<Message>) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::new();
    for m in msgs {
        match out.last_mut() {
            Some(last) if last.role == m.role => {
                last.content.push_str("\n\n");
                last.content.push_str(&m.content);
            }
            _ => out.push(m),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_question_lines_from_external_replies() {
        let t = "I looked at the parser.\n\n**QUESTION:** Should the new flag be on by default?\n- yes\n- no\n";
        let q = from_text(t).unwrap();
        assert_eq!(q.text, "Should the new flag be on by default?");
        assert_eq!(q.options, ["yes", "no"]);
        assert!(from_text("Done. Any questions? None.").is_none());
        let multi = from_text("QUESTION: Which database\nshould I use?").unwrap();
        assert_eq!(multi.text, "Which database should I use?");
    }

    #[test]
    fn ask_args_need_a_question() {
        let args = |v: Value| v.as_object().cloned().unwrap();
        assert_eq!(
            from_args(&args(json!({"question": " Port? ", "options": ["80", "8080"]}))),
            Some(Question { text: "Port?".into(), options: vec!["80".into(), "8080".into()] })
        );
        assert!(from_args(&args(json!({"options": []}))).is_none());
    }
}
