//! Questions from agents to the user.
//!
//! An internal agent asks with the `ask` tool; an external one (Claude
//! Code, Codex…) ends its reply with a `QUESTION:` line. Either way the
//! question becomes an `act` step with `tool: "ask"`, and then:
//!
//! - routed to the user, the run stops in `waiting` until the user answers
//!   (`Engine::answer`) or messages that agent;
//! - routed to the Lead (the graph's `questions` setting), the Lead
//!   answers at once in a short read-only side session and the agent goes
//!   on. A Lead that cannot decide passes the question on to the user.
//!
//! Agents can also ask each other directly, over `consult` edges.
//!
//! A question is not taken from an agent inside a Dispatcher's parallel
//! branch: waiting there would hold up the whole run, so such an agent
//! decides itself or consults a teammate.
//!
//! The answer is an `inject` step addressed to the asking agent, with
//! `answer_to` naming the question's step.

use serde_json::{Value, json};

use crate::providers::ToolDef;
use crate::storage::{Step, StepKind};

/// What the Lead replies when only the user can answer.
pub const ASK_USER: &str = "ASK_USER";

/// The line external agents are told to end with when they need a decision.
pub const EXTERNAL_HOW: &str = "\n\nIf you cannot continue without a decision only a person can make (missing information, \
a product choice), stop and end your reply with one line `QUESTION: <your question>`, optionally followed by lines \
`- <option>` with the choices you see. Prefer a sensible default over asking.";

/// The `ask` line of the text tool reference.
pub const TOOL_LINE: &str = "- ask {\"question\": string, \"options\"?: [string]}: ask the user a question only a person \
can answer (missing information, a product decision). Work stops until the answer arrives, so prefer a sensible default \
and ask only when a wrong guess would waste the work.\n";

/// The `ask` tool as a native tool definition.
pub fn tool_def() -> ToolDef {
    ToolDef {
        name: "ask".into(),
        description: "Ask the user a question only a person can answer; work stops until the answer arrives.".into(),
        parameters: json!({ "type": "object", "properties": {
            "question": { "type": "string" },
            "options": { "type": "array", "items": { "type": "string" }, "description": "the choices you see, if any" }
        }, "required": ["question"] }),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub text: String,
    pub options: Vec<String>,
}

impl Question {
    /// Step metadata for the question.
    pub fn meta(&self) -> Value {
        json!({ "question": self.text, "options": self.options, "route": "user" })
    }

    /// The run's status line while it waits.
    pub fn status(&self, who: &str) -> String {
        format!("{who} asks: {}", self.text)
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
    if q.meta["route"].as_str().is_some_and(|r| r != "user") {
        return None;
    }
    let agent = q.meta["agent"].as_str();
    let answered = steps
        .iter()
        .any(|s| s.seq > q.seq && s.kind == StepKind::Inject && (agent.is_none() || s.meta["agent"].as_str() == agent));
    (!answered).then_some(q)
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
