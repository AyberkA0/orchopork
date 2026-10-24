use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillType {
    /// Adds `prompt_injection` to every role's system prompt.
    SystemModifier,
    /// Gives the actor extra tools, each backed by a shell command.
    ToolDefinition,
    /// Checks the actor's replies and/or gates `finish` on a command.
    Validator,
}

/// A tool the agent can call, implemented by a shell command you write.
///
/// Arguments are *never* interpolated into the command string (that would
/// be a shell-injection hole). Each argument is passed as an environment
/// variable `ORCHOPORK_ARG_<NAME>` (uppercased; strings raw, everything
/// else as JSON), e.g. `command: cargo test "$ORCHOPORK_ARG_FILTER"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema (or any short description object) for the arguments,
    /// shown to the model verbatim.
    #[serde(default)]
    pub parameters: serde_json::Value,
    /// Shell command run in the run's worktree.
    pub command: String,
}

/// Declarative checks. Substring rules apply to every actor reply
/// (case-insensitive); `command` must exit 0 before a `finish` is accepted;
/// `evidence` turns on the engine's checks of what reports claim.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ValidatorSpec {
    #[serde(default)]
    pub forbidden_substrings: Vec<String>,
    #[serde(default)]
    pub required_substrings: Vec<String>,
    #[serde(default)]
    pub command: Option<String>,
    /// Engine-enforced honesty: claims of having seen something are
    /// rejected unless the agent was shown an image, and reports copied
    /// word for word from an earlier one are sent back.
    #[serde(default)]
    pub evidence: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub version: String,
    #[serde(rename = "type")]
    pub kind: SkillType,
    pub description: String,
    #[serde(default)]
    pub prompt_injection: String,
    /// Lower sorts earlier in the composed prompt.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Skills that must not be enabled together with this one.
    #[serde(default)]
    pub conflicts_with: Vec<String>,
    /// Where the skill applies. Empty: everywhere (single agents and every
    /// orchestra node). `["*"]`: every orchestra node. Otherwise the role
    /// keys of the nodes it is for (`lead`, `coder`, `tester`, ...; see
    /// `FlowNode::role_key`).
    #[serde(default)]
    pub roles: Vec<String>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub validator: Option<ValidatorSpec>,
}

fn default_priority() -> i32 {
    100
}

/// Names the engine's own tools use. A skill tool with one of them would
/// never run (the built-in wins), would be approved as the built-in's class,
/// and would send two tool definitions with the same name to the model,
/// which providers reject.
pub const RESERVED_TOOLS: &[&str] = &[
    // The toolbox (graph::tools::BUILTIN).
    "list_files",
    "list_dir",
    "read_file",
    "search",
    "write_file",
    "edit_file",
    "move_file",
    "delete_file",
    "run_command",
    "start_process",
    "process_output",
    "stop_process",
    "screenshot_page",
    "screenshot_screen",
    "screenshot_sequence",
    "view_image",
    "web_search",
    "web_fetch",
    "browser",
    // Engine control and orchestra tools.
    "finish",
    "delegate",
    "handback",
    "review",
    "consult",
    "ask_suggestions",
    "request_test",
    "notebook_write",
    "notebook_read",
    "rerun",
    "spawn_agent",
    "retire_agent",
    "answer",
    "goal_verdict",
    "describe",
];

fn valid_ident(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

/// Which agent a prompt is composed for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Audience<'a> {
    /// A single-agent chat or any non-flow call.
    Single,
    /// An orchestra node with this role key (possibly empty).
    Node(&'a str),
}

impl Skill {
    /// Whether this skill is part of the prompt for `who`.
    pub fn applies_to(&self, who: Audience) -> bool {
        if self.roles.is_empty() {
            return true;
        }
        match who {
            Audience::Single => false,
            Audience::Node(key) => self.roles.iter().any(|r| r == "*" || (!key.is_empty() && r.eq_ignore_ascii_case(key))),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !valid_ident(&self.name) {
            return Err(Error::Skill(format!("invalid skill name {:?} (use [a-z0-9_-], max 64)", self.name)));
        }
        if let Some(bad) = self.roles.iter().find(|r| *r != "*" && !valid_ident(&r.to_lowercase())) {
            return Err(Error::Skill(format!("{}: invalid role {bad:?} (use [a-z0-9_-] or \"*\")", self.name)));
        }
        if self.version.trim().is_empty() {
            return Err(Error::Skill(format!("{}: empty version", self.name)));
        }
        match self.kind {
            SkillType::SystemModifier if self.prompt_injection.trim().is_empty() => {
                Err(Error::Skill(format!("{}: system_modifier needs prompt_injection", self.name)))
            }
            SkillType::ToolDefinition => {
                if self.tools.is_empty() {
                    return Err(Error::Skill(format!("{}: tool_definition needs tools", self.name)));
                }
                for t in &self.tools {
                    if !valid_ident(&t.name) || RESERVED_TOOLS.contains(&t.name.as_str()) {
                        return Err(Error::Skill(format!(
                            "{}: tool name {:?} is invalid or reserved by a built-in",
                            self.name, t.name
                        )));
                    }
                    if t.command.trim().is_empty() {
                        return Err(Error::Skill(format!("{}: tool {} needs a command", self.name, t.name)));
                    }
                }
                Ok(())
            }
            SkillType::Validator => match &self.validator {
                Some(v)
                    if !v.forbidden_substrings.is_empty()
                        || !v.required_substrings.is_empty()
                        || v.evidence
                        || v.command.as_deref().is_some_and(|c| !c.trim().is_empty()) =>
                {
                    Ok(())
                }
                _ => Err(Error::Skill(format!(
                    "{}: validator needs a validator block with substrings, evidence or a command",
                    self.name
                ))),
            },
            _ => Ok(()),
        }
    }
}
