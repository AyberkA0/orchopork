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
    /// A starting-point team the orchestra designer adapts when the mission
    /// fits its `description`. Costs context only in the one design call.
    TeamTemplate,
}

/// Who a skill is for. Every entry is a scope tag (`solo`, `planner`,
/// `critic`, `designer`, `lead`, `commander`, `worker`) or `role:<text>`,
/// which matches an orchestra agent whose id, name or role contains `<text>`
/// (case-insensitive). A leading `!` excludes instead. An empty list means
/// everyone.
pub const SCOPE_TAGS: &[&str] = &["solo", "planner", "critic", "designer", "lead", "commander", "worker"];

/// One agent of a `team_template`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TemplateAgent {
    pub id: String,
    pub name: String,
    pub role: String,
    pub task: String,
    pub parent: Option<String>,
    pub owns: Vec<String>,
    pub done_when: String,
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
/// (case-insensitive); `command` must exit 0 before a `finish` is accepted.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ValidatorSpec {
    #[serde(default)]
    pub forbidden_substrings: Vec<String>,
    #[serde(default)]
    pub required_substrings: Vec<String>,
    #[serde(default)]
    pub command: Option<String>,
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
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub validator: Option<ValidatorSpec>,
    /// Scopes this skill applies to (see `SCOPE_TAGS`); empty = all.
    #[serde(default)]
    pub applies_to: Vec<String>,
    /// `team_template` only.
    #[serde(default)]
    pub team: Vec<TemplateAgent>,
}

fn default_priority() -> i32 {
    100
}

/// Names the built-in toolbox reserves.
pub const RESERVED_TOOLS: &[&str] =
    &["list_files", "read_file", "search", "write_file", "edit_file", "run_command", "finish"];

fn valid_ident(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}

impl Skill {
    pub fn validate(&self) -> Result<()> {
        if !valid_ident(&self.name) {
            return Err(Error::Skill(format!("invalid skill name {:?} (use [a-z0-9_-], max 64)", self.name)));
        }
        if self.version.trim().is_empty() {
            return Err(Error::Skill(format!("{}: empty version", self.name)));
        }
        for a in &self.applies_to {
            let a = a.strip_prefix('!').unwrap_or(a);
            let ok = match a.strip_prefix("role:") {
                Some(r) => !r.trim().is_empty(),
                None => SCOPE_TAGS.contains(&a),
            };
            if !ok {
                return Err(Error::Skill(format!(
                    "{}: unknown applies_to entry {a:?} (use {} or role:<text>)",
                    self.name,
                    SCOPE_TAGS.join(", ")
                )));
            }
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
                        || v.command.as_deref().is_some_and(|c| !c.trim().is_empty()) =>
                {
                    Ok(())
                }
                _ => Err(Error::Skill(format!(
                    "{}: validator needs a validator block with substrings or a command",
                    self.name
                ))),
            },
            SkillType::TeamTemplate => {
                if self.team.iter().filter(|a| a.parent.is_none()).count() != 1
                    || self.team.iter().any(|a| a.id.trim().is_empty() || a.task.trim().is_empty())
                {
                    return Err(Error::Skill(format!(
                        "{}: team_template needs a team with exactly one lead (no parent) and an id and task for every agent",
                        self.name
                    )));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}
