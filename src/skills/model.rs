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
            _ => Ok(()),
        }
    }
}
