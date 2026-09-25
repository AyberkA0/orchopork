use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillType {
    SystemModifier,
    ToolDefinition,
    Validator,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool arguments.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

/// Declarative output check. Case-insensitive substring match.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ValidatorSpec {
    #[serde(default)]
    pub forbidden_substrings: Vec<String>,
    #[serde(default)]
    pub required_substrings: Vec<String>,
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

impl Skill {
    pub fn validate(&self) -> Result<()> {
        let ok_name = !self.name.is_empty()
            && self.name.len() <= 64
            && self.name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_');
        if !ok_name {
            return Err(Error::Skill(format!("invalid skill name {:?} (use [a-z0-9_-], max 64)", self.name)));
        }
        if self.version.trim().is_empty() {
            return Err(Error::Skill(format!("{}: empty version", self.name)));
        }
        match self.kind {
            SkillType::SystemModifier if self.prompt_injection.trim().is_empty() => {
                Err(Error::Skill(format!("{}: system_modifier needs prompt_injection", self.name)))
            }
            SkillType::ToolDefinition if self.tools.is_empty() => {
                Err(Error::Skill(format!("{}: tool_definition needs tools", self.name)))
            }
            SkillType::Validator if self.validator.is_none() => {
                Err(Error::Skill(format!("{}: validator needs a validator block", self.name)))
            }
            _ => Ok(()),
        }
    }
}
