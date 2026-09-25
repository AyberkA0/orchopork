use std::collections::HashSet;

use super::model::{Skill, SkillType, ToolSpec, ValidatorSpec};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Default)]
pub struct ComposedPrompt {
    pub system: String,
    pub tools: Vec<ToolSpec>,
    /// (skill name, spec)
    pub validators: Vec<(String, ValidatorSpec)>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Violation {
    pub skill: String,
    pub detail: String,
}

/// Deterministic composition: base prompt, then each modifier in
/// (priority, name) order inside a delimited block so later layers can be
/// attributed/debugged. Fails on declared conflicts and tool-name collisions.
pub fn compose(base: &str, enabled: &[Skill]) -> Result<ComposedPrompt> {
    let mut skills: Vec<&Skill> = enabled.iter().collect();
    skills.sort_by(|a, b| (a.priority, &a.name).cmp(&(b.priority, &b.name)));

    let names: HashSet<&str> = skills.iter().map(|s| s.name.as_str()).collect();
    for s in &skills {
        if let Some(c) = s.conflicts_with.iter().find(|c| names.contains(c.as_str())) {
            return Err(Error::Skill(format!("{} conflicts with {c}", s.name)));
        }
    }

    let mut out = ComposedPrompt { system: base.trim_end().to_string(), ..Default::default() };
    let mut tool_names = HashSet::new();
    for s in skills {
        match s.kind {
            SkillType::SystemModifier => {
                if !out.system.is_empty() {
                    out.system.push_str("\n\n");
                }
                out.system.push_str(&format!(
                    "<skill name=\"{}\" version=\"{}\">\n{}\n</skill>",
                    s.name, s.version, s.prompt_injection
                ));
            }
            SkillType::ToolDefinition => {
                for t in &s.tools {
                    if !tool_names.insert(t.name.clone()) {
                        return Err(Error::Skill(format!("duplicate tool {:?} (in {})", t.name, s.name)));
                    }
                    out.tools.push(t.clone());
                }
            }
            SkillType::Validator => {
                if let Some(v) = &s.validator {
                    out.validators.push((s.name.clone(), v.clone()));
                }
            }
        }
    }
    Ok(out)
}

impl ComposedPrompt {
    pub fn check_output(&self, text: &str) -> Vec<Violation> {
        let hay = text.to_lowercase();
        let mut v = Vec::new();
        for (skill, spec) in &self.validators {
            for f in &spec.forbidden_substrings {
                if hay.contains(&f.to_lowercase()) {
                    v.push(Violation { skill: skill.clone(), detail: format!("forbidden: {f:?}") });
                }
            }
            for r in &spec.required_substrings {
                if !hay.contains(&r.to_lowercase()) {
                    v.push(Violation { skill: skill.clone(), detail: format!("missing: {r:?}") });
                }
            }
        }
        v
    }
}
