use std::collections::HashSet;

use super::model::{Skill, SkillType, TemplateAgent, ToolSpec, ValidatorSpec};
use crate::error::{Error, Result};

/// Who a composition is for, matched against each skill's `applies_to`.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    tags: Vec<&'static str>,
    /// Lowercased id, name and role of an orchestra agent (for `role:`).
    role: String,
}

impl Scope {
    pub fn of(tags: &[&'static str]) -> Self {
        Self { tags: tags.to_vec(), role: String::new() }
    }

    /// An orchestra agent: `lead` or `worker`, plus `commander` when it has
    /// subordinates.
    pub fn agent(lead: bool, commander: bool, id: &str, name: &str, role: &str) -> Self {
        let mut tags = vec![if lead { "lead" } else { "worker" }];
        if commander {
            tags.push("commander");
        }
        Self { tags, role: format!("{id} {name} {role}").to_lowercase() }
    }

    pub fn matches(&self, applies_to: &[String]) -> bool {
        let hit = |e: &str| match e.strip_prefix("role:") {
            Some(r) => self.role.contains(&r.trim().to_lowercase()),
            None => self.tags.contains(&e),
        };
        let (deny, allow): (Vec<&String>, Vec<&String>) = applies_to.iter().partition(|e| e.starts_with('!'));
        !deny.iter().any(|e| hit(&e[1..])) && (allow.is_empty() || allow.iter().any(|e| hit(e)))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ComposedPrompt {
    pub system: String,
    pub tools: Vec<ToolSpec>,
    /// (skill name, spec, explicitly scoped)
    pub validators: Vec<(String, ValidatorSpec, bool)>,
    /// (skill name, description, team)
    pub teams: Vec<(String, String, Vec<TemplateAgent>)>,
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
                    out.validators.push((s.name.clone(), v.clone(), !s.applies_to.is_empty()));
                }
            }
            SkillType::TeamTemplate => out.teams.push((s.name.clone(), s.description.clone(), s.team.clone())),
        }
    }
    Ok(out)
}

/// `compose` for one scope: conflicts are checked across every enabled
/// skill (so the answer never depends on who is asking), then only the
/// skills that apply to `scope` are layered in.
pub fn compose_for(base: &str, enabled: &[Skill], scope: &Scope) -> Result<ComposedPrompt> {
    compose("", enabled)?;
    let mine: Vec<Skill> = enabled.iter().filter(|s| scope.matches(&s.applies_to)).cloned().collect();
    compose(base, &mine)
}

impl ComposedPrompt {
    pub fn check_output(&self, text: &str) -> Vec<Violation> {
        let hay = text.to_lowercase();
        let mut v = Vec::new();
        for (skill, spec, _) in &self.validators {
            for f in &spec.forbidden_substrings {
                if hay.contains(&f.to_lowercase()) {
                    v.push(Violation { skill: skill.clone(), detail: format!("contains forbidden text {f:?}") });
                }
            }
            for r in &spec.required_substrings {
                if !hay.contains(&r.to_lowercase()) {
                    v.push(Violation { skill: skill.clone(), detail: format!("is missing required text {r:?}") });
                }
            }
        }
        v
    }

    /// (skill name, command) for every validator that gates the final
    /// `finish` (of a solo run, or of an orchestra's lead).
    pub fn verify_commands(&self) -> Vec<(String, String)> {
        self.commands(false)
    }

    /// (skill name, command) for the validators that gate an orchestra
    /// subordinate's report. Only explicitly scoped validators do: an
    /// unscoped one (e.g. a full test suite) would otherwise run at every
    /// report instead of once at the end.
    pub fn report_gates(&self) -> Vec<(String, String)> {
        self.commands(true)
    }

    fn commands(&self, scoped_only: bool) -> Vec<(String, String)> {
        self.validators
            .iter()
            .filter(|(_, _, scoped)| *scoped || !scoped_only)
            .filter_map(|(n, v, _)| v.command.as_ref().filter(|c| !c.trim().is_empty()).map(|c| (n.clone(), c.clone())))
            .collect()
    }

    /// The team templates as the designer reads them.
    pub fn teams_note(&self) -> String {
        if self.teams.is_empty() {
            return String::new();
        }
        let mut s = String::from(
            "\n\n# Team templates\nStarting points for common missions. When one fits, adapt it (rename, drop or add agents, make every task concrete for this mission and repository); otherwise design freely.\n",
        );
        for (name, desc, team) in &self.teams {
            s.push_str(&format!("\n## {name}: {desc}\n"));
            for a in team {
                let parent = a.parent.as_deref().unwrap_or("null");
                s.push_str(&format!("- {} (parent {parent}): {} — {}", a.id, a.role, a.task.trim()));
                if !a.done_when.trim().is_empty() {
                    s.push_str(&format!(" Done when: {}", a.done_when.trim()));
                }
                s.push('\n');
            }
        }
        s
    }
}
