//! Skill engine: declarative tone/persona/tool modules loaded from
//! `.orchopork/skills/`, toggled and hot-reloaded at runtime.

pub mod compose;
pub mod loader;
pub mod model;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::RwLock;

use serde::Serialize;

pub use compose::{ComposedPrompt, Violation, compose};
pub use model::{Audience, Skill, SkillType};

use crate::error::{Error, Result};

#[derive(Default)]
struct Inner {
    skills: BTreeMap<String, Skill>,
    enabled: BTreeSet<String>,
    errors: Vec<(PathBuf, String)>,
}

#[derive(Debug, Serialize)]
pub struct SkillInfo {
    #[serde(flatten)]
    pub skill: Skill,
    pub enabled: bool,
}

pub struct SkillRegistry {
    dir: PathBuf,
    /// Persisted list of enabled skill names (kept outside `dir` so it is not
    /// mistaken for a skill file).
    state_file: PathBuf,
    inner: RwLock<Inner>,
}

impl SkillRegistry {
    /// Installs bundled skills if absent, loads the directory, and restores
    /// the enabled set (`loader::DEFAULT_ENABLED` on first run).
    pub fn open(dir: impl Into<PathBuf>, state_file: impl Into<PathBuf>) -> Result<Self> {
        let reg = Self { dir: dir.into(), state_file: state_file.into(), inner: RwLock::default() };
        let added = loader::install_bundled(&reg.dir)?;
        let first_run = !reg.state_file.exists();
        reg.reload()?;
        if first_run {
            let mut g = reg.inner.write().unwrap();
            g.enabled =
                loader::DEFAULT_ENABLED.iter().filter(|n| g.skills.contains_key(**n)).map(|n| n.to_string()).collect();
            drop(g);
            reg.persist()?;
        } else {
            // An update that ships a new default skill turns it on once; the
            // user can switch it off, and it stays off (the file is there).
            let mut g = reg.inner.write().unwrap();
            let new: Vec<String> =
                added.into_iter().filter(|n| loader::DEFAULT_ENABLED.contains(&n.as_str()) && g.skills.contains_key(n)).collect();
            if !new.is_empty() {
                g.enabled.extend(new);
                drop(g);
                reg.persist()?;
            }
        }
        Ok(reg)
    }

    /// Re-read disk. Builds the new state fully before swapping, so readers
    /// never observe a half-loaded registry.
    pub fn reload(&self) -> Result<Vec<(PathBuf, String)>> {
        let report = loader::load_dir(&self.dir)?;
        let persisted: BTreeSet<String> = match std::fs::read_to_string(&self.state_file) {
            Ok(t) => serde_yaml::from_str(&t)?,
            Err(_) => self.inner.read().unwrap().enabled.clone(),
        };
        let enabled = persisted.into_iter().filter(|n| report.skills.contains_key(n)).collect();
        let errors = report.errors.clone();
        *self.inner.write().unwrap() = Inner { skills: report.skills, enabled, errors: report.errors };
        Ok(errors)
    }

    pub fn list(&self) -> Vec<SkillInfo> {
        let g = self.inner.read().unwrap();
        g.skills.values().map(|s| SkillInfo { skill: s.clone(), enabled: g.enabled.contains(&s.name) }).collect()
    }

    pub fn load_errors(&self) -> Vec<(PathBuf, String)> {
        self.inner.read().unwrap().errors.clone()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.inner.read().unwrap().skills.contains_key(name)
    }

    pub fn dir(&self) -> &std::path::Path {
        &self.dir
    }

    pub fn set_enabled(&self, name: &str, on: bool) -> Result<()> {
        self.replace_enabled(|cur| {
            let mut next = cur.clone();
            if on {
                next.insert(name.into());
            } else {
                next.remove(name);
            }
            next
        })
    }

    pub fn set_enabled_exact(&self, names: &[String]) -> Result<()> {
        self.replace_enabled(|_| names.iter().cloned().collect())
    }

    /// Validates the whole new set (unknown names, declared conflicts,
    /// duplicate tools) *before* committing it, so an invalid combination
    /// is rejected at toggle time instead of failing the next run step.
    /// The new set is computed under the lock, so two toggles at once both
    /// take effect.
    fn replace_enabled(&self, change: impl FnOnce(&BTreeSet<String>) -> BTreeSet<String>) -> Result<()> {
        {
            let mut g = self.inner.write().unwrap();
            let next = change(&g.enabled);
            if let Some(bad) = next.iter().find(|n| !g.skills.contains_key(*n)) {
                return Err(Error::NotFound(format!("skill {bad}")));
            }
            let chosen: Vec<Skill> = next.iter().filter_map(|n| g.skills.get(n).cloned()).collect();
            compose("", &chosen)?;
            g.enabled = next;
        }
        self.persist()
    }

    pub fn enabled_skills(&self) -> Vec<Skill> {
        let g = self.inner.read().unwrap();
        g.enabled.iter().filter_map(|n| g.skills.get(n).cloned()).collect()
    }

    /// The prompt for single agents and non-flow calls: skills with no role.
    pub fn compose(&self, base: &str) -> Result<ComposedPrompt> {
        self.compose_for(base, Audience::Single)
    }

    /// The prompt for `who`: the skills that apply everywhere plus the ones
    /// written for that role.
    pub fn compose_for(&self, base: &str, who: Audience) -> Result<ComposedPrompt> {
        let chosen: Vec<Skill> = self.enabled_skills().into_iter().filter(|s| s.applies_to(who)).collect();
        compose(base, &chosen)
    }

    /// Only the skills written for `role` (no everywhere-skills): the
    /// playbook of an agent that is not run through the normal agent loop.
    pub fn compose_role(&self, role: &str) -> Result<ComposedPrompt> {
        let chosen: Vec<Skill> = self
            .enabled_skills()
            .into_iter()
            .filter(|s| !s.roles.is_empty() && s.applies_to(Audience::Node(role)))
            .collect();
        compose("", &chosen)
    }

    fn persist(&self) -> Result<()> {
        let enabled = self.inner.read().unwrap().enabled.clone();
        crate::fsutil::write_atomic(&self.state_file, serde_yaml::to_string(&enabled)?.as_bytes(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> (tempfile::TempDir, SkillRegistry) {
        let t = tempfile::tempdir().unwrap();
        let r = SkillRegistry::open(t.path().join("skills"), t.path().join("enabled.yaml")).unwrap();
        (t, r)
    }

    #[test]
    fn bundled_skills_load_and_compose_in_priority_order() {
        let (_t, r) = registry();
        assert_eq!(r.list().len(), loader::BUNDLED.len());
        let honest = r.compose("").unwrap();
        assert!(honest.honest && honest.system.contains("Honesty about evidence"), "no-false-claims is on by default");
        r.set_enabled("markdown-memory-sync", true).unwrap();
        let p = r.compose("You are orchopork.").unwrap();
        let a = p.system.find("anti-sycophancy-terse").unwrap();
        let b = p.system.find("markdown-memory-sync").unwrap();
        assert!(p.system.starts_with("You are orchopork.") && a < b);
    }

    #[test]
    fn role_playbooks_reach_only_their_role() {
        let (_t, r) = registry();
        let single = r.compose("").unwrap().system;
        assert!(!single.contains("team-protocol") && !single.contains("ROLE: LEAD"), "single agents get no orchestra roles");
        let lead = r.compose_for("", Audience::Node("lead")).unwrap().system;
        assert!(lead.contains("ROLE: LEAD") && lead.contains("team-protocol") && !lead.contains("ROLE: CODER"));
        let coder = r.compose_for("", Audience::Node("coder")).unwrap().system;
        assert!(coder.contains("ROLE: CODER") && !coder.contains("ROLE: LEAD"));
        let unknown = r.compose_for("", Audience::Node("")).unwrap().system;
        assert!(unknown.contains("team-protocol") && !unknown.contains("ROLE: "), "an unrecognised role gets the team rules only");
        let only = r.compose_role("dispatcher").unwrap().system;
        assert!(only.contains("ROLE: DISPATCHER") && !only.contains("anti-sycophancy"));
        for key in crate::storage::ROLE_KEYS {
            assert!(r.compose_role(key).unwrap().system.contains("ROLE: "), "{key} has a playbook");
        }
    }

    #[test]
    fn a_new_default_skill_is_enabled_once_on_an_existing_workspace() {
        let (t, r) = registry();
        r.set_enabled("role-coder", false).unwrap();
        std::fs::remove_file(t.path().join("skills/role-tester.yaml")).unwrap();
        r.set_enabled("role-tester", false).ok();
        drop(r);
        let r = SkillRegistry::open(t.path().join("skills"), t.path().join("enabled.yaml")).unwrap();
        let on: Vec<String> = r.enabled_skills().into_iter().map(|s| s.name).collect();
        assert!(on.contains(&"role-tester".to_string()), "re-installed file is switched on");
        assert!(!on.contains(&"role-coder".to_string()), "a skill the user switched off stays off");
    }

    #[test]
    fn toggle_persists_and_hot_reload_picks_up_custom_skill() {
        let (t, r) = registry();
        r.set_enabled("test-driven-loop", false).unwrap();
        std::fs::write(
            t.path().join("skills/haiku.md"),
            "---\nname: haiku\nversion: 0.1.0\ntype: system_modifier\ndescription: d\n---\nAnswer in haiku.\n",
        )
        .unwrap();
        std::fs::write(t.path().join("skills/broken.yaml"), "name: Bad Name").unwrap();
        let errs = r.reload().unwrap();
        assert_eq!(errs.len(), 1);
        assert!(r.contains("haiku"));
        r.set_enabled("haiku", true).unwrap();
        let p = r.compose("").unwrap();
        assert!(p.system.contains("Answer in haiku.") && !p.system.contains("failing test"));
    }

    #[test]
    fn conflicts_are_rejected() {
        let mk = |name: &str, c: &str| Skill {
            name: name.into(),
            version: "1".into(),
            kind: SkillType::SystemModifier,
            description: String::new(),
            prompt_injection: "x".into(),
            priority: 1,
            conflicts_with: vec![c.into()],
            roles: vec![],
            tools: vec![],
            validator: None,
        };
        assert!(compose("", &[mk("a", "b"), mk("b", "a")]).is_err());
    }

    #[test]
    fn enabling_a_conflicting_skill_is_rejected_and_state_is_unchanged() {
        let (t, r) = registry();
        std::fs::write(
            t.path().join("skills/chatty.yaml"),
            "name: chatty\nversion: '1'\ntype: system_modifier\ndescription: d\nprompt_injection: Be warm.\nconflicts_with: [anti-sycophancy-terse]\n",
        )
        .unwrap();
        r.reload().unwrap();
        let before: Vec<_> = r.enabled_skills().into_iter().map(|s| s.name).collect();
        assert!(matches!(r.set_enabled("chatty", true), Err(Error::Skill(_))));
        let after: Vec<_> = r.enabled_skills().into_iter().map(|s| s.name).collect();
        assert_eq!(before, after);
    }
}
