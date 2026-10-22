//! Skill engine: declarative tone/persona/tool modules loaded from
//! `.orchopork/skills/`, toggled and hot-reloaded at runtime.

pub mod compose;
pub mod loader;
pub mod model;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::RwLock;

use serde::Serialize;

pub use compose::{ComposedPrompt, Scope, Violation, compose, compose_for};
pub use model::{Skill, SkillType};

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
        loader::install_bundled(&reg.dir)?;
        let first_run = !reg.state_file.exists();
        reg.reload()?;
        if first_run {
            let mut g = reg.inner.write().unwrap();
            g.enabled =
                loader::DEFAULT_ENABLED.iter().filter(|n| g.skills.contains_key(**n)).map(|n| n.to_string()).collect();
            drop(g);
            reg.persist()?;
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
        let mut next = self.inner.read().unwrap().enabled.clone();
        if on {
            next.insert(name.into());
        } else {
            next.remove(name);
        }
        self.replace_enabled(next)
    }

    pub fn set_enabled_exact(&self, names: &[String]) -> Result<()> {
        self.replace_enabled(names.iter().cloned().collect())
    }

    /// Validates the whole new set (unknown names, declared conflicts,
    /// duplicate tools) *before* committing it, so an invalid combination
    /// is rejected at toggle time instead of failing the next run step.
    fn replace_enabled(&self, next: BTreeSet<String>) -> Result<()> {
        {
            let mut g = self.inner.write().unwrap();
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

    pub fn compose(&self, base: &str) -> Result<ComposedPrompt> {
        compose(base, &self.enabled_skills())
    }

    /// The enabled skills that apply to `scope` (see `Skill::applies_to`).
    pub fn compose_for(&self, base: &str, scope: &Scope) -> Result<ComposedPrompt> {
        compose_for(base, &self.enabled_skills(), scope)
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
        assert_eq!(r.list().len(), 7);
        r.set_enabled("markdown-memory-sync", true).unwrap();
        let p = r.compose("You are orchopork.").unwrap();
        let a = p.system.find("anti-sycophancy-terse").unwrap();
        let b = p.system.find("markdown-memory-sync").unwrap();
        assert!(p.system.starts_with("You are orchopork.") && a < b);
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
    fn skills_apply_only_to_their_scopes() {
        let (_t, r) = registry();
        r.set_enabled("markdown-memory-sync", true).unwrap();
        let has = |scope: &Scope, needle: &str| r.compose_for("", scope).unwrap().system.contains(needle);
        let worker = Scope::agent(false, false, "coder", "Coder", "writes the parser");
        let docs = Scope::agent(false, false, "docs", "Docs writer", "documents it");
        let lead = Scope::agent(true, true, "lead", "Lead", "integrates");
        assert!(
            has(&worker, "Before modifying production code")
                && !has(&docs, "Before modifying production code")
                && !has(&lead, "Before modifying production code")
        );
        assert!(
            has(&Scope::of(&["solo"]), "Before modifying production code")
                && !has(&Scope::of(&["critic"]), "Before modifying production code")
        );
        assert!(has(&lead, "STATE.md") && !has(&worker, "STATE.md"));
        assert!(has(&docs, "anti-sycophancy-terse"), "unscoped skills apply everywhere");
        assert!(
            r.compose("").unwrap().system.contains("Before modifying production code"),
            "unscoped compose keeps every skill"
        );
    }

    #[test]
    fn team_templates_reach_the_designer_and_scoped_validators_gate_reports() {
        let (t, r) = registry();
        let write =
            |name: &str, body: &str| std::fs::write(t.path().join(format!("skills/{name}.yaml")), body).unwrap();
        write(
            "unit-tests",
            "name: unit-tests\nversion: '1'\ntype: validator\ndescription: d\napplies_to: [role:test]\nvalidator: {command: 'cargo test --lib'}\n",
        );
        write(
            "fmt",
            "name: fmt\nversion: '1'\ntype: validator\ndescription: d\nvalidator: {command: 'cargo fmt --check'}\n",
        );
        r.reload().unwrap();
        r.set_enabled("unit-tests", true).unwrap();
        r.set_enabled("fmt", true).unwrap();
        let note = r.compose_for("", &Scope::of(&["designer"])).unwrap().teams_note();
        assert!(note.contains("## team-bugfix:") && note.contains("- reproducer (parent lead)"), "{note}");
        let tester = r.compose_for("", &Scope::agent(false, false, "tester", "Tester", "tests")).unwrap();
        assert_eq!(tester.report_gates(), [("unit-tests".to_string(), "cargo test --lib".to_string())]);
        assert_eq!(tester.verify_commands().len(), 2);
        let coder = r.compose_for("", &Scope::agent(false, false, "coder", "Coder", "code")).unwrap();
        assert!(coder.report_gates().is_empty(), "unscoped validators gate only the final finish");
    }

    #[test]
    fn bad_scopes_and_teams_are_rejected() {
        let bad = |rest: &str| {
            let yaml = format!("name: a\nversion: '1'\ndescription: d\n{rest}");
            loader::parse_skill(std::path::Path::new("x.yaml"), &yaml).is_err()
        };
        assert!(bad("type: system_modifier\nprompt_injection: x\napplies_to: [boss]\n"));
        assert!(bad("type: team_template\nteam: []\n"));
        assert!(bad("type: team_template\nteam: [{id: a, task: t}, {id: b, task: t}]\n"));
        assert!(!bad("type: system_modifier\nprompt_injection: x\napplies_to: [worker, '!role:doc']\n"));
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
            tools: vec![],
            validator: None,
            applies_to: vec![],
            team: vec![],
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
