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
    /// the enabled set (defaults to all bundled skills on first run).
    pub fn open(dir: impl Into<PathBuf>, state_file: impl Into<PathBuf>) -> Result<Self> {
        let reg = Self { dir: dir.into(), state_file: state_file.into(), inner: RwLock::default() };
        loader::install_bundled(&reg.dir)?;
        let first_run = !reg.state_file.exists();
        reg.reload()?;
        if first_run {
            let mut g = reg.inner.write().unwrap();
            g.enabled = loader::BUNDLED
                .iter()
                .filter_map(|(f, _)| f.strip_suffix(".yaml"))
                .filter(|n| g.skills.contains_key(*n))
                .map(String::from)
                .collect();
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

    pub fn set_enabled(&self, name: &str, on: bool) -> Result<()> {
        {
            let mut g = self.inner.write().unwrap();
            if !g.skills.contains_key(name) {
                return Err(Error::NotFound(format!("skill {name}")));
            }
            if on { g.enabled.insert(name.into()); } else { g.enabled.remove(name); }
        }
        self.persist()
    }

    pub fn set_enabled_exact(&self, names: &[String]) -> Result<()> {
        {
            let mut g = self.inner.write().unwrap();
            if let Some(bad) = names.iter().find(|n| !g.skills.contains_key(*n)) {
                return Err(Error::NotFound(format!("skill {bad}")));
            }
            g.enabled = names.iter().cloned().collect();
        }
        self.persist()
    }

    /// Create/overwrite a custom skill from the UI, then hot-reload.
    pub fn save_custom(&self, skill: &Skill) -> Result<()> {
        skill.validate()?;
        std::fs::write(self.dir.join(format!("{}.yaml", skill.name)), serde_yaml::to_string(skill)?)?;
        self.reload().map(drop)
    }

    pub fn enabled_skills(&self) -> Vec<Skill> {
        let g = self.inner.read().unwrap();
        g.enabled.iter().filter_map(|n| g.skills.get(n).cloned()).collect()
    }

    pub fn compose(&self, base: &str) -> Result<ComposedPrompt> {
        compose(base, &self.enabled_skills())
    }

    fn persist(&self) -> Result<()> {
        let enabled = self.inner.read().unwrap().enabled.clone();
        if let Some(p) = self.state_file.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(&self.state_file, serde_yaml::to_string(&enabled)?)?;
        Ok(())
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
        assert_eq!(r.list().len(), 4);
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
    fn conflicts_are_rejected() {
        let mk = |name: &str, c: &str| Skill {
            name: name.into(), version: "1".into(), kind: SkillType::SystemModifier,
            description: String::new(), prompt_injection: "x".into(), priority: 1,
            conflicts_with: vec![c.into()], tools: vec![], validator: None,
        };
        assert!(compose("", &[mk("a", "b"), mk("b", "a")]).is_err());
    }
}
