//! Onboarding state machine: Welcome -> Permissions -> Vcs -> Skills -> Dashboard.
//!
//! `Wizard::apply` is pure (no IO) so the transition rules are unit-testable.
//! Environment probing happens in `check_permissions`, and its result is fed
//! in as an event. Secrets (PATs, API keys) are never stored here: the state
//! is serialized to the browser, so it only records *that* something is set.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Welcome,
    Permissions,
    Vcs,
    Skills,
    Dashboard,
}

impl Step {
    pub fn path(self) -> &'static str {
        match self {
            Step::Welcome => "/welcome",
            Step::Permissions => "/setup/permissions",
            Step::Vcs => "/setup/vcs",
            Step::Skills => "/setup/skills",
            Step::Dashboard => "/app",
        }
    }

    fn next(self) -> Option<Step> {
        Some(match self {
            Step::Welcome => Step::Permissions,
            Step::Permissions => Step::Vcs,
            Step::Vcs => Step::Skills,
            Step::Skills => Step::Dashboard,
            Step::Dashboard => return None,
        })
    }

    fn prev(self) -> Option<Step> {
        Some(match self {
            Step::Welcome => return None,
            Step::Permissions => Step::Welcome,
            Step::Vcs => Step::Permissions,
            Step::Skills => Step::Vcs,
            Step::Dashboard => Step::Skills,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toolchain {
    pub name: String,
    pub required: bool,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionReport {
    pub workspace: PathBuf,
    pub readable: bool,
    pub writable: bool,
    pub toolchains: Vec<Toolchain>,
}

impl PermissionReport {
    pub fn ok(&self) -> bool {
        self.readable && self.writable && self.toolchains.iter().all(|t| !t.required || t.version.is_some())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteAuth {
    None,
    Pat,
    Ssh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VcsChoice {
    pub repo_ready: bool,
    pub remote_auth: RemoteAuth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Initialize,
    PermissionsChecked { report: PermissionReport },
    VcsConfigured { choice: VcsChoice },
    SkillsConfirmed { enabled_skills: Vec<String>, providers: Vec<String> },
    Back,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wizard {
    pub step: Step,
    pub permissions: Option<PermissionReport>,
    pub vcs: Option<VcsChoice>,
    pub enabled_skills: Vec<String>,
    /// Providers configured in step 4 (e.g. "ollama", "claude"); no keys.
    pub providers: Vec<String>,
}

impl Default for Wizard {
    fn default() -> Self {
        Self { step: Step::Welcome, permissions: None, vcs: None, enabled_skills: vec![], providers: vec![] }
    }
}

impl Wizard {
    pub fn apply(&mut self, ev: Event) -> Result<Step> {
        let bad = |m: &str| Err(Error::Wizard(m.into()));
        match (self.step, ev) {
            (Step::Welcome, Event::Initialize) => {}
            (Step::Permissions, Event::PermissionsChecked { report }) => {
                if !report.ok() {
                    // Keep the report so the UI can show what failed.
                    self.permissions = Some(report);
                    return bad("workspace or required toolchains failed verification");
                }
                self.permissions = Some(report);
            }
            (Step::Vcs, Event::VcsConfigured { choice }) => {
                if !choice.repo_ready {
                    return bad("workspace must be a git repository (initialize it first)");
                }
                self.vcs = Some(choice);
            }
            (Step::Skills, Event::SkillsConfirmed { enabled_skills, providers }) => {
                if providers.is_empty() {
                    return bad("configure at least one provider (local or cloud)");
                }
                self.enabled_skills = enabled_skills;
                self.providers = providers;
            }
            (_, Event::Back) => {
                self.step = self.step.prev().ok_or_else(|| Error::Wizard("already at first step".into()))?;
                return Ok(self.step);
            }
            (step, _) => return Err(Error::Wizard(format!("event not valid in step {step:?}"))),
        }
        self.step = self.step.next().expect("terminal step handled above");
        Ok(self.step)
    }

    /// Route guard: a page is reachable only if every earlier step is done.
    pub fn can_access(&self, target: Step) -> bool {
        target <= self.step
    }

    pub fn is_complete(&self) -> bool {
        self.step == Step::Dashboard
    }
}

const TOOLCHAINS: &[(&str, bool)] = &[("git", true), ("cargo", false), ("python", false), ("node", false)];

/// Verify the workspace directory (read + write via a real probe file) and
/// probe toolchains with `--version`. Only `git` is required.
pub async fn check_permissions(workspace: &Path) -> PermissionReport {
    let readable = tokio::fs::read_dir(workspace).await.is_ok();
    let probe = workspace.join(format!(".orchopork-probe-{}", std::process::id()));
    let writable = tokio::fs::write(&probe, b"x").await.is_ok();
    let _ = tokio::fs::remove_file(&probe).await;

    let mut toolchains = Vec::new();
    for (name, required) in TOOLCHAINS {
        let out = tokio::process::Command::new(name).arg("--version").stdin(std::process::Stdio::null()).output().await;
        let version = match out {
            Ok(o) if o.status.success() => {
                Some(String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or("").trim().to_string())
            }
            _ => None,
        };
        toolchains.push(Toolchain { name: (*name).into(), required: *required, version });
    }
    PermissionReport { workspace: workspace.to_path_buf(), readable, writable, toolchains }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(ok: bool) -> PermissionReport {
        PermissionReport {
            workspace: "/w".into(),
            readable: true,
            writable: ok,
            toolchains: vec![Toolchain { name: "git".into(), required: true, version: Some("git 2".into()) }],
        }
    }

    #[test]
    fn happy_path_reaches_dashboard_and_gates_routes() {
        let mut w = Wizard::default();
        assert!(!w.can_access(Step::Permissions));
        w.apply(Event::Initialize).unwrap();
        w.apply(Event::PermissionsChecked { report: report(true) }).unwrap();
        w.apply(Event::VcsConfigured { choice: VcsChoice { repo_ready: true, remote_auth: RemoteAuth::None } }).unwrap();
        assert!(!w.can_access(Step::Dashboard));
        let s = w.apply(Event::SkillsConfirmed { enabled_skills: vec![], providers: vec!["ollama".into()] }).unwrap();
        assert_eq!(s, Step::Dashboard);
        assert!(w.is_complete() && w.can_access(Step::Welcome));
    }

    #[test]
    fn failed_checks_do_not_advance_and_steps_cannot_be_skipped() {
        let mut w = Wizard::default();
        assert!(w.apply(Event::PermissionsChecked { report: report(true) }).is_err());
        w.apply(Event::Initialize).unwrap();
        assert!(w.apply(Event::PermissionsChecked { report: report(false) }).is_err());
        assert_eq!(w.step, Step::Permissions);
        assert_eq!(w.apply(Event::Back).unwrap(), Step::Welcome);
    }
}
