//! Onboarding state machine: Welcome -> Permissions -> Vcs -> Skills -> Dashboard.
//!
//! `Wizard::apply` is pure (no IO) so the transition rules are unit-testable.
//! Probing happens in the handlers and is fed in as events. Secrets are
//! never stored here: this state is serialized to the browser.
//!
//! Completion is persisted as `config.onboarded` in the chosen workspace,
//! so restarting the server in an onboarded workspace goes straight to the
//! dashboard.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::RemoteAuth;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VcsStatus {
    pub is_repo: bool,
    pub has_commits: bool,
    pub branch: Option<String>,
    pub origin: Option<String>,
    pub remote_auth: RemoteAuth,
    pub has_github_token: bool,
}

#[derive(Debug, Clone)]
pub enum Event {
    Initialize,
    PermissionsChecked {
        report: PermissionReport,
    },
    VcsConfigured {
        status: VcsStatus,
    },
    SkillsConfirmed,
    /// The chosen workspace was already set up: skip to the dashboard.
    Resume,
    Back,
}

#[derive(Debug, Clone, Serialize)]
pub struct Wizard {
    pub step: Step,
    pub path: &'static str,
    pub permissions: Option<PermissionReport>,
    pub vcs: Option<VcsStatus>,
}

impl Default for Wizard {
    fn default() -> Self {
        Self { step: Step::Welcome, path: Step::Welcome.path(), permissions: None, vcs: None }
    }
}

impl Wizard {
    pub fn apply(&mut self, ev: Event) -> Result<Step> {
        let bad = |m: &str| Err(Error::Wizard(m.into()));
        match (self.step, ev) {
            (Step::Welcome, Event::Initialize) => {}
            (Step::Permissions, Event::PermissionsChecked { report }) => {
                let ok = report.ok();
                // Keep a failing report too, so the UI can show what failed.
                self.permissions = Some(report);
                if !ok {
                    return bad("the workspace is not readable/writable or a required tool is missing");
                }
            }
            (Step::Vcs, Event::VcsConfigured { status }) => {
                let ready = status.is_repo;
                self.vcs = Some(status);
                if !ready {
                    return bad("the workspace must be a git repository (tick \"initialize\" to create one)");
                }
            }
            (Step::Skills, Event::SkillsConfirmed) => {}
            (_, Event::Resume) => return Ok(self.goto(Step::Dashboard)),
            (_, Event::Back) => {
                let prev = self.step.prev().ok_or_else(|| Error::Wizard("already at the first step".into()))?;
                return Ok(self.goto(prev));
            }
            (step, ev) => return Err(Error::Wizard(format!("{ev:?} is not valid in step {step:?}"))),
        }
        let next = self.step.next().expect("terminal step handled above");
        Ok(self.goto(next))
    }

    fn goto(&mut self, s: Step) -> Step {
        self.step = s;
        self.path = s.path();
        s
    }

    /// Route guard: a page is reachable only if every earlier step is done.
    pub fn can_access(&self, target: Step) -> bool {
        target <= self.step
    }

    pub fn is_complete(&self) -> bool {
        self.step == Step::Dashboard
    }
}

/// (display name, binaries to try, required). Only git is required.
const TOOLCHAINS: &[(&str, &[&str], bool)] = &[
    ("git", &["git"], true),
    ("cargo", &["cargo"], false),
    ("python", &["python3", "python"], false),
    ("node", &["node"], false),
];

/// Verify the workspace directory (read + write via a real probe file) and
/// probe toolchains with `--version`.
pub async fn check_permissions(workspace: &Path) -> PermissionReport {
    let readable = tokio::fs::read_dir(workspace).await.is_ok();
    let probe = workspace.join(format!(".orchopork-probe-{}", std::process::id()));
    let writable = tokio::fs::write(&probe, b"x").await.is_ok();
    let _ = tokio::fs::remove_file(&probe).await;

    let mut toolchains = Vec::new();
    for (name, bins, required) in TOOLCHAINS {
        let mut version = None;
        for bin in *bins {
            let out = tokio::process::Command::new(bin)
                .arg("--version")
                .stdin(std::process::Stdio::null())
                .kill_on_drop(true)
                .output()
                .await;
            if let Ok(o) = out
                && o.status.success()
            {
                let text = if o.stdout.is_empty() { o.stderr } else { o.stdout };
                version = Some(String::from_utf8_lossy(&text).lines().next().unwrap_or("").trim().to_string());
                break;
            }
        }
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

    fn vcs(is_repo: bool) -> VcsStatus {
        VcsStatus {
            is_repo,
            has_commits: false,
            branch: None,
            origin: None,
            remote_auth: RemoteAuth::None,
            has_github_token: false,
        }
    }

    #[test]
    fn happy_path_reaches_dashboard_and_gates_routes() {
        let mut w = Wizard::default();
        assert!(!w.can_access(Step::Permissions));
        w.apply(Event::Initialize).unwrap();
        w.apply(Event::PermissionsChecked { report: report(true) }).unwrap();
        assert!(w.apply(Event::VcsConfigured { status: vcs(false) }).is_err());
        w.apply(Event::VcsConfigured { status: vcs(true) }).unwrap();
        assert!(!w.can_access(Step::Dashboard));
        assert_eq!(w.apply(Event::SkillsConfirmed).unwrap(), Step::Dashboard);
        assert!(w.is_complete() && w.can_access(Step::Welcome));
        assert_eq!(w.path, "/app");
    }

    #[test]
    fn failed_checks_do_not_advance_and_steps_cannot_be_skipped() {
        let mut w = Wizard::default();
        assert!(w.apply(Event::PermissionsChecked { report: report(true) }).is_err());
        w.apply(Event::Initialize).unwrap();
        assert!(w.apply(Event::PermissionsChecked { report: report(false) }).is_err());
        assert_eq!(w.step, Step::Permissions);
        assert!(w.permissions.is_some(), "failing report is kept for the UI");
        assert_eq!(w.apply(Event::Back).unwrap(), Step::Welcome);
    }

    #[test]
    fn resume_skips_to_the_dashboard() {
        let mut w = Wizard::default();
        w.apply(Event::Initialize).unwrap();
        assert_eq!(w.apply(Event::Resume).unwrap(), Step::Dashboard);
    }
}
