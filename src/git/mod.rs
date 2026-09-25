//! Hardened git access: no shell, fixed identity, no prompts, validated revs.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::process::Command;

use crate::error::{Error, Result};

/// Directory holding colopork's own state (DB, skills). Never versioned:
/// otherwise checkpoints would commit the checkpoint database itself.
pub const STATE_DIR: &str = ".colopork";

#[derive(Debug, Clone)]
pub struct GitRepo {
    root: PathBuf,
}

impl GitRepo {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) async fn run(&self, args: &[&str]) -> Result<String> {
        let out = Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args([
                "-c",
                "user.name=colopork",
                "-c",
                "user.email=colopork@localhost",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::null())
            .output()
            .await?;
        if !out.status.success() {
            return Err(Error::Git(format!(
                "git {}: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    pub async fn is_repo(&self) -> bool {
        self.run(&["rev-parse", "--git-dir"]).await.is_ok()
    }

    /// Idempotent: init if needed and exclude `.colopork/` from versioning.
    pub async fn init(&self) -> Result<()> {
        if !self.is_repo().await {
            self.run(&["init"]).await?;
        }
        let rel = self.run(&["rev-parse", "--git-path", "info/exclude"]).await?;
        let path = self.root.join(rel); // join keeps absolute paths as-is
        let existing = tokio::fs::read_to_string(&path).await.unwrap_or_default();
        let entry = format!("{STATE_DIR}/");
        if !existing.lines().any(|l| l.trim() == entry) {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            let sep = if existing.is_empty() || existing.ends_with('\n') { "" } else { "\n" };
            tokio::fs::write(&path, format!("{existing}{sep}{entry}\n")).await?;
        }
        Ok(())
    }

    pub async fn head(&self) -> Result<Option<String>> {
        Ok(self.run(&["rev-parse", "--verify", "-q", "HEAD"]).await.ok())
    }

    /// Stage everything and commit (even if nothing changed) so every
    /// snapshot maps 1:1 to a commit. Hooks are skipped: these are machine
    /// checkpoints, not authored history.
    pub async fn commit_all(&self, message: &str) -> Result<String> {
        self.run(&["add", "-A"]).await?;
        self.run(&["commit", "--allow-empty", "--no-verify", "-m", message]).await?;
        self.head().await?.ok_or_else(|| Error::Git("no HEAD after commit".into()))
    }

    pub async fn reset_hard(&self, commit: &str) -> Result<()> {
        validate_commit(commit)?;
        self.run(&["reset", "--hard", commit]).await.map(drop)
    }

    pub async fn update_ref(&self, name: &str, commit: &str) -> Result<()> {
        validate_commit(commit)?;
        if !name.starts_with("refs/colopork/") || name.contains("..") || name.contains(' ') {
            return Err(Error::Git(format!("refusing ref name {name:?}")));
        }
        self.run(&["update-ref", name, commit]).await.map(drop)
    }
}

/// Only full/abbreviated hex object names: cannot be parsed as an option.
fn validate_commit(c: &str) -> Result<()> {
    if (7..=64).contains(&c.len()) && c.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(Error::Git(format!("invalid commit id {c:?}")))
    }
}
