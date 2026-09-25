//! Hardened git access: no shell, fixed identity, no prompts, validated revs.
//!
//! Runs never touch the user's checkout: each run gets its own branch in
//! its own `git worktree` under `.orchopork/worktrees/`, so the user can keep
//! working (or run several agents) while orchopork commits and rewinds.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use tokio::process::Command;

use crate::error::{Error, Result};

/// Directory holding orchopork's own state (DB, skills, worktrees). Never
/// versioned: otherwise checkpoints would commit the checkpoint database.
pub const STATE_DIR: &str = ".orchopork";

fn same_dir(a: &Path, b: &Path) -> bool {
    let c = |p: &Path| crate::fsutil::canonical(p).unwrap_or_else(|_| p.to_path_buf());
    c(a) == c(b)
}

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

    fn command(&self, args: &[&str]) -> Command {
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.root)
            .args(["-c", "user.name=orchopork", "-c", "user.email=orchopork@localhost", "-c", "commit.gpgsign=false"])
            .args(["-c", "core.quotepath=off"])
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::null())
            .kill_on_drop(true);
        cmd
    }

    async fn output(&self, mut cmd: Command, args: &[&str]) -> Result<std::process::Output> {
        cmd.output().await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::Git("the `git` binary was not found on PATH".into())
            } else {
                Error::Git(format!("git {}: {e}", args.join(" ")))
            }
        })
    }

    /// Runs git and returns stdout with trailing whitespace removed.
    pub(crate) async fn run(&self, args: &[&str]) -> Result<String> {
        let out = self.output(self.command(args), args).await?;
        if !out.status.success() {
            return Err(Error::Git(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim())));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
    }

    /// True only when the root is itself the top of a repository. A folder
    /// nested inside some other repo is *not* a repo of its own: runs there
    /// would branch the outer project instead of this folder.
    pub async fn is_repo(&self) -> bool {
        match self.toplevel().await {
            Some(top) => same_dir(&top, &self.root),
            None => false,
        }
    }

    /// The enclosing repository when the root sits inside another repo.
    pub async fn enclosing_repo(&self) -> Option<PathBuf> {
        self.toplevel().await.filter(|t| !same_dir(t, &self.root))
    }

    async fn toplevel(&self) -> Option<PathBuf> {
        self.run(&["rev-parse", "--show-toplevel"]).await.ok().filter(|s| !s.is_empty()).map(PathBuf::from)
    }

    /// `git init` if needed, then exclude the state dir. Only called when the
    /// user explicitly opts in (setup wizard / `orchopork init --git-init`).
    pub async fn init(&self) -> Result<()> {
        if !self.is_repo().await {
            self.run(&["init"]).await?;
        }
        self.ensure_state_dir_excluded().await
    }

    /// Adds `.orchopork/` to the repo's `info/exclude` (idempotent), so the
    /// user's own `git add -A` never picks up the DB, secrets or worktrees.
    pub async fn ensure_state_dir_excluded(&self) -> Result<()> {
        let rel = self.run(&["rev-parse", "--git-path", "info/exclude"]).await?;
        let path = self.root.join(rel); // join keeps absolute paths as-is
        let existing = tokio::fs::read_to_string(&path).await.unwrap_or_default();
        let entry = format!("/{STATE_DIR}/");
        if !existing.lines().any(|l| l.trim() == entry || l.trim() == format!("{STATE_DIR}/")) {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            let sep = if existing.is_empty() || existing.ends_with('\n') { "" } else { "\n" };
            tokio::fs::write(&path, format!("{existing}{sep}{entry}\n")).await?;
        }
        Ok(())
    }

    pub async fn head(&self) -> Result<Option<String>> {
        Ok(self.run(&["rev-parse", "--verify", "-q", "HEAD^{commit}"]).await.ok().filter(|s| !s.is_empty()))
    }

    /// Worktrees need a commit to branch from; a freshly initialized repo
    /// gets an empty root commit.
    pub async fn ensure_initial_commit(&self) -> Result<String> {
        if let Some(h) = self.head().await? {
            return Ok(h);
        }
        self.run(&["commit", "--allow-empty", "--no-verify", "-m", "orchopork: initial commit"]).await?;
        self.head().await?.ok_or_else(|| Error::Git("no HEAD after initial commit".into()))
    }

    pub async fn current_branch(&self) -> Option<String> {
        self.run(&["symbolic-ref", "--short", "-q", "HEAD"]).await.ok().filter(|s| !s.is_empty())
    }

    pub async fn remote_url(&self, name: &str) -> Option<String> {
        self.run(&["remote", "get-url", name]).await.ok().filter(|s| !s.is_empty())
    }

    /// Stage everything and commit (even if nothing changed) so every
    /// checkpoint maps 1:1 to a commit. Hooks are skipped: these are machine
    /// checkpoints, not authored history.
    pub async fn commit_all(&self, message: &str) -> Result<String> {
        self.run(&["add", "-A"]).await?;
        self.run(&["commit", "--allow-empty", "--no-verify", "-q", "-m", message]).await?;
        self.head().await?.ok_or_else(|| Error::Git("no HEAD after commit".into()))
    }

    pub async fn reset_hard(&self, commit: &str) -> Result<()> {
        validate_commit(commit)?;
        self.run(&["reset", "--hard", "-q", commit]).await?;
        // Files the agent created after the target step are untracked there;
        // remove them too so the tree really matches the checkpoint. Ignored
        // files (build caches) are kept.
        self.run(&["clean", "-fdq"]).await.map(drop)
    }

    pub async fn update_ref(&self, name: &str, commit: &str) -> Result<()> {
        validate_commit(commit)?;
        validate_ref_name(name)?;
        self.run(&["update-ref", name, commit]).await.map(drop)
    }

    pub async fn worktree_add(&self, path: &Path, branch: &str, base: &str) -> Result<()> {
        validate_commit(base)?;
        validate_ref_name(&format!("refs/heads/{branch}"))?;
        let p = path.to_string_lossy();
        self.run(&["worktree", "add", "-q", "-b", branch, &p, base]).await.map(drop)
    }

    /// Removes the worktree and its branch. Tolerates either being gone
    /// already (a user may have cleaned up by hand).
    pub async fn worktree_remove(&self, path: &Path, branch: &str) -> Result<()> {
        validate_ref_name(&format!("refs/heads/{branch}"))?;
        let p = path.to_string_lossy();
        if path.exists() {
            self.run(&["worktree", "remove", "--force", &p]).await?;
        }
        let _ = self.run(&["worktree", "prune"]).await;
        let _ = self.run(&["branch", "-D", "-q", branch]).await;
        Ok(())
    }

    /// Tracked + untracked-but-not-ignored files, repo-relative.
    pub async fn ls_files(&self) -> Result<Vec<String>> {
        let out = self.run(&["ls-files", "--cached", "--others", "--exclude-standard"]).await?;
        let mut files: Vec<String> = out.lines().map(str::to_string).collect();
        files.sort();
        files.dedup();
        Ok(files)
    }

    /// `git grep` over tracked and untracked files. No match is `Ok("")`.
    pub async fn grep(&self, pattern: &str, path: &str) -> Result<String> {
        let args = ["grep", "-n", "-I", "--untracked", "--no-color", "-E", "-e", pattern, "--", path];
        let out = self.output(self.command(&args), &args).await?;
        match out.status.code() {
            Some(0) => Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string()),
            Some(1) => Ok(String::new()),
            _ => Err(Error::Git(String::from_utf8_lossy(&out.stderr).trim().to_string())),
        }
    }

    /// Diff of the working tree (committed or not) against `base`.
    pub async fn diff_from(&self, base: &str) -> Result<(String, String)> {
        validate_commit(base)?;
        let stat = self.run(&["diff", "--stat", "--no-color", base]).await?;
        let patch = self.run(&["diff", "--no-color", base]).await?;
        Ok((stat, patch))
    }

    /// Pushes `branch` to `origin`. With a token, it is sent as an HTTP
    /// Basic auth header through git's environment config (never argv, so
    /// it does not show up in `ps`).
    pub async fn push(&self, branch: &str, token: Option<&str>) -> Result<String> {
        validate_ref_name(&format!("refs/heads/{branch}"))?;
        let args = ["push", "--porcelain", "-u", "origin", branch];
        let mut cmd = self.command(&args);
        if let Some(t) = token {
            let header = format!("Authorization: Basic {}", base64(format!("x-access-token:{t}").as_bytes()));
            cmd.env("GIT_CONFIG_COUNT", "1")
                .env("GIT_CONFIG_KEY_0", "http.extraHeader")
                .env("GIT_CONFIG_VALUE_0", header);
        }
        let out = self.output(cmd, &args).await?;
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        if !out.status.success() {
            return Err(Error::Git(format!("push failed: {}", text.trim())));
        }
        Ok(text.trim().to_string())
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

fn validate_ref_name(name: &str) -> Result<()> {
    let ok = name.starts_with("refs/")
        && !name.contains("..")
        && !name.ends_with('/')
        && !name.ends_with(".lock")
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.'));
    if ok { Ok(()) } else { Err(Error::Git(format!("refusing ref name {name:?}"))) }
}

fn base64(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(T[(n >> (18 - 6 * i)) as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        for (i, o) in [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foobar", "Zm9vYmFy")] {
            assert_eq!(base64(i.as_bytes()), o);
        }
    }

    #[test]
    fn ref_and_commit_validation_rejects_option_like_input() {
        assert!(validate_commit("--upload-pack=x").is_err());
        assert!(validate_ref_name("refs/heads/orchopork/abc123").is_ok());
        assert!(validate_ref_name("refs/heads/a b").is_err());
        assert!(validate_ref_name("refs/heads/../x").is_err());
    }

    #[tokio::test]
    async fn worktree_lifecycle_leaves_the_main_checkout_alone() {
        let d = tempfile::tempdir().unwrap();
        let repo = GitRepo::new(d.path());
        repo.init().await.unwrap();
        let base = repo.ensure_initial_commit().await.unwrap();
        let main_branch = repo.current_branch().await;

        let wt_path = d.path().join(STATE_DIR).join("worktrees/r1");
        repo.worktree_add(&wt_path, "orchopork/r1", &base).await.unwrap();
        let wt = GitRepo::new(&wt_path);
        tokio::fs::write(wt_path.join("new.txt"), "hi").await.unwrap();
        let c1 = wt.commit_all("step").await.unwrap();
        assert_ne!(c1, base);
        assert_eq!(repo.head().await.unwrap().as_deref(), Some(base.as_str()));
        assert_eq!(repo.current_branch().await, main_branch);
        assert!(repo.ls_files().await.unwrap().is_empty(), "state dir must stay invisible to the main repo");

        let (stat, patch) = wt.diff_from(&base).await.unwrap();
        assert!(stat.contains("new.txt") && patch.contains("+hi"));
        assert!(wt.grep("h.", ".").await.unwrap().contains("new.txt:1:hi"));
        assert!(wt.grep("absent", ".").await.unwrap().is_empty());

        tokio::fs::write(wt_path.join("stray.txt"), "x").await.unwrap();
        wt.reset_hard(&base).await.unwrap();
        assert!(!wt_path.join("new.txt").exists() && !wt_path.join("stray.txt").exists());

        repo.worktree_remove(&wt_path, "orchopork/r1").await.unwrap();
        assert!(!wt_path.exists());
        assert!(repo.run(&["branch", "--list", "orchopork/r1"]).await.unwrap().is_empty());
    }

    /// A folder inside another repo is not a repo of its own: runs there
    /// must not branch the outer project.
    #[tokio::test]
    async fn nested_folder_is_not_its_own_repo() {
        let d = tempfile::tempdir().unwrap();
        GitRepo::new(d.path()).init().await.unwrap();
        let sub = d.path().join("orchotest");
        std::fs::create_dir(&sub).unwrap();
        let inner = GitRepo::new(&sub);
        assert!(!inner.is_repo().await);
        assert!(inner.enclosing_repo().await.is_some());
        inner.init().await.unwrap();
        assert!(inner.is_repo().await);
        assert!(inner.enclosing_repo().await.is_none());
    }
}
