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

/// (pathspec excluding it from checkpoints, `git clean -e` pattern).
const DEPENDENCY_DIRS: &[(&str, &str)] = &[
    (":(exclude,glob)**/node_modules/**", "node_modules"),
    (":(exclude,glob)**/.venv/**", ".venv"),
    (":(exclude,glob)**/__pycache__/**", "__pycache__"),
    (":(exclude,glob).orchopork-shots/**", ".orchopork-shots"),
];

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
            // Dependency trees (node_modules) easily exceed Windows' 260-char
            // path limit; without this `git add` fails on them.
            .args(["-c", "core.longpaths=true"])
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
        // Installed dependencies are not the agent's work: checkpointing them
        // bloats the repository and slows every step, even when the project
        // has no .gitignore for them yet.
        let mut args = vec!["add", "-A", "--", "."];
        args.extend(DEPENDENCY_DIRS.iter().map(|(pathspec, _)| *pathspec));
        self.run(&args).await?;
        self.run(&["commit", "--allow-empty", "--no-verify", "-q", "-m", message]).await?;
        self.head().await?.ok_or_else(|| Error::Git("no HEAD after commit".into()))
    }

    pub async fn reset_hard(&self, commit: &str) -> Result<()> {
        validate_commit(commit)?;
        self.run(&["reset", "--hard", "-q", commit]).await?;
        // Files the agent created after the target step are untracked there;
        // remove them too so the tree really matches the checkpoint. Ignored
        // files (build caches) are kept.
        // Dependencies are never checkpointed (see `commit_all`); keep them.
        let mut args = vec!["clean", "-fdq"];
        for (_, dir) in DEPENDENCY_DIRS {
            args.extend(["-e", dir]);
        }
        self.run(&args).await.map(drop)
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

    /// Files tracked at `commit`, repo-relative and sorted. Unlike
    /// `ls_files`, this never changes while a run edits its tree, so it can
    /// sit in a cached prompt prefix.
    pub async fn ls_files_at(&self, commit: &str) -> Result<Vec<String>> {
        validate_commit(commit)?;
        // A commit's tree never changes, and every turn of a run asks for
        // the same one: list it once per process (big repos list slowly).
        static CACHE: std::sync::Mutex<Vec<(String, std::sync::Arc<Vec<String>>)>> = std::sync::Mutex::new(Vec::new());
        if let Some((_, v)) = CACHE.lock().unwrap().iter().find(|(c, _)| c == commit) {
            return Ok(v.as_ref().clone());
        }
        let out = self.run(&["ls-tree", "-r", "--name-only", commit]).await?;
        let mut files: Vec<String> = out.lines().map(str::to_string).collect();
        files.sort();
        let mut cache = CACHE.lock().unwrap();
        if cache.len() >= 16 {
            cache.remove(0);
        }
        cache.push((commit.to_string(), std::sync::Arc::new(files.clone())));
        Ok(files)
    }

    /// `git diff --name-status` of the working tree against `base`: one
    /// `M path` / `A path` / `D path` line per changed file.
    pub async fn changed_names(&self, base: &str) -> Result<String> {
        validate_commit(base)?;
        self.run(&["diff", "--name-status", "--no-color", "--no-renames", base]).await
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

    /// (stat, patch) between two commits.
    pub async fn diff_between(&self, from: &str, to: &str) -> Result<(String, String)> {
        validate_commit(from)?;
        validate_commit(to)?;
        let stat = self.run(&["diff", "--stat", "--no-color", from, to]).await?;
        let patch = self.run(&["diff", "--no-color", from, to]).await?;
        Ok((stat, patch))
    }

    /// Lines added and removed since `base` (working tree included).
    pub async fn numstat_from(&self, base: &str) -> Result<(u64, u64)> {
        validate_commit(base)?;
        let out = self.run(&["diff", "--numstat", base]).await?;
        Ok(out.lines().fold((0, 0), |(a, r), l| {
            let mut f = l.split_whitespace();
            let n = |x: Option<&str>| x.and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            (a + n(f.next()), r + n(f.next()))
        }))
    }

    /// (stat, patch) of the working tree (committed or not) against `base`.
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

    /// Merges `branch` into this worktree and commits. On conflicts the
    /// merge is left in progress and the conflicted files are returned:
    /// resolve them and `commit_all`, or `abort_merge`.
    pub async fn merge_or_conflicts(&self, branch: &str, message: &str) -> Result<std::result::Result<(), Vec<String>>> {
        validate_ref_name(&format!("refs/heads/{branch}"))?;
        let args = ["merge", "--no-ff", "--no-verify", "-m", message, branch];
        let out = self.output(self.command(&args), &args).await?;
        if out.status.success() {
            return Ok(Ok(()));
        }
        let files: Vec<String> = self
            .run(&["diff", "--name-only", "--diff-filter=U"])
            .await
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .filter(|l| !l.is_empty())
            .collect();
        if files.is_empty() {
            self.abort_merge().await;
            let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            return Err(Error::Git(format!("merge of {branch} failed: {}", text.trim())));
        }
        Ok(Err(files))
    }

    /// Abandons a merge in progress (no-op without one).
    pub async fn abort_merge(&self) {
        let _ = self.run(&["merge", "--abort"]).await;
    }

    /// Merges `branch` into the current working directory.
    pub async fn merge(&self, branch: &str, message: &str) -> Result<String> {
        validate_ref_name(&format!("refs/heads/{branch}"))?;
        let args = ["merge", "--no-ff", "-m", message, branch];
        let out = self.output(self.command(&args), &args).await?;
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        if !out.status.success() {
            // Never leave the user's checkout half-merged with conflict markers.
            let _ = self.run(&["merge", "--abort"]).await;
            return Err(Error::Git(format!("merge failed (aborted, your checkout is unchanged): {}", text.trim())));
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

use crate::fsutil::base64;

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

    /// A conflicting merge must not leave the user's checkout mid-merge.
    #[tokio::test]
    async fn a_conflicting_merge_is_aborted() {
        let d = tempfile::tempdir().unwrap();
        let repo = GitRepo::new(d.path());
        repo.init().await.unwrap();
        std::fs::write(d.path().join("a.txt"), "base\n").unwrap();
        let base = repo.commit_all("base").await.unwrap();
        let wt_path = d.path().join(STATE_DIR).join("worktrees/r1");
        repo.worktree_add(&wt_path, "orchopork/r1", &base).await.unwrap();
        std::fs::write(wt_path.join("a.txt"), "theirs\n").unwrap();
        GitRepo::new(&wt_path).commit_all("theirs").await.unwrap();
        std::fs::write(d.path().join("a.txt"), "mine\n").unwrap();
        let mine = repo.commit_all("mine").await.unwrap();

        let err = repo.merge("orchopork/r1", "merge").await.unwrap_err().to_string();
        assert!(err.contains("aborted"), "{err}");
        // (`core.autocrlf` may turn the line ending into CRLF on Windows.)
        assert_eq!(std::fs::read_to_string(d.path().join("a.txt")).unwrap().trim_end(), "mine");
        assert_eq!(repo.head().await.unwrap().as_deref(), Some(mine.as_str()));
        assert!(repo.run(&["rev-parse", "-q", "--verify", "MERGE_HEAD"]).await.is_err(), "no merge in progress");
    }

    #[tokio::test]
    async fn checkpoints_skip_dependency_folders_and_rewinds_keep_them() {
        let d = tempfile::tempdir().unwrap();
        let repo = GitRepo::new(d.path());
        repo.init().await.unwrap();
        let base = repo.commit_all("base").await.unwrap();
        let deep = d.path().join("web").join("node_modules").join("pkg");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("index.js"), "x").unwrap();
        std::fs::write(d.path().join("web").join("app.js"), "y").unwrap();
        repo.commit_all("work").await.unwrap();
        let files = repo.ls_files_at(&repo.head().await.unwrap().unwrap()).await.unwrap();
        assert!(files.contains(&"web/app.js".to_string()), "{files:?}");
        assert!(!files.iter().any(|f| f.contains("node_modules")), "{files:?}");
        repo.reset_hard(&base).await.unwrap();
        assert!(!d.path().join("web").join("app.js").exists(), "the agent's work is rewound");
        assert!(deep.join("index.js").exists(), "installed dependencies stay");
    }
}
