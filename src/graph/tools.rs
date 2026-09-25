//! The actor's tools. File tools are confined to the run's worktree
//! (no absolute paths, no `..` escapes, no symlinks pointing outside, no
//! `.git`). `run_command` is a real shell on your machine, confined only by
//! its working directory and a timeout; turn it off in settings
//! (`limits.allow_commands`) if that is not acceptable.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use tokio::process::Command;

use super::protocol::Action;
use crate::config::Limits;
use crate::error::Result;
use crate::fsutil::clip;
use crate::git::GitRepo;
use crate::skills::model::ToolSpec;

/// Upper bound on what one tool call may store; the model sees a further
/// clipped view (`limits.tool_output_chars`).
pub const MAX_STORED_OUTPUT: usize = 64_000;
const READ_DEFAULT_LINES: i64 = 400;

pub struct ToolResult {
    pub ok: bool,
    pub output: String,
}

impl ToolResult {
    fn ok(output: impl Into<String>) -> Self {
        Self { ok: true, output: output.into() }
    }
    fn err(output: impl Into<String>) -> Self {
        Self { ok: false, output: output.into() }
    }
}

pub struct ToolBox {
    root: PathBuf,
    allow_commands: bool,
    timeout: Duration,
    custom: Vec<ToolSpec>,
}

impl ToolBox {
    pub fn new(worktree: &Path, limits: &Limits, custom: Vec<ToolSpec>) -> Result<Self> {
        Ok(Self {
            root: worktree.canonicalize()?,
            allow_commands: limits.allow_commands,
            timeout: Duration::from_secs(limits.command_timeout_secs.max(1)),
            custom,
        })
    }

    /// Tool reference for the actor's system prompt.
    pub fn describe(&self) -> String {
        let mut s = String::from(
            "- list_files {\"path\"?: string}: files under a directory (default: whole repo), respecting .gitignore.\n\
             - read_file {\"path\": string, \"start_line\"?: int, \"end_line\"?: int}: file contents with line numbers.\n\
             - search {\"pattern\": string, \"path\"?: string}: regex search (git grep -E) with file:line matches.\n\
             - write_file {\"path\": string, \"content\": string}: create or overwrite a whole file.\n\
             - edit_file {\"path\": string, \"old\": string, \"new\": string}: replace one exact, unique occurrence of `old`.\n",
        );
        if self.allow_commands {
            s.push_str(
                "- run_command {\"command\": string}: run a shell command in the repo root (non-interactive, timeout applies).\n",
            );
        }
        for t in &self.custom {
            let params = if t.parameters.is_null() { "{}".to_string() } else { t.parameters.to_string() };
            s.push_str(&format!("- {} {}: {}\n", t.name, params, t.description.trim()));
        }
        s.push_str("- finish {\"summary\": string}: the goal is complete and verified; summarize what changed.\n");
        s
    }

    pub async fn execute(&self, a: &Action) -> ToolResult {
        let r = match a.tool.as_str() {
            "list_files" => self.list_files(a).await,
            "read_file" => self.read_file(a).await,
            "search" => self.search(a).await,
            "write_file" => self.write_file(a).await,
            "edit_file" => self.edit_file(a).await,
            "run_command" => self.run_command(a).await,
            other => match self.custom.iter().find(|t| t.name == other) {
                Some(t) => self.custom_tool(t, a).await,
                None => Err(format!("unknown tool {other:?}; use one of the tools listed in the instructions")),
            },
        };
        match r {
            Ok(t) => t,
            Err(msg) => ToolResult::err(msg),
        }
    }

    /// Maps a model-supplied relative path into the worktree, or explains
    /// why it is refused.
    fn resolve(&self, raw: &str) -> Result<PathBuf, String> {
        let raw = raw.trim();
        let p = Path::new(if raw.is_empty() { "." } else { raw });
        let mut out = self.root.clone();
        for c in p.components() {
            match c {
                Component::Normal(part) => {
                    if part == ".git" {
                        return Err("paths inside .git are off limits".into());
                    }
                    out.push(part);
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    if out == self.root {
                        return Err(format!("{raw:?} escapes the repository"));
                    }
                    out.pop();
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(format!("use a path relative to the repository root, not {raw:?}"));
                }
            }
        }
        // Symlinks: the deepest existing ancestor must really be inside.
        let mut probe = out.as_path();
        while !probe.exists() {
            probe = probe.parent().ok_or("invalid path")?;
        }
        let real = probe.canonicalize().map_err(|e| e.to_string())?;
        if !real.starts_with(&self.root) {
            return Err(format!("{raw:?} resolves outside the repository"));
        }
        Ok(out)
    }

    fn rel<'a>(&self, p: &'a Path) -> std::borrow::Cow<'a, str> {
        p.strip_prefix(&self.root).unwrap_or(p).to_string_lossy()
    }

    async fn list_files(&self, a: &Action) -> Result<ToolResult, String> {
        let dir = self.resolve(a.str_arg("path").unwrap_or("."))?;
        let prefix = self.rel(&dir).replace('\\', "/");
        let files = GitRepo::new(&self.root).ls_files().await.map_err(|e| e.to_string())?;
        let matching: Vec<&String> = files
            .iter()
            .filter(|f| prefix.is_empty() || *f == &prefix || f.starts_with(&format!("{prefix}/")))
            .collect();
        if matching.is_empty() {
            return Ok(ToolResult::ok(format!("no files under {:?}", if prefix.is_empty() { "." } else { &prefix })));
        }
        const MAX: usize = 400;
        let mut out: String = matching.iter().take(MAX).map(|f| format!("{f}\n")).collect();
        if matching.len() > MAX {
            out.push_str(&format!("[{} more files; list a subdirectory]\n", matching.len() - MAX));
        }
        Ok(ToolResult::ok(out))
    }

    async fn read_file(&self, a: &Action) -> Result<ToolResult, String> {
        let path_arg = a.str_arg("path").ok_or("read_file needs \"path\"")?;
        let path = self.resolve(path_arg)?;
        let bytes = tokio::fs::read(&path).await.map_err(|e| format!("cannot read {path_arg}: {e}"))?;
        if bytes.iter().take(8192).any(|b| *b == 0) {
            return Err(format!("{path_arg} looks like a binary file ({} bytes)", bytes.len()));
        }
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len() as i64;
        if total == 0 {
            return Ok(ToolResult::ok(format!("{path_arg} is empty")));
        }
        let start = a.int_arg("start_line").unwrap_or(1).clamp(1, total);
        let end = a.int_arg("end_line").unwrap_or(start + READ_DEFAULT_LINES - 1).clamp(start, total);
        let mut out = String::new();
        for (i, line) in lines[(start - 1) as usize..end as usize].iter().enumerate() {
            out.push_str(&format!("{:>5}| {line}\n", start + i as i64));
        }
        if start > 1 || end < total {
            out.push_str(&format!("[lines {start}-{end} of {total}; pass start_line/end_line to read more]\n"));
        }
        Ok(ToolResult::ok(out))
    }

    async fn search(&self, a: &Action) -> Result<ToolResult, String> {
        let pattern = a.str_arg("pattern").or_else(|| a.str_arg("query")).ok_or("search needs \"pattern\"")?;
        let dir = self.resolve(a.str_arg("path").unwrap_or("."))?;
        let rel = self.rel(&dir);
        let rel = if rel.is_empty() { "." } else { &rel };
        let hits = GitRepo::new(&self.root).grep(pattern, rel).await.map_err(|e| format!("search failed: {e}"))?;
        if hits.is_empty() {
            return Ok(ToolResult::ok(format!("no matches for {pattern:?}")));
        }
        Ok(ToolResult::ok(clip(&hits, 20_000)))
    }

    async fn write_file(&self, a: &Action) -> Result<ToolResult, String> {
        let path_arg = a.str_arg("path").ok_or("write_file needs \"path\"")?;
        let content = a.str_arg("content").ok_or("write_file needs \"content\" (a string)")?;
        let path = self.resolve(path_arg)?;
        if path.is_dir() {
            return Err(format!("{path_arg} is a directory"));
        }
        let existed = path.exists();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
        }
        tokio::fs::write(&path, content).await.map_err(|e| format!("cannot write {path_arg}: {e}"))?;
        let lines = content.lines().count();
        Ok(ToolResult::ok(format!(
            "{} {path_arg} ({lines} lines, {} bytes)",
            if existed { "overwrote" } else { "created" },
            content.len()
        )))
    }

    async fn edit_file(&self, a: &Action) -> Result<ToolResult, String> {
        let path_arg = a.str_arg("path").ok_or("edit_file needs \"path\"")?;
        let old = a.str_arg("old").or_else(|| a.str_arg("old_text")).or_else(|| a.str_arg("search"));
        let new = a.str_arg("new").or_else(|| a.str_arg("new_text")).or_else(|| a.str_arg("replace"));
        let (old, new) = match (old, new) {
            (Some(o), Some(n)) if !o.is_empty() => (o, n),
            _ => return Err("edit_file needs non-empty \"old\" and a \"new\" string".into()),
        };
        let path = self.resolve(path_arg)?;
        let text = tokio::fs::read_to_string(&path).await.map_err(|e| format!("cannot read {path_arg}: {e}"))?;
        let count = text.matches(old).count();
        if count == 0 {
            let hint = if text.contains(old.trim()) {
                " (it matches after trimming whitespace: copy the exact indentation)"
            } else {
                " (read_file the current content and copy the text exactly)"
            };
            return Err(format!("\"old\" text not found in {path_arg}{hint}"));
        }
        if count > 1 {
            return Err(format!("\"old\" text occurs {count} times in {path_arg}; include more surrounding lines"));
        }
        tokio::fs::write(&path, text.replacen(old, new, 1)).await.map_err(|e| e.to_string())?;
        Ok(ToolResult::ok(format!("edited {path_arg}")))
    }

    async fn run_command(&self, a: &Action) -> Result<ToolResult, String> {
        if !self.allow_commands {
            return Err("run_command is disabled in this workspace's settings".into());
        }
        let command = a.str_arg("command").or_else(|| a.str_arg("cmd")).ok_or("run_command needs \"command\"")?;
        let out = run_shell(&self.root, command, &[], self.timeout).await;
        Ok(out.into_result(command))
    }

    async fn custom_tool(&self, t: &ToolSpec, a: &Action) -> Result<ToolResult, String> {
        let env: Vec<(String, String)> = a
            .args
            .iter()
            .map(|(k, v)| {
                let key: String =
                    k.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' }).collect();
                let val = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (format!("ORCHOPORK_ARG_{key}"), val)
            })
            .collect();
        let out = run_shell(&self.root, &t.command, &env, self.timeout).await;
        Ok(out.into_result(&t.name))
    }
}

pub struct ShellOutput {
    pub exit_code: Option<i32>,
    pub output: String,
    pub timed_out: bool,
}

impl ShellOutput {
    pub fn success(&self) -> bool {
        self.exit_code == Some(0)
    }

    fn into_result(self, label: &str) -> ToolResult {
        let status = match (self.timed_out, self.exit_code) {
            (true, _) => "timed out".to_string(),
            (_, Some(c)) => format!("exit code {c}"),
            (_, None) => "killed by a signal".to_string(),
        };
        let body = if self.output.trim().is_empty() { "(no output)".to_string() } else { self.output };
        let ok = self.exit_code == Some(0);
        ToolResult { ok, output: format!("$ {label}\n[{status}]\n{}", clip(&body, MAX_STORED_OUTPUT)) }
    }
}

/// Runs `command` through the platform shell in `cwd`, with stdin closed
/// and a hard timeout. On unix the command gets its own process group so a
/// timeout kills everything it spawned (`cargo test` → test binaries), not
/// just the shell.
pub async fn run_shell(cwd: &Path, command: &str, env: &[(String, String)], timeout: Duration) -> ShellOutput {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    };
    cmd.current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .env("ORCHOPORK", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    #[cfg(unix)]
    cmd.process_group(0);

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return ShellOutput {
                exit_code: None,
                output: format!("failed to start the shell: {e}"),
                timed_out: false,
            };
        }
    };
    let pid = child.id();
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let output = match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
                (_, true) => stdout.into_owned(),
                (true, false) => stderr.into_owned(),
                (false, false) => format!("{stdout}\n[stderr]\n{stderr}"),
            };
            ShellOutput { exit_code: out.status.code(), output, timed_out: false }
        }
        Ok(Err(e)) => {
            ShellOutput { exit_code: None, output: format!("failed waiting for the command: {e}"), timed_out: false }
        }
        Err(_) => {
            #[cfg(unix)]
            if let Some(pid) = pid {
                let _ = std::process::Command::new("kill")
                    .args(["-KILL", &format!("-{pid}")])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }
            #[cfg(not(unix))]
            let _ = pid;
            ShellOutput {
                exit_code: None,
                output: format!("command did not finish within {}s and was killed", timeout.as_secs()),
                timed_out: true,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn act(tool: &str, args: Value) -> Action {
        Action { tool: tool.into(), args: args.as_object().unwrap().clone() }
    }

    async fn toolbox() -> (tempfile::TempDir, ToolBox) {
        let d = tempfile::tempdir().unwrap();
        GitRepo::new(d.path()).init().await.unwrap();
        let limits = Limits { command_timeout_secs: 2, ..Limits::default() };
        let tb = ToolBox::new(d.path(), &limits, vec![]).unwrap();
        (d, tb)
    }

    #[tokio::test]
    async fn write_read_edit_search_round_trip() {
        let (_d, tb) = toolbox().await;
        let w = tb.execute(&act("write_file", json!({"path": "src/a.rs", "content": "fn a() {}\nfn b() {}\n"}))).await;
        assert!(w.ok, "{}", w.output);
        let r = tb.execute(&act("read_file", json!({"path": "src/a.rs", "start_line": 2}))).await;
        assert!(r.output.contains("    2| fn b() {}"), "{}", r.output);
        let e = tb.execute(&act("edit_file", json!({"path": "src/a.rs", "old": "fn b()", "new": "fn c()"}))).await;
        assert!(e.ok, "{}", e.output);
        let s = tb.execute(&act("search", json!({"pattern": "fn c"}))).await;
        assert!(s.output.contains("src/a.rs:2:"), "{}", s.output);
        let l = tb.execute(&act("list_files", json!({"path": "src"}))).await;
        assert_eq!(l.output.trim(), "src/a.rs");
        let dup = tb.execute(&act("edit_file", json!({"path": "src/a.rs", "old": "fn ", "new": "x"}))).await;
        assert!(!dup.ok && dup.output.contains("2 times"));
    }

    #[tokio::test]
    async fn paths_cannot_escape_the_worktree() {
        let (d, tb) = toolbox().await;
        for p in ["../outside.txt", "/etc/passwd", "a/../../x", ".git/config"] {
            let r = tb.execute(&act("write_file", json!({"path": p, "content": "x"}))).await;
            assert!(!r.ok, "{p} should be refused");
        }
        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            std::os::unix::fs::symlink(outside.path(), d.path().join("link")).unwrap();
            let r = tb.execute(&act("write_file", json!({"path": "link/x.txt", "content": "x"}))).await;
            assert!(!r.ok && !outside.path().join("x.txt").exists());
        }
        assert!(tb.execute(&act("read_file", json!({"path": "sub/../ok.txt"}))).await.output.contains("cannot read"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn commands_report_exit_codes_and_time_out() {
        let (_d, tb) = toolbox().await;
        let ok = tb.execute(&act("run_command", json!({"command": "echo hi"}))).await;
        assert!(ok.ok && ok.output.contains("exit code 0") && ok.output.contains("hi"));
        let bad = tb.execute(&act("run_command", json!({"command": "echo oops >&2; exit 3"}))).await;
        assert!(!bad.ok && bad.output.contains("exit code 3") && bad.output.contains("oops"));
        let slow = tb.execute(&act("run_command", json!({"command": "sleep 30"}))).await;
        assert!(!slow.ok && slow.output.contains("timed out"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn custom_tools_get_args_as_env_not_interpolated() {
        let d = tempfile::tempdir().unwrap();
        let spec = ToolSpec {
            name: "echo_arg".into(),
            description: "d".into(),
            parameters: Value::Null,
            command: "printf '%s' \"$ORCHOPORK_ARG_TEXT\"".into(),
        };
        let tb = ToolBox::new(d.path(), &Limits::default(), vec![spec]).unwrap();
        let r = tb.execute(&act("echo_arg", json!({"text": "$(touch pwned); `id`"}))).await;
        assert!(r.ok && r.output.contains("$(touch pwned); `id`"), "{}", r.output);
        assert!(!d.path().join("pwned").exists());
        assert!(tb.describe().contains("echo_arg"));
    }
}
