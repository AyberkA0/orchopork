//! External agents: coding agents orchopork drives as subprocesses, usable
//! anywhere a model is (`acp:<id>` in a `ModelRef`).
//!
//! Two ways to talk to one:
//!
//! - **ACP** ([`acp`]): the Agent Client Protocol, a JSON-RPC session over
//!   stdio. orchopork answers the agent's permission and file requests
//!   itself, so writes are confined to the worktree.
//! - **Headless CLI** ([`cli`]): the tool's own one-shot mode
//!   (`claude -p`, `codex exec`, `gemini -p`, `cursor-agent -p`, `aider
//!   --message-file`, …). Nothing to install beyond the CLI you already
//!   use, and the tool runs exactly as it does in your terminal: your
//!   login, settings, MCP servers, plugins and memory files. Its own
//!   permission system applies; orchopork maps `allow_commands` onto it.
//!
//! Either way each agent keeps its own login/subscription; nothing is
//! impersonated, and its cost is not counted against the budget (the
//! tokens and cost it reports are recorded for the metrics).

pub mod acp;
pub mod cli;
mod presets;
pub mod trust;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::process::{Child, Command};

pub use presets::{Preset, Setup, catalog, default_agents, install, setup_for};

use crate::error::{Error, Result};

/// How orchopork talks to an agent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    #[default]
    Acp,
    Cli,
}

/// Which CLI a headless agent is: decides the flags orchopork adds
/// (output format, permissions, model, isolation) and how its output is
/// read. `generic` adds nothing and takes stdout as the report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dialect {
    #[default]
    Generic,
    Claude,
    Codex,
    Gemini,
    Qwen,
    Cursor,
    Copilot,
    Aider,
    Antigravity,
}

fn yes() -> bool {
    true
}
fn is_true(b: &bool) -> bool {
    *b
}
fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    *t == T::default()
}

/// An agent orchopork can launch. `id` is what a `ModelRef` names
/// (`acp:claude-code`). For CLI agents, `args` may contain `{prompt}` (the
/// task as one argument) or `{prompt_file}` (a temporary file holding it);
/// without either the task goes to stdin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalAgent {
    pub id: String,
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub protocol: Protocol,
    #[serde(default, skip_serializing_if = "is_default")]
    pub dialect: Dialect,
    /// Extra environment variables for the agent process.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Run with the tool's own user configuration (settings, MCP servers,
    /// plugins, memory files). Off: orchopork asks the CLI to ignore it
    /// where the CLI has a flag for that (Claude Code does).
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub inherit_config: bool,
}

impl ExternalAgent {
    pub fn new(id: &str, name: &str, command: &str, args: &[&str]) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            command: command.into(),
            args: args.iter().map(|s| s.to_string()).collect(),
            protocol: Protocol::Acp,
            dialect: Dialect::Generic,
            env: BTreeMap::new(),
            inherit_config: true,
        }
    }

    pub fn cli(mut self, dialect: Dialect) -> Self {
        self.protocol = Protocol::Cli;
        self.dialect = dialect;
        self
    }

    /// "ACP" or "CLI", for labels.
    pub fn kind(&self) -> &'static str {
        match self.protocol {
            Protocol::Acp => "ACP",
            Protocol::Cli => "CLI",
        }
    }
}

/// What the agent may do without asking a human.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// Allow shell commands.
    pub allow_execute: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub status: String,
}

/// Tokens and cost as the agent reported them (CLI agents only; ACP does
/// not carry usage). Nothing here is charged to the budget.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Usage {
    /// All input tokens, cached ones included.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// What the tool says the turn cost (Claude Code reports this even on a
    /// subscription, where it is notional).
    pub reported_cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    /// The agent's final message text.
    pub text: String,
    pub stop_reason: String,
    pub tool_calls: Vec<ToolCall>,
    pub cancelled: bool,
    /// Options applied to the session, by display name
    /// (e.g. "Model" → "Opus 5.5", "Effort" → "High").
    pub settings: BTreeMap<String, String>,
    pub usage: Usage,
}

/// Human-readable progress, streamed to the UI while a turn runs.
pub enum Update<'a> {
    Tool(&'a ToolCall),
    Plan(String),
    Thinking,
    Text,
    /// Something orchopork did on the agent's behalf (e.g. trusting the
    /// worktrees folder), worth showing once.
    Note(String),
}

/// Runs one task with `agent` in `cwd`, applying `options` (e.g.
/// `model=opus`, `effort=high`).
#[allow(clippy::too_many_arguments)]
pub async fn run(
    agent: &ExternalAgent,
    cwd: &Path,
    prompt: &str,
    options: &BTreeMap<String, String>,
    policy: Policy,
    timeout: Duration,
    cancel: &AtomicBool,
    mut on_update: impl FnMut(Update) + Send,
) -> Result<Outcome> {
    if let Some(note) = trust::ensure(agent, cwd) {
        on_update(Update::Note(note));
    }
    match agent.protocol {
        Protocol::Acp => acp::run(agent, cwd, prompt, options, policy, timeout, cancel, on_update).await,
        Protocol::Cli => cli::run(agent, cwd, prompt, options, policy, timeout, cancel, on_update).await,
    }
}

/// What the agent lets you choose (model, effort, …): asked over ACP, or
/// the known choices of a CLI.
pub async fn discover(agent: &ExternalAgent, cwd: &Path) -> Result<Vec<Value>> {
    match agent.protocol {
        Protocol::Acp => acp::discover(agent, cwd).await,
        Protocol::Cli => Ok(cli::choices(agent.dialect)),
    }
}

/// Checks that the agent starts: the ACP handshake, or `--version`.
pub async fn probe(agent: &ExternalAgent, cwd: &Path) -> Result<Value> {
    match agent.protocol {
        Protocol::Acp => acp::probe(agent, cwd).await,
        Protocol::Cli => cli::probe(agent, cwd).await,
    }
}

/// Folders where agent CLIs are commonly installed but which are often
/// missing from PATH when orchopork is started from a GUI or a service.
fn extra_dirs() -> Vec<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    let mut dirs = Vec::new();
    if let Some(h) = &home {
        for rel in [".local/bin", ".claude/local", ".npm-global/bin", ".bun/bin", ".volta/bin", ".cargo/bin", "go/bin"]
        {
            dirs.push(h.join(rel));
        }
    }
    if cfg!(windows) {
        if let Some(app) = std::env::var_os("APPDATA") {
            dirs.push(PathBuf::from(app).join("npm"));
        }
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            dirs.push(PathBuf::from(local).join("Programs"));
        }
    } else {
        dirs.extend(["/opt/homebrew/bin", "/usr/local/bin"].map(PathBuf::from));
    }
    dirs
}

/// Looks `command` up on PATH (with PATHEXT on Windows), then in the usual
/// install folders of agent CLIs.
pub fn resolve_command(command: &str) -> Option<PathBuf> {
    let p = Path::new(command);
    if p.components().count() > 1 {
        return p.is_file().then(|| p.to_path_buf());
    }
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into())
            .split(';')
            .map(|e| e.to_ascii_lowercase())
            .chain(std::iter::once(String::new()))
            .collect()
    } else {
        vec![String::new()]
    };
    let path = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default();
    path.into_iter()
        .chain(extra_dirs())
        .find_map(|dir| exts.iter().map(|e| dir.join(format!("{command}{e}"))).find(|c| c.is_file()))
}

/// Builds the command for `agent` with `args` (its own args, already
/// expanded), stdio piped. `direct` spawns a Windows .cmd/.bat shim
/// without wrapping it in `cmd /C`, so the standard library quotes the
/// arguments (needed when the prompt itself is an argument).
pub(crate) fn command(agent: &ExternalAgent, cwd: &Path, args: &[String], direct: bool) -> Result<Command> {
    let exe = resolve_command(&agent.command)
        .ok_or_else(|| Error::Provider(format!("{}: `{}` was not found on PATH", agent.name, agent.command)))?;
    let is_script = exe
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("cmd") || e.eq_ignore_ascii_case("bat"));
    // Windows cannot exec .cmd/.bat shims (npx, npm-installed CLIs) directly.
    let mut cmd = if is_script && !direct {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(&exe);
        c
    } else {
        Command::new(&exe)
    };
    // A CLI found outside PATH (e.g. ~/.local/bin) may need its siblings
    // (node next to an npm shim) to start.
    if let Some(dir) = exe.parent() {
        let mut paths = vec![dir.to_path_buf()];
        paths.extend(
            std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect::<Vec<_>>()).unwrap_or_default(),
        );
        if let Ok(joined) = std::env::join_paths(paths) {
            cmd.env("PATH", joined);
        }
    }
    // Claude Code refuses to start inside another Claude Code session,
    // which is what orchopork looks like when it was itself started from one.
    cmd.env_remove("CLAUDECODE");
    cmd.args(args)
        .envs(&agent.env)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Own process group, so stopping the agent stops what it spawned.
    #[cfg(unix)]
    cmd.process_group(0);
    Ok(cmd)
}

pub(crate) fn spawn(agent: &ExternalAgent, cwd: &Path, args: &[String], direct: bool) -> Result<Child> {
    command(agent, cwd, args, direct)?
        .spawn()
        .map_err(|e| Error::Provider(format!("{}: failed to start: {e}", agent.name)))
}

/// Stops the agent and everything it started: npx and npm shims run the
/// real agent as a grandchild, which killing the direct child would leave
/// running.
pub(crate) fn kill_tree(child: &mut Child) {
    if let Some(pid) = child.id() {
        #[cfg(windows)]
        let _ = std::process::Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        #[cfg(unix)]
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &format!("-{pid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.start_kill();
}

/// Collects the tail of a child's stderr for error messages (login
/// prompts etc.).
pub(crate) fn stderr_tail(child: &mut Child) -> tokio::task::JoinHandle<String> {
    use tokio::io::AsyncReadExt;
    let err = child.stderr.take();
    tokio::spawn(async move {
        let Some(mut err) = err else { return String::new() };
        let mut buf = Vec::new();
        let _ = err.read_to_end(&mut buf).await;
        let s = strip_ansi(&String::from_utf8_lossy(&buf));
        let n = s.chars().count();
        s.chars().skip(n.saturating_sub(1500)).collect()
    })
}

/// Drops terminal color and cursor codes, which CLIs print even to a pipe
/// and which would show up as garbage in the UI.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match it.next() {
            // CSI: parameters, then one final byte in @..~
            Some('[') => {
                for d in it.by_ref() {
                    if ('@'..='~').contains(&d) {
                        break;
                    }
                }
            }
            // OSC: up to BEL or ESC \
            Some(']') => {
                while let Some(d) = it.next() {
                    if d == '\u{7}' || (d == '\u{1b}' && it.next_if_eq(&'\\').is_some()) {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_codes_are_stripped() {
        assert_eq!(strip_ansi("\u{1b}[1;31mError\u{1b}[0m: x \u{1b}]8;;http://a\u{7}link"), "Error: x link");
    }

    #[test]
    fn old_configs_without_protocol_stay_acp() {
        let a: ExternalAgent = serde_yaml::from_str("{id: x, name: X, command: npx, args: [a]}").unwrap();
        assert_eq!((a.protocol, a.dialect, a.inherit_config), (Protocol::Acp, Dialect::Generic, true));
        let text = serde_yaml::to_string(&a).unwrap();
        assert!(!text.contains("protocol") && !text.contains("inherit_config"), "defaults are not written: {text}");
        let c = ExternalAgent::new("c", "C", "claude", &[]).cli(Dialect::Claude);
        let back: ExternalAgent = serde_yaml::from_str(&serde_yaml::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }
}
