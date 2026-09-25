//! A minimal Agent Client Protocol (ACP) client.
//!
//! ACP lets an editor drive a coding agent that runs as a subprocess,
//! speaking newline-delimited JSON-RPC 2.0 over stdio. Claude Code (through
//! Zed's adapter), Gemini CLI and Codex (through an adapter) all speak it,
//! so one client turns every installed agent into an orchestra member.
//! Each agent keeps its own login/subscription; nothing is impersonated.
//!
//! One turn = spawn → `initialize` → `session/new` (cwd = the run's
//! worktree) → `session/prompt`, streaming `session/update`s until the
//! prompt returns a stop reason. The agent's requests are answered here:
//! `session/request_permission` by policy, `fs/read_text_file` and
//! `fs/write_text_file` only inside the worktree.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::error::{Error, Result};

const PROTOCOL_VERSION: u64 = 1;

/// An agent launchable over ACP. `id` is what a `ModelRef` names
/// (`acp:claude-code`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalAgent {
    pub id: String,
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// Presets for the agents people usually have installed. Each uses the
/// user's existing login of that tool; edit command/args in settings if an
/// adapter is installed differently.
pub fn default_agents() -> Vec<ExternalAgent> {
    let a = |id: &str, name: &str, command: &str, args: &[&str]| ExternalAgent {
        id: id.into(),
        name: name.into(),
        command: command.into(),
        args: args.iter().map(|s| s.to_string()).collect(),
    };
    vec![
        a("claude-code", "Claude Code", "npx", &["-y", "@agentclientprotocol/claude-agent-acp"]),
        a("gemini-cli", "Gemini CLI", "gemini", &["--acp"]),
        a("codex", "Codex", "npx", &["-y", "@agentclientprotocol/codex-acp"]),
    ]
}

/// What the agent may do without asking a human.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// Allow tool calls of kind `execute` (shell commands).
    pub allow_execute: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    /// The agent's final message text.
    pub text: String,
    pub stop_reason: String,
    pub tool_calls: Vec<ToolCall>,
    pub cancelled: bool,
}

/// Human-readable progress, streamed to the UI while a turn runs.
pub enum Update<'a> {
    Tool(&'a ToolCall),
    Plan(String),
    Thinking,
    Text,
}

/// Looks `command` up on PATH (with PATHEXT on Windows).
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
    std::env::split_paths(&std::env::var_os("PATH")?)
        .find_map(|dir| exts.iter().map(|e| dir.join(format!("{command}{e}"))).find(|c| c.is_file()))
}

fn spawn(agent: &ExternalAgent, cwd: &Path) -> Result<Child> {
    let exe = resolve_command(&agent.command)
        .ok_or_else(|| Error::Provider(format!("{}: `{}` was not found on PATH", agent.name, agent.command)))?;
    let is_script = exe
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("cmd") || e.eq_ignore_ascii_case("bat"));
    // Windows cannot exec .cmd/.bat shims (npx, npm-installed CLIs) directly.
    let mut cmd = if is_script {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(&exe);
        c
    } else {
        Command::new(&exe)
    };
    cmd.args(&agent.args)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd.spawn().map_err(|e| Error::Provider(format!("{}: failed to start: {e}", agent.name)))
}

struct Conn<'a> {
    agent: &'a ExternalAgent,
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    stderr: tokio::task::JoinHandle<String>,
    next_id: u64,
    root: PathBuf,
    policy: Policy,
    session: Option<String>,
    text: String,
    tools: Vec<ToolCall>,
    /// When `session/cancel` was sent (the agent then gets 20 s to stop).
    cancel_sent: Option<Instant>,
}

impl<'a> Conn<'a> {
    fn open(agent: &'a ExternalAgent, cwd: &Path, policy: Policy) -> Result<Self> {
        let mut child = spawn(agent, cwd)?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = child.stdout.take().expect("piped");
        let mut err = child.stderr.take().expect("piped");
        // Keep the tail of stderr for error messages (login prompts etc.).
        let stderr = tokio::spawn(async move {
            let mut buf = Vec::new();
            let _ = err.read_to_end(&mut buf).await;
            let s = String::from_utf8_lossy(&buf).into_owned();
            let n = s.chars().count();
            s.chars().skip(n.saturating_sub(1500)).collect()
        });
        Ok(Self {
            agent,
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
            stderr,
            next_id: 0,
            root: crate::fsutil::canonical(cwd)?,
            policy,
            session: None,
            text: String::new(),
            tools: Vec::new(),
            cancel_sent: None,
        })
    }

    async fn send(&mut self, v: &Value) -> Result<()> {
        let mut line = serde_json::to_vec(v)?;
        line.push(b'\n');
        self.stdin.write_all(&line).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    /// Sends a request and serves the agent's own requests/notifications
    /// until the matching response arrives.
    async fn request(
        &mut self,
        method: &str,
        params: Value,
        deadline: Instant,
        cancel: &AtomicBool,
        on_update: &mut (dyn FnMut(Update) + Send),
    ) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })).await?;
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        loop {
            tokio::select! {
                line = self.lines.next_line() => {
                    let Some(line) = line? else {
                        let _ = self.child.wait().await;
                        let err = (&mut self.stderr).await.unwrap_or_default();
                        return Err(Error::Provider(format!(
                            "{} exited during {method}{}", self.agent.name,
                            if err.trim().is_empty() { String::new() } else { format!(": {}", err.trim()) }
                        )));
                    };
                    let Ok(msg) = serde_json::from_str::<Value>(line.trim()) else { continue };
                    if msg.get("method").is_some() {
                        self.incoming(msg, on_update).await?;
                    } else if msg["id"].as_u64() == Some(id) {
                        if let Some(e) = msg.get("error") {
                            let text = e["message"].as_str().unwrap_or("error");
                            let data = e.get("data").map(|d| format!(" ({d})")).unwrap_or_default();
                            return Err(Error::Provider(format!("{} {method}: {text}{data}", self.agent.name)));
                        }
                        return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
                    }
                }
                _ = tick.tick() => {
                    if cancel.load(Ordering::SeqCst) && self.cancel_sent.is_none() {
                        self.cancel_sent = Some(Instant::now());
                        match self.session.clone() {
                            Some(sid) => self.send(&json!({ "jsonrpc": "2.0", "method": "session/cancel", "params": { "sessionId": sid } })).await?,
                            None => return Err(Error::Provider("cancelled".into())),
                        }
                    }
                    if self.cancel_sent.is_some_and(|t| t.elapsed() > Duration::from_secs(20)) {
                        return Err(Error::Provider(format!("{} did not stop after cancel", self.agent.name)));
                    }
                    if Instant::now() > deadline {
                        return Err(Error::Provider(format!("{} did not finish in time", self.agent.name)));
                    }
                }
            }
        }
    }

    async fn incoming(&mut self, msg: Value, on_update: &mut (dyn FnMut(Update) + Send)) -> Result<()> {
        let method = msg["method"].as_str().unwrap_or("");
        let params = &msg["params"];
        let Some(id) = msg.get("id").cloned() else {
            if method == "session/update" {
                self.update(&params["update"], on_update);
            }
            return Ok(());
        };
        let reply = match method {
            "session/request_permission" => Ok(self.permission(params)),
            "fs/read_text_file" => self.read_file(params).await,
            "fs/write_text_file" => self.write_file(params).await,
            other => Err((-32601, format!("method not supported: {other}"))),
        };
        let resp = match reply {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => {
                json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
            }
        };
        self.send(&resp).await
    }

    fn update(&mut self, u: &Value, on_update: &mut (dyn FnMut(Update) + Send)) {
        match u["sessionUpdate"].as_str().unwrap_or("") {
            "agent_message_chunk" => {
                if let Some(t) = u["content"]["text"].as_str() {
                    self.text.push_str(t);
                    on_update(Update::Text);
                }
            }
            "agent_thought_chunk" => on_update(Update::Thinking),
            "tool_call" => {
                let tc = ToolCall {
                    id: u["toolCallId"].as_str().unwrap_or("").into(),
                    title: u["title"].as_str().unwrap_or("tool").into(),
                    kind: u["kind"].as_str().unwrap_or("other").into(),
                    status: u["status"].as_str().unwrap_or("pending").into(),
                };
                self.tools.push(tc);
                on_update(Update::Tool(self.tools.last().expect("pushed")));
            }
            "tool_call_update" => {
                let id = u["toolCallId"].as_str().unwrap_or("");
                if let Some(tc) = self.tools.iter_mut().rev().find(|t| t.id == id) {
                    if let Some(s) = u["status"].as_str() {
                        tc.status = s.into();
                    }
                    if let Some(t) = u["title"].as_str() {
                        tc.title = t.into();
                    }
                }
            }
            "plan" => {
                let entries: Vec<&str> =
                    u["entries"].as_array().into_iter().flatten().filter_map(|e| e["content"].as_str()).collect();
                if !entries.is_empty() {
                    on_update(Update::Plan(entries.join(" · ")));
                }
            }
            _ => {}
        }
    }

    /// Picks an allow option unless policy forbids this kind of call.
    fn permission(&self, p: &Value) -> Value {
        let kind = p["toolCall"]["kind"].as_str().unwrap_or("other");
        let allowed = kind != "execute" || self.policy.allow_execute;
        let options = p["options"].as_array().cloned().unwrap_or_default();
        let pick = |kinds: &[&str]| {
            options
                .iter()
                .find(|o| kinds.contains(&o["kind"].as_str().unwrap_or("")))
                .and_then(|o| o["optionId"].as_str())
        };
        let choice =
            if allowed { pick(&["allow_once", "allow_always"]) } else { pick(&["reject_once", "reject_always"]) };
        match choice {
            Some(id) => json!({ "outcome": { "outcome": "selected", "optionId": id } }),
            None => json!({ "outcome": { "outcome": "cancelled" } }),
        }
    }

    /// Absolute path inside the worktree, or an error for the agent.
    fn inside(&self, raw: &str) -> std::result::Result<PathBuf, (i64, String)> {
        let p = Path::new(raw);
        let p = if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) };
        let p = crate::fsutil::strip_verbatim(p);
        if p.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err((-32602, "path must not contain ..".into()));
        }
        let mut probe = p.as_path();
        while !probe.exists() {
            probe = probe.parent().ok_or((-32602, "invalid path".to_string()))?;
        }
        let real = crate::fsutil::canonical(probe).map_err(|e| (-32603, e.to_string()))?;
        if !real.starts_with(&self.root) || p.components().any(|c| c.as_os_str() == ".git") {
            return Err((-32602, format!("{raw} is outside the working tree")));
        }
        Ok(p)
    }

    async fn read_file(&self, p: &Value) -> std::result::Result<Value, (i64, String)> {
        let path = self.inside(p["path"].as_str().unwrap_or(""))?;
        let text = tokio::fs::read_to_string(&path).await.map_err(|e| (-32603, e.to_string()))?;
        let line = p["line"].as_u64().map(|l| l.max(1) as usize);
        let limit = p["limit"].as_u64().map(|l| l as usize);
        let content = match (line, limit) {
            (None, None) => text,
            (l, n) => {
                text.lines().skip(l.unwrap_or(1) - 1).take(n.unwrap_or(usize::MAX)).collect::<Vec<_>>().join("\n")
            }
        };
        Ok(json!({ "content": content }))
    }

    async fn write_file(&self, p: &Value) -> std::result::Result<Value, (i64, String)> {
        let path = self.inside(p["path"].as_str().unwrap_or(""))?;
        if let Some(dir) = path.parent() {
            tokio::fs::create_dir_all(dir).await.map_err(|e| (-32603, e.to_string()))?;
        }
        tokio::fs::write(&path, p["content"].as_str().unwrap_or("")).await.map_err(|e| (-32603, e.to_string()))?;
        Ok(Value::Null)
    }

    async fn handshake(
        &mut self,
        deadline: Instant,
        cancel: &AtomicBool,
        on_update: &mut (dyn FnMut(Update) + Send),
    ) -> Result<Value> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": { "fs": { "readTextFile": true, "writeTextFile": true }, "terminal": false },
            "clientInfo": { "name": "orchopork", "version": env!("CARGO_PKG_VERSION") },
        });
        self.request("initialize", params, deadline, cancel, on_update).await
    }
}

/// Runs one prompt turn with `agent` in `cwd`.
pub async fn run(
    agent: &ExternalAgent,
    cwd: &Path,
    prompt: &str,
    policy: Policy,
    timeout: Duration,
    cancel: &AtomicBool,
    mut on_update: impl FnMut(Update) + Send,
) -> Result<Outcome> {
    let deadline = Instant::now() + timeout;
    let mut c = Conn::open(agent, cwd, policy)?;
    let on: &mut (dyn FnMut(Update) + Send) = &mut on_update;
    c.handshake(deadline, cancel, on).await?;
    let cwd_str = c.root.to_string_lossy().into_owned();
    let session = c
        .request("session/new", json!({ "cwd": cwd_str, "mcpServers": [] }), deadline, cancel, on)
        .await
        .map_err(|e| match e {
            Error::Provider(m) if m.to_lowercase().contains("auth") => {
                Error::Provider(format!("{m}. Sign in to {} once in a terminal, then retry.", agent.name))
            }
            other => other,
        })?;
    let sid = session["sessionId"]
        .as_str()
        .ok_or_else(|| Error::Provider(format!("{}: session/new returned no sessionId", agent.name)))?
        .to_string();
    c.session = Some(sid.clone());
    let result = c
        .request(
            "session/prompt",
            json!({ "sessionId": sid, "prompt": [{ "type": "text", "text": prompt }] }),
            deadline + Duration::from_secs(15),
            cancel,
            on,
        )
        .await?;
    let stop_reason = result["stopReason"].as_str().unwrap_or("end_turn").to_string();
    let _ = c.child.start_kill();
    Ok(Outcome {
        cancelled: stop_reason == "cancelled",
        text: c.text.trim().to_string(),
        stop_reason,
        tool_calls: c.tools,
    })
}

/// `initialize` only: checks that the agent starts and speaks ACP.
pub async fn probe(agent: &ExternalAgent, cwd: &Path) -> Result<Value> {
    let never = AtomicBool::new(false);
    let mut c = Conn::open(agent, cwd, Policy { allow_execute: false })?;
    let r = c.handshake(Instant::now() + Duration::from_secs(90), &never, &mut |_| {}).await;
    let _ = c.child.start_kill();
    r
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A fake ACP agent in Python: asks permission for an `execute` call,
    /// writes a file through the client, and replies.
    const FAKE: &str = r#"
import json, sys
def send(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
def read(): return json.loads(sys.stdin.readline())
sid = "s1"; n = 100
while True:
    m = read()
    if m.get("method") == "initialize":
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"protocolVersion": 1, "agentCapabilities": {}}})
    elif m.get("method") == "session/new":
        send({"jsonrpc": "2.0", "id": m["id"], "result": {"sessionId": sid}})
    elif m.get("method") == "session/prompt":
        pid = m["id"]
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Run tests", "kind": "execute", "status": "pending"}}})
        send({"jsonrpc": "2.0", "id": n, "method": "session/request_permission", "params": {"sessionId": sid, "toolCall": {"toolCallId": "t1", "kind": "execute"}, "options": [{"optionId": "yes", "name": "Allow", "kind": "allow_once"}, {"optionId": "no", "name": "Reject", "kind": "reject_once"}]}})
        perm = read()
        send({"jsonrpc": "2.0", "id": n + 1, "method": "fs/write_text_file", "params": {"sessionId": sid, "path": "out.txt", "content": perm["result"]["outcome"]["optionId"]}})
        w = read()
        send({"jsonrpc": "2.0", "id": n + 2, "method": "fs/write_text_file", "params": {"sessionId": sid, "path": "/etc/nope", "content": "x"}})
        bad = read()
        send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Done: " + ("blocked" if "error" in bad else "leak")}}}})
        send({"jsonrpc": "2.0", "id": pid, "result": {"stopReason": "end_turn"}})
"#;

    #[tokio::test]
    async fn drives_a_fake_agent_through_a_full_turn() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("fake.py"), FAKE).unwrap();
        let agent = ExternalAgent {
            id: "fake".into(),
            name: "Fake".into(),
            command: "python3".into(),
            args: vec![d.path().join("fake.py").to_string_lossy().into()],
        };
        let never = AtomicBool::new(false);
        let mut seen = Vec::new();
        for allow in [true, false] {
            let out =
                run(&agent, d.path(), "hi", Policy { allow_execute: allow }, Duration::from_secs(20), &never, |u| {
                    if let Update::Tool(t) = u {
                        seen.push(t.title.clone());
                    }
                })
                .await
                .unwrap();
            assert_eq!(out.text, "Done: blocked", "writes outside the tree must be refused");
            assert_eq!(out.tool_calls.len(), 1);
            let written = std::fs::read_to_string(d.path().join("out.txt")).unwrap();
            assert_eq!(written, if allow { "yes" } else { "no" });
        }
        assert_eq!(seen, ["Run tests", "Run tests"]);
        assert!(probe(&agent, d.path()).await.is_ok());
    }

    #[tokio::test]
    async fn missing_binaries_are_reported_clearly() {
        let agent = ExternalAgent {
            id: "x".into(),
            name: "X".into(),
            command: "definitely-not-installed-xyz".into(),
            args: vec![],
        };
        let never = AtomicBool::new(false);
        let err =
            run(&agent, Path::new("."), "hi", Policy { allow_execute: false }, Duration::from_secs(5), &never, |_| {})
                .await
                .unwrap_err();
        assert!(err.to_string().contains("not found on PATH"));
    }
}
