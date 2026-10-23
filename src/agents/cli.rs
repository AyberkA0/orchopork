//! Headless CLI agents: one task per process, in the tool's own
//! non-interactive mode.
//!
//! The agent runs in the worktree exactly as it would in your terminal, so
//! it uses your login, settings, MCP servers, plugins and memory files
//! (unless `inherit_config` is off and the CLI can be told to ignore them).
//! orchopork adds the flags each [`Dialect`] needs (structured output,
//! permissions, model), sends the task on stdin (or as `{prompt}` /
//! `{prompt_file}` in the args), and reads the event stream:
//!
//! | Dialect | Launched as                                         | Output       |
//! |---------|-----------------------------------------------------|--------------|
//! | claude  | `claude -p --output-format stream-json --verbose`   | JSON lines   |
//! | codex   | `codex exec --json -`                               | JSON lines   |
//! | gemini  | `gemini --output-format stream-json` (also qwen)    | JSON lines   |
//! | cursor  | `cursor-agent -p --output-format stream-json`       | JSON lines   |
//! | copilot | `copilot -p`                                        | text         |
//! | aider   | `aider --message-file … --yes-always`               | text         |
//! | antigravity | `agy -p --input-format stream-json --output-format stream-json` | JSON lines |
//! | generic | the command as written                              | text         |
//!
//! One lenient reader handles every JSON dialect: event shapes differ
//! between tools and versions, so unknown lines are skipped rather than
//! failing the turn.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::{Dialect, ExternalAgent, Outcome, Policy, ToolCall, Update, Usage, kill_tree, spawn, stderr_tail};
use crate::error::{Error, Result};

/// Reports from text-mode agents are clipped to their tail.
const TEXT_TAIL: usize = 20_000;

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// Whether the dialect's output is JSON lines (else plain text).
fn json_output(d: Dialect) -> bool {
    matches!(
        d,
        Dialect::Claude | Dialect::Codex | Dialect::Gemini | Dialect::Qwen | Dialect::Cursor | Dialect::Antigravity
    )
}

/// The full argument list: the agent's own args (with the prompt
/// placeholders expanded) plus the dialect's flags.
pub fn build_args(
    agent: &ExternalAgent,
    options: &BTreeMap<String, String>,
    policy: Policy,
    prompt_file: Option<&Path>,
    prompt: &str,
) -> Vec<String> {
    let mut lead: Vec<String> = Vec::new();
    let mut tail: Vec<String> = Vec::new();
    let opts: BTreeMap<&str, &str> = options
        .iter()
        .filter(|(_, v)| !v.is_empty() && v.as_str() != "default")
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let mut handled: Vec<&str> = vec![];
    match agent.dialect {
        Dialect::Claude => {
            lead.extend(s(&["-p", "--output-format", "stream-json", "--verbose"]));
            // Edits inside the working directory are accepted; shell
            // commands follow `allow_commands`. Anything else (web, MCP
            // tools) follows your own Claude Code permission settings.
            tail.extend(s(&["--permission-mode", "acceptEdits"]));
            tail.extend(s(if policy.allow_execute {
                &["--allowedTools", "Bash"]
            } else {
                &["--disallowedTools", "Bash"]
            }));
            if !agent.inherit_config {
                tail.extend(s(&["--setting-sources", "project,local", "--strict-mcp-config"]));
            }
            // `model` and `effort` pass through as `--model` / `--effort`.
        }
        Dialect::Codex => {
            lead.extend(s(&["exec", "--json", "--skip-git-repo-check"]));
            // Codex has no separate switch for commands: its sandbox keeps
            // them inside the worktree (and offline) either way.
            tail.extend(s(&["--sandbox", "workspace-write"]));
            if let Some(e) = opts.get("effort") {
                tail.extend(["-c".into(), format!("model_reasoning_effort={e}")]);
                handled.push("effort");
            }
        }
        Dialect::Gemini | Dialect::Qwen => {
            lead.extend(s(&["--output-format", "stream-json"]));
            tail.extend(s(&["--approval-mode", if policy.allow_execute { "yolo" } else { "auto_edit" }]));
            // Every run gets a fresh worktree that Gemini CLI has never seen;
            // untrusted, it drops the approval mode back to "ask", which a
            // headless run cannot answer. Trust it for this session only.
            if agent.dialect == Dialect::Gemini {
                tail.push("--skip-trust".into());
            }
        }
        Dialect::Cursor => {
            lead.extend(s(&["-p", "--output-format", "stream-json"]));
            if policy.allow_execute {
                tail.push("--force".into());
            }
        }
        Dialect::Copilot => {
            lead.push("-p".into());
            tail.extend(s(&["--allow-tool", "write"]));
            tail.extend(s(if policy.allow_execute { &["--allow-all-tools"] } else { &["--deny-tool", "shell"] }));
        }
        Dialect::Aider => {
            // orchopork checkpoints the worktree itself.
            tail.extend(s(&["--yes-always", "--no-auto-commits", "--no-check-update", "--no-pretty", "--no-stream"]));
            if !policy.allow_execute {
                tail.push("--no-suggest-shell-commands".into());
            }
        }
        Dialect::Antigravity => {
            // The task goes in as one stream-json message on stdin (see
            // `stdin_payload`): no command-line length or quoting limits.
            lead.extend(s(&["-p", "--input-format", "stream-json", "--output-format", "stream-json"]));
            // Reads and writes inside the workspace are always allowed;
            // agy has no flag that allows only shell commands, so with
            // commands on every tool call is approved. With them off,
            // your own `permissions.allow` rules in agy's settings apply.
            if policy.allow_execute {
                tail.push("--dangerously-skip-permissions".into());
            }
        }
        Dialect::Generic => {}
    }
    let mut out = lead;
    for a in &agent.args {
        if a.contains("{prompt_file}")
            && let Some(f) = prompt_file
        {
            out.push(a.replace("{prompt_file}", &f.to_string_lossy()));
        } else if a.contains("{prompt}") {
            out.push(a.replace("{prompt}", prompt));
        } else {
            out.push(a.clone());
        }
    }
    out.extend(tail);
    // Codex reads the task from stdin when the prompt argument is `-`.
    if agent.dialect == Dialect::Codex && !prompt_in_args(agent) {
        out.push("-".into());
    }
    for (k, v) in opts.into_iter().filter(|(k, _)| !handled.contains(k)) {
        if agent.dialect == Dialect::Codex && k == "model" {
            out.splice(1..1, ["--model".to_string(), v.to_string()]);
        } else {
            out.extend([format!("--{k}"), v.to_string()]);
        }
    }
    out
}

/// What goes to stdin: the task as is, or wrapped as one stream-json
/// message for CLIs that only read stdin in that form.
fn stdin_payload(d: Dialect, prompt: &str) -> String {
    match d {
        Dialect::Antigravity => format!("{}\n", json!({ "event": "user", "message": { "content": prompt } })),
        _ => prompt.to_string(),
    }
}

fn prompt_in_args(agent: &ExternalAgent) -> bool {
    agent.args.iter().any(|a| a.contains("{prompt}") || a.contains("{prompt_file}"))
}

/// The choices a CLI offers, in the shape ACP discovery returns. Only
/// stable values are listed; anything else still works as `?key=value`
/// (passed on as `--key value`).
pub fn choices(d: Dialect) -> Vec<Value> {
    let sel = |id: &str, name: &str, category: &str, values: &[(&str, &str)]| {
        json!({ "id": id, "name": name, "category": category, "current": null,
                "choices": values.iter().map(|(v, n)| json!({ "value": v, "name": n })).collect::<Vec<_>>() })
    };
    let efforts = [("low", "Low"), ("medium", "Medium"), ("high", "High"), ("xhigh", "Extra high"), ("max", "Max")];
    match d {
        Dialect::Claude => vec![
            sel("model", "Model", "model", &[("opus", "Opus"), ("sonnet", "Sonnet"), ("haiku", "Haiku")]),
            sel("effort", "Effort", "thought_level", &efforts),
        ],
        Dialect::Antigravity => vec![sel("effort", "Effort", "thought_level", &efforts[..3])],
        Dialect::Codex => vec![sel(
            "effort",
            "Effort",
            "thought_level",
            &[("minimal", "Minimal"), ("low", "Low"), ("medium", "Medium"), ("high", "High")],
        )],
        _ => vec![],
    }
}

/// Checks that the CLI starts (`--version`).
pub async fn probe(agent: &ExternalAgent, cwd: &Path) -> Result<Value> {
    let mut child = spawn(agent, cwd, &s(&["--version"]), false)?;
    drop(child.stdin.take());
    let out = tokio::time::timeout(Duration::from_secs(60), child.wait_with_output())
        .await
        .map_err(|_| Error::Provider(format!("{}: `--version` did not answer within 60 s", agent.name)))??;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let text = text.trim();
    if !out.status.success() {
        return Err(Error::Provider(format!(
            "{}: {}",
            agent.name,
            if text.is_empty() { "failed to start" } else { text }
        )));
    }
    let version = text.lines().next().unwrap_or("").to_string();
    Ok(json!({ "agentInfo": { "name": agent.name, "title": agent.name, "version": version }, "protocol": "cli" }))
}

/// Runs one task.
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
    let deadline = Instant::now() + timeout;
    // The prompt file lives outside the worktree, so it never shows up in
    // the diff; it is removed when this turn ends.
    let file = agent
        .args
        .iter()
        .any(|a| a.contains("{prompt_file}"))
        .then(|| std::env::temp_dir().join(format!("orchopork-task-{}.md", uuid::Uuid::new_v4().simple())));
    struct Remove(Option<std::path::PathBuf>);
    impl Drop for Remove {
        fn drop(&mut self) {
            if let Some(f) = &self.0 {
                let _ = std::fs::remove_file(f);
            }
        }
    }
    let _cleanup = Remove(file.clone());
    if let Some(f) = &file {
        tokio::fs::write(f, prompt).await?;
    }
    let mut args = build_args(agent, options, policy, file.as_deref(), prompt);
    if agent.dialect == Dialect::Antigravity {
        // agy gives up after 5 minutes by default; orchopork's own limit rules.
        args.extend(["--print-timeout".into(), format!("{}m", timeout.as_secs().div_ceil(60).max(1))]);
    }
    let direct = agent.args.iter().any(|a| a.contains("{prompt}"));
    let mut child = spawn(agent, cwd, &args, direct)?;
    let stderr = stderr_tail(&mut child);
    let mut stdin = child.stdin.take().expect("piped");
    let stdout = child.stdout.take().expect("piped");
    let feed = !prompt_in_args(agent);
    let body = stdin_payload(agent.dialect, prompt);
    // Written concurrently with reading, so a large prompt cannot deadlock
    // against a full stdout pipe.
    let writer = tokio::spawn(async move {
        if feed {
            let _ = stdin.write_all(body.as_bytes()).await;
        }
        let _ = stdin.shutdown().await;
    });

    let mut reader = Stream::new(json_output(agent.dialect));
    let mut lines = BufReader::new(stdout).lines();
    let mut tick = tokio::time::interval(Duration::from_millis(250));
    let mut cancelled = false;
    loop {
        tokio::select! {
            line = lines.next_line() => match line {
                Ok(Some(l)) => {
                    reader.line(&l, &mut on_update);
                    // agy's `result` closes the turn; it may keep waiting
                    // for more stream input, so stop reading there.
                    if reader.ended && agent.dialect == Dialect::Antigravity {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => { kill_tree(&mut child); return Err(e.into()) }
            },
            _ = tick.tick() => {
                if cancel.load(Ordering::SeqCst) {
                    cancelled = true;
                    break;
                }
                if Instant::now() > deadline {
                    kill_tree(&mut child);
                    return Err(Error::Provider(format!("{} did not finish in time", agent.name)));
                }
            }
        }
    }
    writer.abort();
    if cancelled {
        kill_tree(&mut child);
        let _ = child.wait().await;
        return Ok(reader.finish("cancelled", true));
    }
    let grace = Duration::from_secs(if reader.ended { 3 } else { 10 });
    let status = match tokio::time::timeout(grace, child.wait()).await {
        Ok(s) => s?,
        Err(_) => {
            // Finished its turn but did not exit (only after a `result`).
            kill_tree(&mut child);
            child.wait().await?
        }
    };
    let status_ok = status.success() || reader.ended;
    let err = stderr.await.unwrap_or_default();
    let err = err.trim();
    let failed = |why: String| {
        let low = why.to_lowercase();
        let hint = if low.contains("login") || low.contains("auth") {
            format!(". Sign in to {} once in a terminal, then retry.", agent.name)
        } else {
            String::new()
        };
        Err(Error::Provider(format!("{}: {why}{hint}", agent.name)))
    };
    if let Some(e) = reader.error.clone() {
        return failed(e);
    }
    if !status_ok && reader.final_text().is_empty() {
        return failed(if err.is_empty() { format!("exited with {status}") } else { err.to_string() });
    }
    Ok(reader.finish(if status_ok { "end_turn" } else { "error" }, false))
}

/// Reads an agent's stdout, line by line.
pub struct Stream {
    json: bool,
    /// Every assistant text, in order.
    all: String,
    /// Assistant text since the last tool call: the final message when the
    /// tool sends no explicit result.
    since_tool: String,
    /// An explicit final result (Claude/Cursor `result`, Codex's last
    /// agent message).
    result: Option<String>,
    tools: Vec<ToolCall>,
    usage: Usage,
    settings: BTreeMap<String, String>,
    /// A failure the agent reported in-band.
    error: Option<String>,
    /// A final `result` event was read.
    pub ended: bool,
}

fn tool_kind(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if ["bash", "shell", "command", "exec", "terminal"].iter().any(|k| n.contains(k)) {
        "execute"
    } else if ["edit", "write", "replace", "patch", "file_change", "create"].iter().any(|k| n.contains(k)) {
        "edit"
    } else if ["grep", "glob", "search", "find", "ls"].iter().any(|k| n.contains(k)) {
        "search"
    } else if n.contains("read") || n.contains("view") || n.contains("cat") {
        "read"
    } else if n.contains("fetch") || n.contains("web") {
        "fetch"
    } else if n.contains("think") || n.contains("reason") || n.contains("todo") || n.contains("plan") {
        "think"
    } else {
        "other"
    }
}

/// A short human title for a tool call: its name plus the most telling
/// argument (path, command, pattern).
fn tool_title(name: &str, input: &Value) -> String {
    let arg = ["command", "file_path", "path", "pattern", "url", "query", "description"]
        .iter()
        .find_map(|k| input.get(*k).and_then(Value::as_str))
        .map(|a| a.lines().next().unwrap_or("").chars().take(80).collect::<String>());
    match arg {
        Some(a) if !a.is_empty() => format!("{name} {a}"),
        _ => name.to_string(),
    }
}

fn n(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}

impl Stream {
    pub fn new(json: bool) -> Self {
        Self {
            json,
            all: String::new(),
            since_tool: String::new(),
            result: None,
            tools: Vec::new(),
            usage: Usage::default(),
            settings: BTreeMap::new(),
            error: None,
            ended: false,
        }
    }

    fn text(&mut self, t: &str, on: &mut dyn FnMut(Update)) {
        if t.is_empty() {
            return;
        }
        self.all.push_str(t);
        self.since_tool.push_str(t);
        on(Update::Text);
    }

    fn tool(&mut self, id: &str, title: String, kind: &str, status: &str, on: &mut dyn FnMut(Update)) {
        if let Some(t) = self.tools.iter_mut().rev().find(|t| !id.is_empty() && t.id == id) {
            t.status = status.into();
            return;
        }
        self.since_tool.clear();
        self.tools.push(ToolCall { id: id.into(), title, kind: kind.into(), status: status.into() });
        on(Update::Tool(self.tools.last().expect("pushed")));
    }

    fn finish_tool(&mut self, id: &str, ok: bool) {
        if let Some(t) = self.tools.iter_mut().rev().find(|t| t.id == id) {
            t.status = if ok { "completed" } else { "failed" }.into();
        }
    }

    pub fn line(&mut self, line: &str, on: &mut dyn FnMut(Update)) {
        if !self.json {
            self.all.push_str(line);
            self.all.push('\n');
            on(Update::Text);
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { return };
        // Antigravity names its events `event`, with the payload under the
        // same key.
        if let Some(ev) = v["event"].as_str() {
            return self.antigravity(ev, &v[ev], on);
        }
        let ty = v["type"].as_str().unwrap_or("");
        match ty {
            // ---- Claude Code and Cursor (stream-json) ----
            "system" if v["subtype"] == "init" => {
                if let Some(m) = v["model"].as_str() {
                    self.settings.insert("Model".into(), m.into());
                }
            }
            "assistant" => {
                for c in v["message"]["content"].as_array().into_iter().flatten() {
                    match c["type"].as_str() {
                        Some("text") => self.text(c["text"].as_str().unwrap_or(""), on),
                        Some("thinking") => on(Update::Thinking),
                        Some("tool_use") => {
                            let name = c["name"].as_str().unwrap_or("tool");
                            let kind = tool_kind(name);
                            self.tool(
                                c["id"].as_str().unwrap_or(""),
                                tool_title(name, &c["input"]),
                                kind,
                                "pending",
                                on,
                            );
                        }
                        _ => {}
                    }
                }
            }
            "user" => {
                for c in v["message"]["content"].as_array().into_iter().flatten() {
                    if c["type"] == "tool_result" {
                        self.finish_tool(c["tool_use_id"].as_str().unwrap_or(""), c["is_error"] != true);
                    }
                }
            }
            // Cursor: {"type":"tool_call","subtype":"started","call_id":…,"tool_call":{"readToolCall":{"args":{…}}}}
            "tool_call" => {
                let id = v["call_id"].as_str().unwrap_or("");
                let (name, args) = v["tool_call"]
                    .as_object()
                    .and_then(|o| o.iter().next())
                    .map(|(k, x)| (k.trim_end_matches("ToolCall").to_string(), x["args"].clone()))
                    .unwrap_or_else(|| ("tool".into(), Value::Null));
                if v["subtype"] == "completed" {
                    self.finish_tool(id, true);
                } else {
                    let kind = tool_kind(&name);
                    self.tool(id, tool_title(&name, &args), kind, "pending", on);
                }
            }
            "result" => {
                self.ended = true;
                if let Some(r) = v["result"].as_str() {
                    self.result = Some(r.to_string());
                }
                if v["is_error"] == true {
                    let why = v["result"].as_str().or(v["subtype"].as_str()).unwrap_or("the agent reported an error");
                    self.error = Some(why.to_string());
                }
                if let Some(c) = v["total_cost_usd"].as_f64() {
                    self.usage.reported_cost_usd = Some(c);
                }
                let u = &v["usage"];
                if u.is_object() {
                    let (read, write) = (n(&u["cache_read_input_tokens"]), n(&u["cache_creation_input_tokens"]));
                    self.usage.prompt_tokens = n(&u["input_tokens"]) + read + write;
                    self.usage.completion_tokens = n(&u["output_tokens"]);
                    self.usage.cache_read_tokens = read;
                    self.usage.cache_write_tokens = write;
                }
                // Gemini: {"type":"result","status":"error"|"success","stats":{…}}
                if v["status"] == "error" && self.error.is_none() {
                    self.error = Some(v["error"]["message"].as_str().unwrap_or("the agent reported an error").into());
                }
                self.gemini_stats(&v["stats"]);
            }
            // ---- Gemini CLI / Qwen Code (stream-json) ----
            "init" => {
                if let Some(m) = v["model"].as_str() {
                    self.settings.insert("Model".into(), m.into());
                }
            }
            "message" if v["role"] == "assistant" => {
                let t = v["content"].as_str().map(str::to_string).unwrap_or_else(|| {
                    v["content"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|c| c["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("")
                });
                self.text(&t, on);
            }
            "tool_use" => {
                let name = v["tool_name"].as_str().or(v["name"].as_str()).unwrap_or("tool");
                let id = v["tool_id"].as_str().or(v["id"].as_str()).unwrap_or("");
                let input = if v["parameters"].is_object() { &v["parameters"] } else { &v["input"] };
                let kind = tool_kind(name);
                self.tool(id, tool_title(name, input), kind, "pending", on);
            }
            "tool_result" => {
                let id = v["tool_id"].as_str().or(v["tool_use_id"].as_str()).unwrap_or("");
                self.finish_tool(id, v["status"].as_str().is_none_or(|s| s == "success"));
            }
            "error" => {
                let m = v["message"].as_str().or(v["error"]["message"].as_str()).unwrap_or("error");
                // Gemini warnings are `error` events with severity "warning".
                if v["severity"] != "warning" {
                    self.error = Some(m.to_string());
                }
            }
            // ---- Codex (exec --json) ----
            "thread.started" | "turn.started" => {}
            "item.started" | "item.updated" | "item.completed" => self.codex_item(ty, &v["item"], on),
            "turn.completed" => {
                let u = &v["usage"];
                self.usage.prompt_tokens += n(&u["input_tokens"]);
                self.usage.cache_read_tokens += n(&u["cached_input_tokens"]);
                self.usage.completion_tokens += n(&u["output_tokens"]);
            }
            "turn.failed" => {
                self.error = Some(v["error"]["message"].as_str().unwrap_or("the turn failed").to_string());
            }
            _ => {}
        }
    }

    /// Antigravity CLI (`agy`) stream-json: `init`, `step_update`s, `result`.
    fn antigravity(&mut self, ev: &str, p: &Value, on: &mut dyn FnMut(Update)) {
        match ev {
            "init" => {
                if let Some(m) = p["model"].as_str() {
                    self.settings.insert("Model".into(), m.into());
                }
            }
            "step_update" => {
                let done = p["state"] == "DONE";
                let id = format!("step-{}", p["step_index"]);
                match p["step_type"].as_str().unwrap_or("") {
                    "agent_response" => {
                        if let Some(t) = p["text_delta"].as_str() {
                            self.text(t, on);
                        }
                    }
                    "tool" => {
                        let info = &p["tool_info"];
                        let name = info["name"].as_str().or(p["tool_name"].as_str()).unwrap_or("tool");
                        let failed = info.get("error").is_some_and(|e| !e.is_null());
                        let status = match (done, failed) {
                            (false, _) => "pending",
                            (true, true) => "failed",
                            (true, false) => "completed",
                        };
                        let title = tool_title(name, &info["parameters"]);
                        let kind = tool_kind(name);
                        self.tool(&id, title, kind, status, on);
                    }
                    _ => {}
                }
            }
            "result" => {
                self.ended = true;
                if let Some(r) = p["response"].as_str() {
                    self.result = Some(r.to_string());
                }
                let status = p["status"].as_str().unwrap_or("SUCCESS");
                if status != "SUCCESS" && status != "CANCELED" {
                    self.error = Some(p["error"].as_str().map(str::to_string).unwrap_or(format!("status {status}")));
                }
                let u = &p["usage"];
                if u.is_object() {
                    self.usage.prompt_tokens = n(&u["input_tokens"]);
                    self.usage.completion_tokens = n(&u["output_tokens"]) + n(&u["thinking_tokens"]);
                    self.usage.cache_read_tokens = n(&u["cache_read_tokens"]);
                }
            }
            _ => {}
        }
    }

    fn gemini_stats(&mut self, stats: &Value) {
        if !stats.is_object() || self.usage.prompt_tokens > 0 {
            return;
        }
        // Flat totals, or a per-model breakdown.
        let mut add = |t: &Value| {
            self.usage.prompt_tokens += n(&t["input_tokens"]) + n(&t["prompt"]);
            self.usage.completion_tokens += n(&t["output_tokens"]) + n(&t["candidates"]);
            self.usage.cache_read_tokens += n(&t["cached"]) + n(&t["cached_tokens"]);
        };
        add(stats);
        for m in stats["models"].as_object().into_iter().flat_map(|o| o.values()) {
            add(&m["tokens"]);
        }
    }

    fn codex_item(&mut self, ty: &str, item: &Value, on: &mut dyn FnMut(Update)) {
        let id = item["id"].as_str().unwrap_or("");
        let done = ty == "item.completed";
        let kind = item["type"].as_str().or(item["item_type"].as_str()).unwrap_or("");
        match kind {
            "agent_message" | "assistant_message" if done => {
                let t = item["text"].as_str().unwrap_or("");
                self.text(t, on);
                self.result = Some(t.to_string());
            }
            "reasoning" => on(Update::Thinking),
            "todo_list" => {
                let items: Vec<&str> =
                    item["items"].as_array().into_iter().flatten().filter_map(|i| i["text"].as_str()).collect();
                if !items.is_empty() {
                    on(Update::Plan(items.join(" · ")));
                }
            }
            "command_execution" | "file_change" | "mcp_tool_call" | "web_search" => {
                let failed = item["status"] == "failed" || item["exit_code"].as_i64().is_some_and(|c| c != 0);
                if done && self.tools.iter().any(|t| t.id == id) {
                    self.finish_tool(id, !failed);
                    return;
                }
                let title = match kind {
                    "command_execution" => format!("shell {}", item["command"].as_str().unwrap_or("")),
                    "file_change" => {
                        let paths: Vec<&str> = item["changes"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|c| c["path"].as_str())
                            .collect();
                        format!("edit {}", paths.join(", "))
                    }
                    "mcp_tool_call" => {
                        format!("{} {}", item["server"].as_str().unwrap_or("mcp"), item["tool"].as_str().unwrap_or(""))
                    }
                    _ => format!("web search {}", item["query"].as_str().unwrap_or("")),
                };
                let tk = match kind {
                    "command_execution" => "execute",
                    "file_change" => "edit",
                    "web_search" => "fetch",
                    _ => "other",
                };
                let status = if !done {
                    "pending"
                } else if failed {
                    "failed"
                } else {
                    "completed"
                };
                self.tool(id, title.trim().to_string(), tk, status, on);
            }
            _ => {}
        }
    }

    /// The agent's final message.
    pub fn final_text(&self) -> String {
        if !self.json {
            let n = self.all.chars().count();
            return self.all.chars().skip(n.saturating_sub(TEXT_TAIL)).collect::<String>().trim().to_string();
        }
        let t = self.result.clone().unwrap_or_else(|| {
            if self.since_tool.trim().is_empty() { self.all.clone() } else { self.since_tool.clone() }
        });
        t.trim().to_string()
    }

    pub fn finish(self, stop_reason: &str, cancelled: bool) -> Outcome {
        Outcome {
            text: self.final_text(),
            stop_reason: stop_reason.into(),
            tool_calls: self.tools,
            cancelled,
            settings: self.settings,
            usage: self.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(json: bool, lines: &[&str]) -> (Stream, Vec<String>) {
        let mut st = Stream::new(json);
        let mut seen = Vec::new();
        for l in lines {
            st.line(l, &mut |u| {
                if let Update::Tool(t) = u {
                    seen.push(format!("{}:{}", t.kind, t.title));
                }
            });
        }
        (st, seen)
    }

    #[test]
    fn reads_claude_code_stream_json() {
        let (st, seen) = read(
            true,
            &[
                r#"{"type":"system","subtype":"init","model":"claude-opus-5-5","session_id":"s"}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Looking."},{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"cargo test"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","is_error":false}]}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t2","name":"Edit","input":{"file_path":"src/a.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t2","is_error":true}]}}"#,
                "not json at all",
                r#"{"type":"result","subtype":"success","is_error":false,"result":"Fixed it.","total_cost_usd":0.12,"usage":{"input_tokens":10,"cache_read_input_tokens":900,"cache_creation_input_tokens":90,"output_tokens":50}}"#,
            ],
        );
        assert_eq!(seen, ["execute:Bash cargo test", "edit:Edit src/a.rs"]);
        let out = st.finish("end_turn", false);
        assert_eq!(out.text, "Fixed it.");
        assert_eq!(out.settings["Model"], "claude-opus-5-5");
        assert_eq!(out.tool_calls[0].status, "completed");
        assert_eq!(out.tool_calls[1].status, "failed");
        assert_eq!(
            out.usage,
            Usage {
                prompt_tokens: 1000,
                completion_tokens: 50,
                cache_read_tokens: 900,
                cache_write_tokens: 90,
                reported_cost_usd: Some(0.12)
            }
        );
    }

    #[test]
    fn reads_codex_exec_json() {
        let (st, seen) = read(
            true,
            &[
                r#"{"type":"thread.started","thread_id":"x"}"#,
                r#"{"type":"item.started","item":{"id":"i1","type":"command_execution","command":"ls","status":"in_progress"}}"#,
                r#"{"type":"item.completed","item":{"id":"i1","type":"command_execution","command":"ls","exit_code":0,"status":"completed"}}"#,
                r#"{"type":"item.completed","item":{"id":"i2","type":"file_change","changes":[{"path":"a.txt","kind":"add"}],"status":"completed"}}"#,
                r#"{"type":"item.completed","item":{"id":"i3","type":"agent_message","text":"All done."}}"#,
                r#"{"type":"turn.completed","usage":{"input_tokens":2000,"cached_input_tokens":1500,"output_tokens":80}}"#,
            ],
        );
        assert_eq!(seen, ["execute:shell ls", "edit:edit a.txt"]);
        assert!(st.error.is_none());
        let out = st.finish("end_turn", false);
        assert_eq!(out.text, "All done.");
        assert!(out.tool_calls.iter().all(|t| t.status == "completed"));
        assert_eq!(
            (out.usage.prompt_tokens, out.usage.cache_read_tokens, out.usage.completion_tokens),
            (2000, 1500, 80)
        );
        let (failed, _) = read(true, &[r#"{"type":"turn.failed","error":{"message":"quota"}}"#]);
        assert_eq!(failed.error.as_deref(), Some("quota"));
    }

    #[test]
    fn reads_gemini_stream_json() {
        let (st, seen) = read(
            true,
            &[
                r#"{"type":"init","session_id":"s","model":"gemini-pro"}"#,
                r#"{"type":"message","role":"user","content":"task"}"#,
                r#"{"type":"message","role":"assistant","content":"Checking ","delta":true}"#,
                r#"{"type":"tool_use","tool_name":"read_file","tool_id":"r1","parameters":{"path":"README.md"}}"#,
                r#"{"type":"tool_result","tool_id":"r1","status":"success"}"#,
                r#"{"type":"message","role":"assistant","content":"Done ","delta":true}"#,
                r#"{"type":"message","role":"assistant","content":"here.","delta":true}"#,
                r#"{"type":"result","status":"success","stats":{"models":{"gemini-pro":{"tokens":{"prompt":300,"candidates":20,"cached":100}}}}}"#,
            ],
        );
        assert_eq!(seen, ["read:read_file README.md"]);
        let out = st.finish("end_turn", false);
        assert_eq!(out.text, "Done here.", "the final message is the text after the last tool call");
        assert_eq!(out.settings["Model"], "gemini-pro");
        assert_eq!((out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.cache_read_tokens), (300, 20, 100));
    }

    #[test]
    fn reads_cursor_tool_calls_and_plain_text() {
        let (st, seen) = read(
            true,
            &[
                r#"{"type":"tool_call","subtype":"started","call_id":"c1","tool_call":{"writeToolCall":{"args":{"path":"x.rs"}}}}"#,
                r#"{"type":"tool_call","subtype":"completed","call_id":"c1","tool_call":{"writeToolCall":{"args":{"path":"x.rs"}}}}"#,
                r#"{"type":"result","subtype":"success","is_error":false,"result":"ok"}"#,
            ],
        );
        assert_eq!(seen, ["edit:write x.rs"]);
        assert_eq!(st.finish("end_turn", false).text, "ok");
        let (st, _) = read(false, &["Applied edit to a.py", "Done."]);
        assert_eq!(st.final_text(), "Applied edit to a.py\nDone.");
    }

    #[test]
    fn reads_antigravity_stream_json() {
        let (st, seen) = read(
            true,
            &[
                r#"{"event":"init","conversation_id":"c","init":{"cwd":"/w","tools":[],"permission_mode":"always-proceed","model":"gemini-3-pro"}}"#,
                r#"{"event":"step_update","step_update":{"conversation_id":"c","step_index":0,"state":"DONE","step_type":"user_input"}}"#,
                r#"{"event":"step_update","step_update":{"conversation_id":"c","step_index":1,"state":"ACTIVE","step_type":"tool","tool_name":"run_command","tool_info":{"name":"run_command","parameters":{"CommandLine":"cargo test"}}}}"#,
                r#"{"event":"step_update","step_update":{"conversation_id":"c","step_index":1,"state":"DONE","step_type":"tool","tool_info":{"name":"run_command","parameters":{"CommandLine":"cargo test"},"output":"ok"}}}"#,
                r#"{"event":"step_update","step_update":{"conversation_id":"c","step_index":2,"state":"DONE","step_type":"tool","tool_info":{"name":"write_to_file","parameters":{"TargetFile":"a.rs"},"error":{"type":"x","message":"denied"}}}}"#,
                r#"{"event":"step_update","step_update":{"conversation_id":"c","step_index":3,"state":"ACTIVE","step_type":"agent_response","text_delta":"All "}}"#,
                r#"{"event":"step_update","step_update":{"conversation_id":"c","step_index":3,"state":"ACTIVE","step_type":"agent_response","text_delta":"good."}}"#,
                r#"{"event":"result","result":{"conversation_id":"c","status":"SUCCESS","response":"All good.","num_turns":1,"usage":{"input_tokens":10415,"output_tokens":657,"thinking_tokens":616,"cache_read_tokens":8113,"total_tokens":11072}}}"#,
            ],
        );
        assert_eq!(seen, ["execute:run_command", "edit:write_to_file"]);
        assert!(st.ended && st.error.is_none());
        let out = st.finish("end_turn", false);
        assert_eq!(out.text, "All good.");
        assert_eq!(out.settings["Model"], "gemini-3-pro");
        assert_eq!(out.tool_calls[0].status, "completed");
        assert_eq!(out.tool_calls[1].status, "failed");
        assert_eq!(
            (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.cache_read_tokens),
            (10415, 1273, 8113)
        );
        let (bad, _) =
            read(true, &[r#"{"event":"result","result":{"status":"ERROR","error":"authentication required"}}"#]);
        assert_eq!(bad.error.as_deref(), Some("authentication required"));

        let agy = ExternalAgent::new("g", "G", "agy", &[]).cli(Dialect::Antigravity);
        let a = build_args(&agy, &BTreeMap::new(), Policy { allow_execute: false }, None, "x");
        assert_eq!(a, ["-p", "--input-format", "stream-json", "--output-format", "stream-json"]);
        assert!(
            build_args(&agy, &BTreeMap::new(), Policy { allow_execute: true }, None, "x")
                .contains(&"--dangerously-skip-permissions".to_string())
        );
        let line = stdin_payload(Dialect::Antigravity, "fix \"it\"\nnow");
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["message"]["content"], "fix \"it\"\nnow");
    }

    #[test]
    fn builds_flags_from_policy_and_options() {
        let claude = ExternalAgent::new("c", "C", "claude", &["--add-dir", "../shared"]).cli(Dialect::Claude);
        let opts: BTreeMap<String, String> =
            [("model".to_string(), "opus".to_string()), ("effort".to_string(), "high".to_string())].into();
        let a = build_args(&claude, &opts, Policy { allow_execute: false }, None, "hi").join(" ");
        assert!(a.starts_with("-p --output-format stream-json --verbose --add-dir ../shared"), "{a}");
        assert!(a.contains("--disallowedTools Bash") && a.contains("--model opus"), "{a}");
        assert!(a.contains("--effort high"), "{a}");
        assert!(!a.contains("--strict-mcp-config"), "your MCP servers are used by default");
        let mut isolated = claude.clone();
        isolated.inherit_config = false;
        assert!(
            build_args(&isolated, &BTreeMap::new(), Policy { allow_execute: true }, None, "")
                .join(" ")
                .contains("--allowedTools Bash --setting-sources project,local --strict-mcp-config")
        );

        let codex = ExternalAgent::new("x", "X", "codex", &[]).cli(Dialect::Codex);
        let m: BTreeMap<String, String> = [("model".to_string(), "gpt-5".to_string())].into();
        let a = build_args(&codex, &m, Policy { allow_execute: true }, None, "hi");
        assert_eq!(a[..3], ["exec", "--model", "gpt-5"]);
        assert_eq!(a.last().map(String::as_str), Some("-"), "codex reads the task from stdin");

        let aider = ExternalAgent::new("a", "A", "aider", &["--message-file", "{prompt_file}"]).cli(Dialect::Aider);
        let a =
            build_args(&aider, &BTreeMap::new(), Policy { allow_execute: false }, Some(Path::new("/tmp/t.md")), "x");
        assert_eq!(a[..2], ["--message-file", "/tmp/t.md"]);
        assert!(a.contains(&"--no-auto-commits".to_string()));
        let gemini = ExternalAgent::new("g", "G", "gemini", &[]).cli(Dialect::Gemini);
        let a = build_args(&gemini, &BTreeMap::new(), Policy { allow_execute: true }, None, "x");
        assert_eq!(a, ["--output-format", "stream-json", "--approval-mode", "yolo", "--skip-trust"]);
        let qwen = ExternalAgent::new("q", "Q", "qwen", &[]).cli(Dialect::Qwen);
        assert!(
            !build_args(&qwen, &BTreeMap::new(), Policy { allow_execute: true }, None, "x")
                .contains(&"--skip-trust".into())
        );
        let cursor = ExternalAgent::new("u", "U", "cursor-agent", &["{prompt}"]).cli(Dialect::Cursor);
        let a = build_args(&cursor, &BTreeMap::new(), Policy { allow_execute: true }, None, "do it");
        assert_eq!(a, ["-p", "--output-format", "stream-json", "do it", "--force"]);
    }

    fn python() -> Option<String> {
        ["python3", "python"].into_iter().find(|p| super::super::resolve_command(p).is_some()).map(String::from)
    }

    /// A fake headless CLI: echoes its stdin back as a Claude-style event
    /// stream and writes a file, or fails like a signed-out tool.
    const FAKE: &str = r#"
import json, sys
args = sys.argv[1:]
if "--version" in args:
    print("fake 1.2.3"); sys.exit(0)
if "--input-format" in args:
    # Like agy: one stream-json message in, events out, then it waits.
    import time
    msg = json.loads(sys.stdin.readline())
    task = msg["message"]["content"]
    print(json.dumps({"event": "init", "init": {"model": "agy-fake"}}), flush=True)
    print(json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "got " + str(len(task)),
        "usage": {"input_tokens": 3, "output_tokens": 4}}}), flush=True)
    time.sleep(60); sys.exit(0)
task = sys.stdin.read()
if "fail" in task:
    sys.stderr.write("Please run /login\n"); sys.exit(1)
open("made.txt", "w").write(task)
def send(o): print(json.dumps(o), flush=True)
send({"type": "system", "subtype": "init", "model": "fake-model"})
send({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "w", "name": "Write", "input": {"file_path": "made.txt"}}]}})
send({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "w"}]}})
send({"type": "result", "is_error": False, "result": "wrote " + str(len(task)) + " chars; bash " + ("allowed" if "--allowedTools" in args else "denied"), "usage": {"input_tokens": 5, "output_tokens": 7}})
"#;

    /// A launcher for the fake CLI that passes every argument on.
    fn launcher(dir: &Path, py: &str) -> std::path::PathBuf {
        let script = dir.join("fake_cli.py");
        std::fs::write(&script, FAKE).unwrap();
        let py = super::super::resolve_command(py).unwrap();
        if cfg!(windows) {
            let l = dir.join("fake.cmd");
            std::fs::write(
                &l,
                format!(
                    "@\"{}\" \"{}\" %*
",
                    py.display(),
                    script.display()
                ),
            )
            .unwrap();
            l
        } else {
            let l = dir.join("fake.sh");
            std::fs::write(
                &l,
                format!(
                    "#!/bin/sh
exec \"{}\" \"{}\" \"$@\"
",
                    py.display(),
                    script.display()
                ),
            )
            .unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&l, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            l
        }
    }

    #[tokio::test]
    async fn drives_a_fake_cli_through_a_turn() {
        let Some(py) = python() else { return };
        let d = tempfile::tempdir().unwrap();
        let exe = launcher(d.path(), &py);
        let agent = ExternalAgent::new("f", "Fake", &exe.to_string_lossy(), &[]).cli(Dialect::Claude);
        let never = AtomicBool::new(false);
        // Far beyond any command-line limit: the task goes through stdin.
        let prompt = format!("task {}", "x".repeat(300_000));
        let mut seen = Vec::new();
        let out = run(
            &agent,
            d.path(),
            &prompt,
            &BTreeMap::new(),
            Policy { allow_execute: true },
            Duration::from_secs(60),
            &never,
            |u| {
                if let Update::Tool(t) = u {
                    seen.push(t.title.clone());
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(out.text, format!("wrote {} chars; bash allowed", prompt.len()));
        assert_eq!(seen, ["Write made.txt"]);
        assert_eq!(out.settings["Model"], "fake-model");
        assert_eq!((out.usage.prompt_tokens, out.usage.completion_tokens), (5, 7));
        assert_eq!(std::fs::read_to_string(d.path().join("made.txt")).unwrap().len(), prompt.len());

        let denied = run(
            &agent,
            d.path(),
            "t",
            &BTreeMap::new(),
            Policy { allow_execute: false },
            Duration::from_secs(60),
            &never,
            |_| {},
        )
        .await
        .unwrap();
        assert!(denied.text.ends_with("bash denied"), "{}", denied.text);

        let err = run(
            &agent,
            d.path(),
            "please fail",
            &BTreeMap::new(),
            Policy { allow_execute: true },
            Duration::from_secs(60),
            &never,
            |_| {},
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("/login") && err.contains("Sign in"), "{err}");
        let v = probe(&agent, d.path()).await.unwrap();
        assert_eq!(v["agentInfo"]["version"], "fake 1.2.3");

        let stop = AtomicBool::new(true);
        let out = run(
            &agent,
            d.path(),
            "t",
            &BTreeMap::new(),
            Policy { allow_execute: true },
            Duration::from_secs(60),
            &stop,
            |_| {},
        )
        .await
        .unwrap();
        assert!(out.cancelled);

        let agy = ExternalAgent::new("g", "Agy", &exe.to_string_lossy(), &[]).cli(Dialect::Antigravity);
        let task = format!(
            "multi
line \"task\" {}",
            "y".repeat(100_000)
        );
        let started = Instant::now();
        let out = run(
            &agy,
            d.path(),
            &task,
            &BTreeMap::new(),
            Policy { allow_execute: true },
            Duration::from_secs(60),
            &never,
            |_| {},
        )
        .await
        .unwrap();
        assert_eq!(out.text, format!("got {}", task.chars().count()));
        assert_eq!(out.settings["Model"], "agy-fake");
        assert!(started.elapsed() < Duration::from_secs(20), "a CLI that keeps waiting after its result is stopped");
    }
}
