//! The actor's tools. File tools are confined to the run's worktree
//! (no absolute paths, no `..` escapes, no symlinks pointing outside, no
//! `.git`). `run_command` is a real shell on your machine, confined only by
//! its working directory and a timeout; turn it off in settings
//! (`limits.allow_commands`) if that is not acceptable.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use reqwest::Url;
use serde_json::{Value, json};
use tokio::process::Command;

use super::protocol::Action;
use crate::config::{Limits, NetAccess};
use crate::providers::ToolDef;
use crate::storage::Autonomy;
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

fn is_inside_root(path: &Path, root: &Path) -> bool {
    #[cfg(windows)]
    {
        let p_str = path.to_string_lossy().to_lowercase().replace('/', "\\");
        let r_str = root.to_string_lossy().to_lowercase().replace('/', "\\");
        if p_str == r_str {
            return true;
        }
        let r_prefix = if r_str.ends_with('\\') { r_str } else { format!("{r_str}\\") };
        p_str.starts_with(&r_prefix)
    }
    #[cfg(not(windows))]
    {
        path.starts_with(root)
    }
}

fn is_same_path(a: &Path, b: &Path) -> bool {
    #[cfg(windows)]
    {
        let a_str = a.to_string_lossy().to_lowercase().replace('/', "\\");
        let b_str = b.to_string_lossy().to_lowercase().replace('/', "\\");
        a_str.trim_end_matches('\\') == b_str.trim_end_matches('\\')
    }
    #[cfg(not(windows))]
    {
        a == b
    }
}

fn strip_read_file_line_numbers(s: &str) -> Option<String> {
    let lines: Vec<&str> = s.split('\n').collect();
    let any_numbered = lines.iter().any(|l| {
        let t = l.trim_start();
        if let Some(pos) = t.find('|') {
            let num = t[..pos].trim();
            !num.is_empty() && num.chars().all(|c| c.is_ascii_digit())
        } else {
            false
        }
    });
    if !any_numbered {
        return None;
    }
    let stripped: Vec<String> = lines
        .iter()
        .map(|l| {
            let t = l.trim_start();
            if let Some(pos) = t.find('|') {
                let num = t[..pos].trim();
                if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
                    let rest = &t[pos + 1..];
                    return rest.strip_prefix(' ').unwrap_or(rest).to_string();
                }
            }
            l.to_string()
        })
        .collect();
    Some(stripped.join("\n"))
}

fn replace_by_line_matching<F>(text_lf: &str, old_lf: &str, new_lf: &str, match_line: F) -> Option<String>
where
    F: Fn(&str, &str) -> bool,
{
    let text_lines: Vec<&str> = text_lf.split('\n').collect();
    let mut old_lines: Vec<&str> = old_lf.split('\n').collect();
    if old_lines.last() == Some(&"") && old_lines.len() > 1 {
        old_lines.pop();
    }
    if old_lines.first() == Some(&"") && old_lines.len() > 1 {
        old_lines.remove(0);
    }
    if old_lines.is_empty() || text_lines.len() < old_lines.len() {
        return None;
    }

    let m = old_lines.len();
    let mut matching_indices = Vec::new();
    for i in 0..=(text_lines.len() - m) {
        if (0..m).all(|j| match_line(text_lines[i + j], old_lines[j])) {
            matching_indices.push(i);
        }
    }

    if matching_indices.len() == 1 {
        let start_idx = matching_indices[0];
        let end_idx = start_idx + m;

        let mut out_parts = Vec::new();
        if start_idx > 0 {
            out_parts.push(text_lines[..start_idx].join("\n"));
        }
        out_parts.push(new_lf.to_string());
        if end_idx < text_lines.len() {
            out_parts.push(text_lines[end_idx..].join("\n"));
        }
        Some(out_parts.join("\n"))
    } else {
        None
    }
}

pub fn apply_edit(text: &str, old: &str, new: &str) -> Result<String, String> {
    // Stage 1: Exact string match
    let count = text.matches(old).count();
    if count == 1 {
        return Ok(text.replacen(old, new, 1));
    }
    if count > 1 {
        return Err(format!("\"old\" text occurs {count} times; include more surrounding lines to identify a unique match"));
    }

    // Stage 2: Line-ending normalization (handle CRLF vs LF)
    let is_crlf = text.contains("\r\n");
    let text_lf = text.replace("\r\n", "\n");
    let old_lf = old.replace("\r\n", "\n");
    let new_lf = new.replace("\r\n", "\n");

    let count_lf = text_lf.matches(&old_lf).count();
    if count_lf == 1 {
        let replaced_lf = text_lf.replacen(&old_lf, &new_lf, 1);
        return Ok(if is_crlf { replaced_lf.replace('\n', "\r\n") } else { replaced_lf });
    }
    if count_lf > 1 {
        return Err(format!("\"old\" text occurs {count_lf} times; include more surrounding lines to identify a unique match"));
    }

    // Stage 3: If old_lf contains line numbers copied from read_file, strip them
    if let Some(stripped_old) = strip_read_file_line_numbers(&old_lf) {
        let count_s = text_lf.matches(&stripped_old).count();
        if count_s == 1 {
            let replaced_lf = text_lf.replacen(&stripped_old, &new_lf, 1);
            return Ok(if is_crlf { replaced_lf.replace('\n', "\r\n") } else { replaced_lf });
        }
        if count_s > 1 {
            return Err(format!("\"old\" text occurs {count_s} times; include more surrounding lines to identify a unique match"));
        }
    }

    // Stage 4: Trailing whitespace normalization on each line
    if let Some(res) = replace_by_line_matching(&text_lf, &old_lf, &new_lf, |a, b| a.trim_end() == b.trim_end()) {
        return Ok(if is_crlf { res.replace('\n', "\r\n") } else { res });
    }

    // Stage 5: Line-by-line whitespace-relaxed match (ignoring leading and trailing whitespace per line)
    // Allows edits to succeed when indentation differs (2 spaces vs 4 spaces, tabs vs spaces)
    if let Some(res) = replace_by_line_matching(&text_lf, &old_lf, &new_lf, |a, b| a.trim() == b.trim()) {
        return Ok(if is_crlf { res.replace('\n', "\r\n") } else { res });
    }

    // Stage 6: Trim leading/trailing blank lines in old
    let old_trimmed_block = old_lf.trim();
    if !old_trimmed_block.is_empty() && old_trimmed_block != old_lf {
        let count_block = text_lf.matches(old_trimmed_block).count();
        if count_block == 1 {
            let replaced_lf = text_lf.replacen(old_trimmed_block, &new_lf, 1);
            return Ok(if is_crlf { replaced_lf.replace('\n', "\r\n") } else { replaced_lf });
        }
        if count_block > 1 {
            return Err(format!("\"old\" text occurs {count_block} times; include more surrounding lines to identify a unique match"));
        }
    }

    let hint = if text_lf.contains(old_lf.trim()) {
        " (it matches after trimming whitespace: check indentation)"
    } else {
        " (read_file the current content and copy the text exactly)"
    };
    Err(format!("\"old\" text not found in file{hint}"))
}

/// What a tool call can do, which decides whether it needs approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolClass {
    /// Looks at things only.
    Read,
    /// Changes files in the worktree.
    Write,
    /// Runs programs or removes things.
    Exec,
    /// `finish`, `delegate`: engine control, never gated.
    Control,
}

impl ToolClass {
    /// Whether a call of this class waits for the user under `autonomy`.
    pub fn needs_approval(self, autonomy: Autonomy) -> bool {
        match autonomy {
            Autonomy::Full => false,
            Autonomy::Semi => self == ToolClass::Exec,
            Autonomy::Manual => matches!(self, ToolClass::Write | ToolClass::Exec),
        }
    }
}

struct BuiltinTool {
    name: &'static str,
    class: ToolClass,
    description: &'static str,
    /// JSON schema properties: (name, type, description).
    params: &'static [(&'static str, &'static str, &'static str)],
    required: &'static [&'static str],
    needs: Needs,
}

#[derive(PartialEq)]
enum Needs {
    Nothing,
    Commands,
    /// `web_search` / `web_fetch`: the internet level is `docs` or more.
    Web,
    /// Browser and page screenshots: `local` or more (any URL only at `full`).
    Pages,
    ScreenCapture,
    /// The agent's own model accepts images.
    Vision,
}

/// The first obvious download or remote-access command in a shell command
/// line (curl, wget, PowerShell's web cmdlets, ssh, git's remote commands).
/// A best-effort guard for agents without internet access, not a firewall:
/// build and install commands still reach their package registries.
fn net_command(command: &str) -> Option<String> {
    const WORDS: &[&str] = &[
        "curl", "wget", "invoke-webrequest", "invoke-restmethod", "iwr", "irm", "nc", "ncat", "telnet", "ftp", "sftp", "scp", "ssh",
    ];
    let words: Vec<String> = command
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_'))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    for (i, w) in words.iter().enumerate() {
        if WORDS.contains(&w.as_str()) {
            return Some(w.clone());
        }
        if w == "git" && words.get(i + 1).is_some_and(|n| ["clone", "fetch", "pull", "push", "ls-remote"].contains(&n.as_str())) {
            return Some(format!("git {}", words[i + 1]));
        }
    }
    None
}

/// Where screenshots are saved (inside the worktree, never checkpointed).
pub const SHOTS_DIR: &str = ".orchopork-shots";

/// Lines added and removed between two versions of a file (a multiset
/// comparison of lines, close to `git diff --numstat` for typical edits).
pub fn line_delta(before: &str, after: &str) -> (usize, usize) {
    let mut count: std::collections::HashMap<&str, i64> = std::collections::HashMap::new();
    for l in before.lines() {
        *count.entry(l).or_default() += 1;
    }
    let mut added = 0;
    for l in after.lines() {
        match count.get_mut(l) {
            Some(c) if *c > 0 => *c -= 1,
            _ => added += 1,
        }
    }
    let removed = count.values().filter(|c| **c > 0).sum::<i64>() as usize;
    (added, removed)
}

fn delta_tag((added, removed): (usize, usize)) -> String {
    format!("[+{added} -{removed} lines]")
}

/// The `[+a -r lines]` tag a file tool put in its result.
pub fn lines_of(obs: &str) -> Option<(u64, u64)> {
    let start = obs.find("[+")?;
    let tag = &obs[start + 2..];
    let (a, rest) = tag.split_once(" -")?;
    let r = rest.split_once(' ')?.0;
    Some((a.trim().parse().ok()?, r.trim().parse().ok()?))
}

/// Images shown at once by `view_image`.
pub const MAX_VIEW: usize = 6;

/// The image paths a `view_image` call names (`path` and/or `paths`).
pub fn image_args(a: &Action) -> Vec<String> {
    let mut v: Vec<String> = a.str_arg("path").or_else(|| a.str_arg("file")).map(str::to_string).into_iter().collect();
    match a.args.get("paths") {
        Some(Value::Array(list)) => v.extend(list.iter().filter_map(Value::as_str).map(str::to_string)),
        Some(Value::String(s)) => v.extend(s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty())),
        _ => {}
    }
    v.dedup();
    v.truncate(MAX_VIEW);
    v
}

/// Which consecutive frames differ, judged by their pixels' encoding
/// (identical files mean nothing on screen changed).
fn frame_changes(frames: &[(PathBuf, String)]) -> String {
    if frames.len() < 2 {
        return String::new();
    }
    let bytes: Vec<Vec<u8>> = frames.iter().map(|(p, _)| std::fs::read(p).unwrap_or_default()).collect();
    let changes: Vec<String> = (1..bytes.len())
        .map(|i| format!("{}→{} {}", i, i + 1, if bytes[i] == bytes[i - 1] { "same" } else { "changed" }))
        .collect();
    let moved = changes.iter().filter(|c| c.ends_with("changed")).count();
    format!(
        "\nFrame changes: {} ({})",
        changes.join(", "),
        match moved {
            0 => "nothing changed between frames".to_string(),
            n if n == changes.len() => "every frame differs from the one before".to_string(),
            n => format!("{n} of {} steps changed", changes.len()),
        }
    )
}

const BUILTIN: &[BuiltinTool] = &[
    BuiltinTool {
        name: "list_files",
        class: ToolClass::Read,
        description: "Files under a directory (default: whole repo), recursively, respecting .gitignore.",
        params: &[("path", "string", "directory relative to the repo root")],
        required: &[],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "list_dir",
        class: ToolClass::Read,
        description: "Entries of one directory with sizes, including untracked and ignored files.",
        params: &[("path", "string", "directory relative to the repo root")],
        required: &[],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "read_file",
        class: ToolClass::Read,
        description: "File contents with line numbers (400 lines at a time).",
        params: &[
            ("path", "string", "file relative to the repo root"),
            ("start_line", "integer", "first line, 1-based"),
            ("end_line", "integer", "last line, inclusive"),
        ],
        required: &["path"],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "search",
        class: ToolClass::Read,
        description: "Regex search (git grep -E) returning file:line matches.",
        params: &[("pattern", "string", "extended regular expression"), ("path", "string", "limit to this directory")],
        required: &["pattern"],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "write_file",
        class: ToolClass::Write,
        description: "Create or overwrite a whole file (parent directories are created).",
        params: &[("path", "string", "file relative to the repo root"), ("content", "string", "the complete new content")],
        required: &["path", "content"],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "edit_file",
        class: ToolClass::Write,
        description: "Replace one exact, unique occurrence of `old` with `new` (or every occurrence with replace_all).",
        params: &[
            ("path", "string", "file relative to the repo root"),
            ("old", "string", "existing text, copied exactly (without line numbers)"),
            ("new", "string", "replacement text"),
            ("replace_all", "boolean", "replace every occurrence instead of exactly one"),
        ],
        required: &["path", "old", "new"],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "move_file",
        class: ToolClass::Write,
        description: "Move or rename a file or directory inside the repo.",
        params: &[("from", "string", "current path"), ("to", "string", "new path")],
        required: &["from", "to"],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "delete_file",
        class: ToolClass::Exec,
        description: "Delete a file, or a directory with recursive=true.",
        params: &[("path", "string", "file or directory"), ("recursive", "boolean", "required to delete a directory")],
        required: &["path"],
        needs: Needs::Nothing,
    },
    BuiltinTool {
        name: "run_command",
        class: ToolClass::Exec,
        description: "Run a shell command (cmd.exe on Windows, sh elsewhere) and wait for it; non-interactive, returns output and exit code. For servers and watchers use start_process.",
        params: &[
            ("command", "string", "the command line"),
            ("cwd", "string", "working directory relative to the repo root"),
            ("timeout_secs", "integer", "kill it after this many seconds"),
        ],
        required: &["command"],
        needs: Needs::Commands,
    },
    BuiltinTool {
        name: "start_process",
        class: ToolClass::Exec,
        description: "Start a long-running command in the background (dev server, watcher); returns its id and first output.",
        params: &[
            ("command", "string", "the command line"),
            ("cwd", "string", "working directory relative to the repo root"),
            ("wait_secs", "integer", "seconds to collect startup output (default 3)"),
        ],
        required: &["command"],
        needs: Needs::Commands,
    },
    BuiltinTool {
        name: "process_output",
        class: ToolClass::Read,
        description: "Recent output and status of a background process; without id, lists this run's processes.",
        params: &[("id", "string", "process id from start_process"), ("tail", "integer", "characters from the end (default 4000)")],
        required: &[],
        needs: Needs::Commands,
    },
    BuiltinTool {
        name: "stop_process",
        class: ToolClass::Exec,
        description: "Stop a background process and everything it started.",
        params: &[("id", "string", "process id from start_process")],
        required: &["id"],
        needs: Needs::Commands,
    },
    BuiltinTool {
        name: "screenshot_page",
        class: ToolClass::Read,
        description: "Take a screenshot of a web page (a repo file such as index.html, a localhost URL or any URL) in a headless browser; saved under .orchopork-shots/. Look at it with view_image, or pass it to a teammate who can see images (consult with images).",
        params: &[
            ("url", "string", "a repo file such as index.html (served over http for you), a web URL, or the URL of a server you started"),
            ("name", "string", "file name for the screenshot (default: a timestamp)"),
            ("wait_ms", "integer", "milliseconds to let scripts run first (default 2000)"),
            ("width", "integer", "viewport width (default 1280)"),
            ("height", "integer", "viewport height (default 800)"),
        ],
        required: &["url"],
        needs: Needs::Pages,
    },
    BuiltinTool {
        name: "screenshot_screen",
        class: ToolClass::Exec,
        description: "Take a screenshot of the whole screen of this computer (everything visible, e.g. a desktop app you started), or several in a row with count/interval_ms to see what changes; saved under .orchopork-shots/.",
        params: &[
            ("name", "string", "file name for the screenshot (default: a timestamp)"),
            ("count", "integer", "how many screenshots in a row (default 1, at most 10)"),
            ("interval_ms", "integer", "time between them (default 1000)"),
        ],
        required: &[],
        needs: Needs::ScreenCapture,
    },
    BuiltinTool {
        name: "screenshot_sequence",
        class: ToolClass::Read,
        description: "Take several screenshots of a web page in a row, a fixed time apart, to check what moves or changes (animations, simulations, loading states). Reports which consecutive frames differ; saved under .orchopork-shots/ as <name>-1.png, <name>-2.png, ... Compare them with view_image (several paths at once) or pass them to a teammate who can see images.",
        params: &[
            ("url", "string", "a repo file such as index.html (served over http for you), a web URL, or the URL of a server you started"),
            ("count", "integer", "number of frames (default 4, 2-10)"),
            ("interval_ms", "integer", "time between frames (default 1000)"),
            ("start_ms", "integer", "time before the first frame (default 1000)"),
            ("name", "string", "base file name (default: a timestamp)"),
            ("width", "integer", "viewport width (default 1280)"),
            ("height", "integer", "viewport height (default 800)"),
        ],
        required: &["url"],
        needs: Needs::Pages,
    },
    BuiltinTool {
        name: "view_image",
        class: ToolClass::Read,
        description: "Look at image files (screenshots, designs, charts, a sequence of frames): they are shown to you with the result.",
        params: &[
            ("path", "string", "repo-relative image path (png, jpeg, webp, gif)"),
            ("paths", "array", "several image paths to see side by side (at most 6)"),
        ],
        required: &[],
        needs: Needs::Vision,
    },
    BuiltinTool {
        name: "web_search",
        class: ToolClass::Read,
        description: "Search the web; returns titles, links and snippets.",
        params: &[("query", "string", "search terms")],
        required: &["query"],
        needs: Needs::Web,
    },
    BuiltinTool {
        name: "web_fetch",
        class: ToolClass::Read,
        description: "GET a URL and return it as readable text (no JavaScript). Good for docs and APIs.",
        params: &[("url", "string", "http(s) URL")],
        required: &["url"],
        needs: Needs::Web,
    },
    BuiltinTool {
        name: "browser",
        class: ToolClass::Read,
        description: "Open a page in a real headless browser (runs JavaScript): returns the rendered text, console errors, uncaught exceptions and failed requests (404s), optionally saves a screenshot. `url` may be a repo file such as index.html (served over http, so ES modules work; no server needed), a web URL, or a localhost URL of a server you started with start_process.",
        params: &[
            ("url", "string", "a repo file such as index.html (served over http for you), a web URL, or the URL of a server you started"),
            ("wait_ms", "integer", "milliseconds to let scripts run (default 2000)"),
            ("screenshot", "string", "repo-relative .png path to save a screenshot to"),
            ("width", "integer", "viewport width (default 1280)"),
            ("height", "integer", "viewport height (default 800)"),
        ],
        required: &["url"],
        needs: Needs::Pages,
    },
];

pub fn finish_def() -> ToolDef {
    ToolDef {
        name: "finish".into(),
        description: "The goal is complete and verified; summarize what changed.".into(),
        parameters: json!({ "type": "object", "properties": { "summary": { "type": "string", "description": "what changed and how it was verified" } }, "required": ["summary"] }),
    }
}

pub fn delegate_def() -> ToolDef {
    ToolDef {
        name: "delegate".into(),
        description: "Send work to one of your direct subordinates (by id); they act on it and report back to you.".into(),
        parameters: json!({ "type": "object", "properties": {
            "agent": { "type": "string", "description": "subordinate id" },
            "instruction": { "type": "string", "description": "what they should do" }
        }, "required": ["agent", "instruction"] }),
    }
}

pub struct ToolBox {
    root: PathBuf,
    allow_commands: bool,
    net: NetAccess,
    allow_screen: bool,
    vision: bool,
    timeout: Duration,
    custom: Vec<ToolSpec>,
    /// Only these built-in tools (and no custom ones), when set.
    only: Option<Vec<&'static str>>,
}

/// What a consulted or reviewing agent may use: looking, not changing.
pub const READ_ONLY: &[&str] =
    &["list_files", "list_dir", "read_file", "search", "web_search", "web_fetch", "browser", "process_output"];
/// A tester may also run things.
pub const TESTING: &[&str] = &[
    "list_files",
    "list_dir",
    "read_file",
    "search",
    "web_search",
    "web_fetch",
    "browser",
    "process_output",
    "run_command",
    "start_process",
    "stop_process",
];

impl ToolBox {
    /// Narrows the internet level to `own` (never beyond the workspace's).
    pub fn with_network(mut self, own: Option<NetAccess>) -> Self {
        if let Some(own) = own {
            self.net = self.net.min(own);
        }
        self
    }

    /// The internet level this box was built with.
    pub fn network(&self) -> NetAccess {
        self.net
    }

    /// Offers `view_image`: the agent's model accepts images.
    pub fn with_vision(mut self, on: bool) -> Self {
        self.vision = on;
        self
    }

    /// Whether the agent using this box can see images.
    pub fn sees(&self) -> bool {
        self.vision
    }

    /// Limits the box to `names` (built-in tools only).
    pub fn restricted(mut self, names: &[&'static str]) -> Self {
        self.only = Some(names.to_vec());
        self
    }

    pub fn new(worktree: &Path, limits: &Limits, custom: Vec<ToolSpec>) -> Result<Self> {
        Ok(Self {
            root: crate::fsutil::canonical(worktree)?,
            allow_commands: limits.allow_commands,
            net: limits.net(),
            allow_screen: limits.allow_screen_capture,
            vision: false,
            timeout: Duration::from_secs(limits.command_timeout_secs.max(1)),
            custom,
            only: None,
        })
    }

    fn enabled(&self) -> impl Iterator<Item = &'static BuiltinTool> + '_ {
        BUILTIN.iter().filter(|t| {
            let on = match t.needs {
                Needs::Nothing => true,
                Needs::Commands => self.allow_commands,
                Needs::Web => self.net >= NetAccess::Docs,
                Needs::Pages => self.net >= NetAccess::Local,
                Needs::ScreenCapture => self.allow_screen,
                Needs::Vision => self.vision,
            };
            on && self.only.as_ref().is_none_or(|o| o.contains(&t.name))
        })
    }

    fn custom_tools(&self) -> &[ToolSpec] {
        if self.only.is_some() { &[] } else { &self.custom }
    }

    /// How a call to `tool` is classified for approvals. Custom (skill)
    /// tools and unknown names count as commands.
    pub fn class(&self, tool: &str) -> ToolClass {
        match tool {
            "finish" | "delegate" | "ask" => ToolClass::Control,
            _ => BUILTIN.iter().find(|t| t.name == tool).map_or(ToolClass::Exec, |t| t.class),
        }
    }

    /// Tool reference for the actor's system prompt.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for t in self.enabled() {
            let args: Vec<String> = t
                .params
                .iter()
                .map(|(n, ty, _)| {
                    let ty = match *ty {
                        "integer" => "int",
                        "boolean" => "bool",
                        other => other,
                    };
                    format!("\"{n}\"{}: {ty}", if t.required.contains(n) { "" } else { "?" })
                })
                .collect();
            s.push_str(&format!("- {} {{{}}}: {}\n", t.name, args.join(", "), t.description));
        }
        for t in self.custom_tools() {
            let params = if t.parameters.is_null() { "{}".to_string() } else { t.parameters.to_string() };
            s.push_str(&format!("- {} {}: {}\n", t.name, params, t.description.trim()));
        }
        s
    }

    /// `describe` plus the `finish` line, for single-agent runs.
    pub fn describe_with_finish(&self) -> String {
        let mut s = self.describe();
        s.push_str(super::ask::TOOL_LINE);
        s.push_str("- finish {\"summary\": string}: the goal is complete and verified; summarize what changed.\n");
        s
    }

    /// The same tools as native tool definitions.
    pub fn schemas(&self) -> Vec<ToolDef> {
        let mut v: Vec<ToolDef> = self
            .enabled()
            .map(|t| {
                let props: serde_json::Map<String, Value> = t
                    .params
                    .iter()
                    .map(|(n, ty, d)| {
                        let schema = if *ty == "array" {
                            json!({ "type": "array", "items": { "type": "string" }, "description": d })
                        } else {
                            json!({ "type": ty, "description": d })
                        };
                        ((*n).to_string(), schema)
                    })
                    .collect();
                ToolDef {
                    name: t.name.into(),
                    description: t.description.into(),
                    parameters: json!({ "type": "object", "properties": props, "required": t.required }),
                }
            })
            .collect();
        for t in self.custom_tools() {
            let parameters = match &t.parameters {
                Value::Object(m) if m.contains_key("type") => t.parameters.clone(),
                Value::Object(m) => {
                    // Shorthand {"name": "description"}: a string property each.
                    let props: serde_json::Map<String, Value> = m
                        .iter()
                        .map(|(k, v)| {
                            let d = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
                            (k.clone(), json!({ "type": "string", "description": d }))
                        })
                        .collect();
                    json!({ "type": "object", "properties": props })
                }
                _ => json!({ "type": "object", "properties": {} }),
            };
            v.push(ToolDef { name: t.name.clone(), description: t.description.trim().into(), parameters });
        }
        v
    }

    /// `schemas` plus `finish`, for single-agent runs.
    pub fn schemas_with_finish(&self) -> Vec<ToolDef> {
        let mut v = self.schemas();
        v.push(super::ask::tool_def());
        v.push(finish_def());
        v
    }

    pub async fn execute(&self, a: &Action) -> ToolResult {
        if let Some(only) = &self.only
            && !only.contains(&a.tool.as_str())
        {
            return ToolResult::err(format!("{} is not available here; you may use: {}", a.tool, only.join(", ")));
        }
        // A model can name a tool that was never offered to it: the level
        // is enforced here, not only by what the prompt lists.
        if let Some(t) = BUILTIN.iter().find(|t| t.name == a.tool) {
            let denied = match t.needs {
                Needs::Web => self.net < NetAccess::Docs,
                Needs::Pages => self.net < NetAccess::Local,
                _ => false,
            };
            if denied {
                return ToolResult::err(format!(
                    "{} is not available: your internet access is \"{}\". If you need it, ask a teammate who has it.",
                    a.tool,
                    self.net.as_str()
                ));
            }
        }
        let r = match a.tool.as_str() {
            "list_files" => self.list_files(a).await,
            "list_dir" => self.list_dir(a).await,
            "read_file" => self.read_file(a).await,
            "search" => self.search(a).await,
            "write_file" => self.write_file(a).await,
            "edit_file" => self.edit_file(a).await,
            "move_file" => self.move_file(a).await,
            "delete_file" => self.delete_file(a).await,
            "run_command" => self.run_command(a).await,
            "start_process" => self.start_process(a).await,
            "process_output" => self.process_output(a).await,
            "stop_process" => self.stop_process(a).await,
            "web_search" => self.web_search(a).await,
            "web_fetch" => self.web_fetch(a).await,
            "browser" => self.browser(a).await,
            "screenshot_page" => self.screenshot_page(a).await,
            "screenshot_screen" => self.screenshot_screen(a).await,
            "screenshot_sequence" => self.screenshot_sequence(a).await,
            "view_image" => self.view_image(a).await,
            other => match self.custom_tools().iter().find(|t| t.name == other) {
                Some(t) => self.custom_tool(t, a).await,
                None => {
                    let names: Vec<&str> =
                        self.enabled().map(|t| t.name).chain(self.custom_tools().iter().map(|t| t.name.as_str())).collect();
                    Err(format!(
                        "unknown tool {other:?}. Call a tool by its bare name, one of: {}, finish",
                        names.join(", ")
                    ))
                }
            },
        };
        match r {
            Ok(t) => t,
            Err(msg) => ToolResult::err(msg),
        }
    }

    /// Maps a model-supplied path into the worktree, resolving leading slashes
    /// and checking boundaries carefully.
    fn resolve(&self, raw: &str) -> Result<PathBuf, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Ok(self.root.clone());
        }

        // Refuse common unix root system paths that attempt to escape
        let lower = raw.to_lowercase().replace('\\', "/");
        if lower.starts_with("/etc")
            || lower.starts_with("/var")
            || lower.starts_with("/usr")
            || lower.starts_with("/bin")
            || lower.starts_with("/sbin")
            || lower.starts_with("/dev")
            || lower.starts_with("/proc")
            || lower.starts_with("/sys")
            || lower.starts_with("/root")
            || lower.starts_with("/home")
            || lower.starts_with("/private")
        {
            return Err(format!("{raw:?} resolves outside the repository"));
        }

        let p_abs = Path::new(raw);
        if p_abs.is_absolute() {
            if is_inside_root(p_abs, &self.root) {
                if let Ok(rel) = p_abs.strip_prefix(&self.root) {
                    return self.resolve_relative(rel, raw);
                }
                #[cfg(windows)]
                {
                    let p_str = p_abs.to_string_lossy();
                    let r_str = self.root.to_string_lossy();
                    let r_prefix = if r_str.ends_with('\\') || r_str.ends_with('/') {
                        r_str.to_string()
                    } else {
                        format!("{r_str}\\")
                    };
                    if p_str.len() >= r_prefix.len() && p_str[..r_prefix.len()].eq_ignore_ascii_case(&r_prefix) {
                        let rel = Path::new(&p_str[r_prefix.len()..]);
                        return self.resolve_relative(rel, raw);
                    }
                }
            } else {
                return Err(format!("{raw:?} resolves outside the repository"));
            }
        }

        // Strip leading '/' or '\' so "/water.js" or "\src\file.js" is treated as relative to the root
        let trimmed = raw.trim_start_matches(['/', '\\']);
        self.resolve_relative(Path::new(if trimmed.is_empty() { "." } else { trimmed }), raw)
    }

    fn resolve_relative(&self, p: &Path, raw_for_err: &str) -> Result<PathBuf, String> {
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
                    if is_same_path(&out, &self.root) {
                        return Err(format!("{raw_for_err:?} escapes the repository"));
                    }
                    out.pop();
                }
                Component::RootDir | Component::Prefix(_) => {
                    continue;
                }
            }
        }
        // Symlinks: the deepest existing ancestor must really be inside. A
        // dangling symlink counts as existing (writing through it would
        // create its target, wherever that is); it cannot be resolved, so
        // it is refused.
        let mut probe = out.as_path();
        while std::fs::symlink_metadata(probe).is_err() {
            probe = probe.parent().ok_or("invalid path")?;
        }
        let real = crate::fsutil::canonical(probe)
            .map_err(|_| format!("{raw_for_err:?} is a link that points nowhere"))?;
        if !is_inside_root(&real, &self.root) {
            return Err(format!("{raw_for_err:?} resolves outside the repository"));
        }
        Ok(out)
    }

    fn rel<'a>(&self, p: &'a Path) -> std::borrow::Cow<'a, str> {
        if let Ok(rel) = p.strip_prefix(&self.root) {
            return rel.to_string_lossy();
        }
        #[cfg(windows)]
        {
            let p_str = p.to_string_lossy();
            let r_str = self.root.to_string_lossy();
            if is_same_path(p, &self.root) {
                return std::borrow::Cow::Borrowed("");
            }
            let r_prefix = if r_str.ends_with('\\') || r_str.ends_with('/') {
                r_str.to_string()
            } else {
                format!("{r_str}\\")
            };
            if p_str.len() >= r_prefix.len() && p_str[..r_prefix.len()].eq_ignore_ascii_case(&r_prefix) {
                return std::borrow::Cow::Owned(p_str[r_prefix.len()..].to_string());
            }
        }
        p.to_string_lossy()
    }

    async fn list_files(&self, a: &Action) -> Result<ToolResult, String> {
        let dir = self.resolve(a.str_arg("path").unwrap_or("."))?;
        let mut prefix = self.rel(&dir).replace('\\', "/");
        if prefix == "." || prefix == "./" {
            prefix.clear();
        }
        let prefix = prefix.trim_matches('/').to_string();
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
        let path_arg = a
            .str_arg("path")
            .or_else(|| a.str_arg("file"))
            .or_else(|| a.str_arg("filename"))
            .ok_or("read_file needs \"path\"")?;
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
        let start = a.int_arg("start_line").or_else(|| a.int_arg("line_start")).or_else(|| a.int_arg("offset")).unwrap_or(1).clamp(1, total);
        let end = a
            .int_arg("end_line")
            .or_else(|| a.int_arg("line_end"))
            .or_else(|| a.int_arg("limit").map(|n| start + n.max(1) - 1))
            .unwrap_or(start + READ_DEFAULT_LINES - 1)
            .clamp(start, total);
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
        let mut rel_str = rel.replace('\\', "/");
        if rel_str == "." || rel_str == "./" {
            rel_str.clear();
        }
        let rel_str = rel_str.trim_matches('/').to_string();
        let rel = if rel_str.is_empty() { "." } else { &rel_str };
        let hits = GitRepo::new(&self.root).grep(pattern, rel).await.map_err(|e| format!("search failed: {e}"))?;
        if hits.is_empty() {
            return Ok(ToolResult::ok(format!("no matches for {pattern:?}")));
        }
        Ok(ToolResult::ok(clip(&hits, 20_000)))
    }

    async fn write_file(&self, a: &Action) -> Result<ToolResult, String> {
        let path_arg = a
            .str_arg("path")
            .or_else(|| a.str_arg("file"))
            .or_else(|| a.str_arg("filename"))
            .ok_or("write_file needs \"path\"")?;
        let content = a
            .str_arg("content")
            .or_else(|| a.str_arg("text"))
            .or_else(|| a.str_arg("body"))
            .or_else(|| a.str_arg("data"))
            .ok_or("write_file needs \"content\" (a string)")?;
        let path = self.resolve(path_arg)?;
        if path.is_dir() {
            return Err(format!("{path_arg} is a directory"));
        }
        let existed = path.exists();
        let before = if existed { tokio::fs::read_to_string(&path).await.unwrap_or_default() } else { String::new() };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
        }
        tokio::fs::write(&path, content).await.map_err(|e| format!("cannot write {path_arg}: {e}"))?;
        let lines = content.lines().count();
        Ok(ToolResult::ok(format!(
            "{} {path_arg} ({lines} lines, {} bytes) {}",
            if existed { "overwrote" } else { "created" },
            content.len(),
            delta_tag(line_delta(&before, content))
        )))
    }

    async fn edit_file(&self, a: &Action) -> Result<ToolResult, String> {
        let path_arg = a
            .str_arg("path")
            .or_else(|| a.str_arg("file"))
            .or_else(|| a.str_arg("filename"))
            .ok_or("edit_file needs \"path\"")?;
        let old = a
            .str_arg("old")
            .or_else(|| a.str_arg("old_text"))
            .or_else(|| a.str_arg("search"))
            .or_else(|| a.str_arg("target"));
        let new = a
            .str_arg("new")
            .or_else(|| a.str_arg("new_text"))
            .or_else(|| a.str_arg("replace"))
            .or_else(|| a.str_arg("replacement"));
        let (old, new) = match (old, new) {
            (Some(o), Some(n)) if !o.is_empty() => (o, n),
            (Some(""), Some(n)) if !n.is_empty() => {
                // An empty `old` means "this is the file": fine for a new file.
                let path = self.resolve(path_arg)?;
                if !path.exists() {
                    let mut args = a.args.clone();
                    args.insert("content".into(), Value::String(n.to_string()));
                    return self.write_file(&Action { tool: "write_file".into(), args }).await;
                }
                return Err(format!(
                    "edit_file needs \"old\": text copied exactly from {path_arg} (read_file it first). To replace the whole file, call write_file {{\"path\": \"{path_arg}\", \"content\": <the complete new file>}} instead."
                ));
            }
            _ => return Err("edit_file needs non-empty \"old\" (text copied from the file) and a \"new\" string; to replace a whole file use write_file".into()),
        };
        let path = self.resolve(path_arg)?;
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(format!("{path_arg} does not exist; use write_file to create it"));
            }
            Err(e) => return Err(format!("cannot read {path_arg}: {e}")),
        };
        if a.args.get("replace_all").and_then(Value::as_bool).unwrap_or(false) {
            let n = text.matches(old).count();
            if n == 0 {
                return Err(format!("\"old\" text not found in {path_arg}"));
            }
            let updated = text.replace(old, new);
            tokio::fs::write(&path, &updated).await.map_err(|e| e.to_string())?;
            return Ok(ToolResult::ok(format!("edited {path_arg} ({n} replacements) {}", delta_tag(line_delta(&text, &updated)))));
        }
        let updated = apply_edit(&text, old, new).map_err(|e| format!("{e} in {path_arg}"))?;
        tokio::fs::write(&path, &updated).await.map_err(|e| e.to_string())?;
        Ok(ToolResult::ok(format!("edited {path_arg} {}", delta_tag(line_delta(&text, &updated)))))
    }

    async fn list_dir(&self, a: &Action) -> Result<ToolResult, String> {
        let raw = a.str_arg("path").or_else(|| a.str_arg("dir")).unwrap_or(".");
        let dir = self.resolve(raw)?;
        let mut rd = tokio::fs::read_dir(&dir).await.map_err(|e| format!("cannot list {raw}: {e}"))?;
        let mut entries = Vec::new();
        while let Ok(Some(e)) = rd.next_entry().await {
            let name = e.file_name().to_string_lossy().into_owned();
            if name == ".git" {
                continue;
            }
            let md = e.metadata().await.ok();
            let is_dir = md.as_ref().is_some_and(|m| m.is_dir());
            let size = md.map_or(0, |m| m.len());
            entries.push((is_dir, name, size));
        }
        if entries.is_empty() {
            return Ok(ToolResult::ok(format!("{raw} is empty")));
        }
        entries.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.to_lowercase().cmp(&b.1.to_lowercase())));
        let out: String = entries
            .iter()
            .take(500)
            .map(|(d, n, s)| if *d { format!("{n}/\n") } else { format!("{n}  ({})\n", human_size(*s)) })
            .collect();
        Ok(ToolResult::ok(out))
    }

    async fn move_file(&self, a: &Action) -> Result<ToolResult, String> {
        let from_raw = a.str_arg("from").or_else(|| a.str_arg("source")).or_else(|| a.str_arg("path")).ok_or("move_file needs \"from\"")?;
        let to_raw = a.str_arg("to").or_else(|| a.str_arg("destination")).or_else(|| a.str_arg("dest")).ok_or("move_file needs \"to\"")?;
        let from = self.resolve(from_raw)?;
        let to = self.resolve(to_raw)?;
        if is_same_path(&from, &self.root) {
            return Err("cannot move the repository root".into());
        }
        if !from.exists() {
            return Err(format!("{from_raw} does not exist"));
        }
        if to.exists() {
            return Err(format!("{to_raw} already exists; delete it first or pick another name"));
        }
        if let Some(parent) = to.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
        }
        tokio::fs::rename(&from, &to).await.map_err(|e| format!("cannot move {from_raw} to {to_raw}: {e}"))?;
        Ok(ToolResult::ok(format!("moved {from_raw} -> {to_raw}")))
    }

    async fn delete_file(&self, a: &Action) -> Result<ToolResult, String> {
        let raw = a.str_arg("path").or_else(|| a.str_arg("file")).ok_or("delete_file needs \"path\"")?;
        let path = self.resolve(raw)?;
        if is_same_path(&path, &self.root) {
            return Err("cannot delete the repository root".into());
        }
        let md = tokio::fs::symlink_metadata(&path).await.map_err(|_| format!("{raw} does not exist"))?;
        let removed = if md.is_file() { tokio::fs::read_to_string(&path).await.map(|t| t.lines().count()).unwrap_or(0) } else { 0 };
        if md.is_dir() {
            if !a.args.get("recursive").and_then(Value::as_bool).unwrap_or(false) {
                return Err(format!("{raw} is a directory; pass \"recursive\": true to delete it with its contents"));
            }
            tokio::fs::remove_dir_all(&path).await.map_err(|e| format!("cannot delete {raw}: {e}"))?;
        } else {
            tokio::fs::remove_file(&path).await.map_err(|e| format!("cannot delete {raw}: {e}"))?;
        }
        Ok(ToolResult::ok(format!("deleted {raw} {}", delta_tag((0, removed)))))
    }

    fn command_cwd(&self, a: &Action) -> Result<PathBuf, String> {
        let dir = self.resolve(a.str_arg("cwd").or_else(|| a.str_arg("dir")).unwrap_or("."))?;
        if !dir.is_dir() {
            return Err(format!("working directory {:?} does not exist", a.str_arg("cwd").unwrap_or(".")));
        }
        Ok(dir)
    }

    /// Refuses obvious download commands when this agent has no internet.
    fn check_net_command(&self, command: &str) -> Result<(), String> {
        match net_command(command).filter(|_| self.net < NetAccess::Docs) {
            Some(w) => Err(format!(
                "{w} reaches the internet, and your internet access is \"{}\". Work with what is in the repository, or ask a teammate who has internet access.",
                self.net.as_str()
            )),
            None => Ok(()),
        }
    }

    async fn run_command(&self, a: &Action) -> Result<ToolResult, String> {
        if !self.allow_commands {
            return Err("run_command is disabled in this workspace's settings".into());
        }
        let command = a.str_arg("command").or_else(|| a.str_arg("cmd")).ok_or("run_command needs \"command\"")?;
        self.check_net_command(command)?;
        if cfg!(windows)
            && let Some(problem) = cmd_exe_problem(command)
        {
            return Err(problem.into());
        }
        let cwd = self.command_cwd(a)?;
        let timeout = a
            .int_arg("timeout_secs")
            .or_else(|| a.int_arg("timeout"))
            .map_or(self.timeout, |t| Duration::from_secs(t.clamp(1, 3600) as u64));
        let out = run_shell(&cwd, command, &[], timeout).await;
        Ok(out.into_result(command))
    }

    async fn start_process(&self, a: &Action) -> Result<ToolResult, String> {
        if !self.allow_commands {
            return Err("commands are disabled in this workspace's settings".into());
        }
        let command = a.str_arg("command").or_else(|| a.str_arg("cmd")).ok_or("start_process needs \"command\"")?;
        self.check_net_command(command)?;
        let cwd = self.command_cwd(a)?;
        let wait = a.int_arg("wait_secs").unwrap_or(3).clamp(0, 60) as u64;
        let id = processes::start(&self.root, &cwd, command)?;
        tokio::time::sleep(Duration::from_secs(wait)).await;
        let (status, out) = processes::output(&self.root, &id, 4000).ok_or("process vanished")?;
        Ok(ToolResult::ok(format!(
            "started process {id}: {command}\n[{status}]\n{}\n(use process_output {{\"id\": \"{id}\"}} for more, stop_process to end it)",
            if out.trim().is_empty() { "(no output yet)" } else { out.as_str() }
        )))
    }

    async fn process_output(&self, a: &Action) -> Result<ToolResult, String> {
        let Some(id) = a.str_arg("id").or_else(|| a.str_arg("process")) else {
            let list = processes::list(&self.root);
            if list.is_empty() {
                return Ok(ToolResult::ok("no background processes in this run"));
            }
            return Ok(ToolResult::ok(list.join("\n")));
        };
        let tail = a.int_arg("tail").unwrap_or(4000).clamp(200, 60_000) as usize;
        let (status, out) = processes::output(&self.root, id, tail).ok_or(format!("no process {id:?} in this run"))?;
        Ok(ToolResult::ok(format!("[{id}: {status}]\n{}", if out.trim().is_empty() { "(no output)" } else { out.as_str() })))
    }

    async fn stop_process(&self, a: &Action) -> Result<ToolResult, String> {
        let id = a.str_arg("id").or_else(|| a.str_arg("process")).ok_or("stop_process needs \"id\"")?;
        if !processes::stop(&self.root, id) {
            return Err(format!("no process {id:?} in this run"));
        }
        Ok(ToolResult::ok(format!("stopped {id}")))
    }

    async fn web_search(&self, a: &Action) -> Result<ToolResult, String> {
        let q = a.str_arg("query").or_else(|| a.str_arg("q")).filter(|q| !q.trim().is_empty()).ok_or("web_search needs \"query\"")?;
        Ok(ToolResult::ok(super::browser::search(q.trim()).await?))
    }

    async fn web_fetch(&self, a: &Action) -> Result<ToolResult, String> {
        let raw = a.str_arg("url").ok_or("web_fetch needs \"url\"")?;
        let url = super::browser::parse_web_url(raw)?;
        Ok(ToolResult::ok(clip(&super::browser::fetch(&url).await?, 40_000)))
    }

    /// A fresh path under `SHOTS_DIR` for a screenshot named `name`.
    fn shot_path(&self, name: Option<&str>) -> Result<(PathBuf, String), String> {
        let stem: String = name
            .unwrap_or("")
            .trim()
            .trim_end_matches(".png")
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        let stem = if stem.is_empty() {
            format!("shot-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0))
        } else {
            stem
        };
        let rel = format!("{SHOTS_DIR}/{stem}.png");
        let path = self.resolve(&rel)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        Ok((path, rel))
    }

    async fn screenshot_page(&self, a: &Action) -> Result<ToolResult, String> {
        let (path, rel) = self.shot_path(a.str_arg("name"))?;
        let (url, _server) = self.page_url(a).await?;
        let mut opts = self.render_opts(a, 2000);
        opts.shots = vec![path];
        let r = self.render_page(&url, &opts).await?;
        Ok(ToolResult::ok(format!("Saved the screenshot to {rel}.{}\n{}", self.look_hint_one(), page_report(&url, &r, false))))
    }

    async fn screenshot_screen(&self, a: &Action) -> Result<ToolResult, String> {
        if !self.allow_screen {
            return Err("screen capture is off in this workspace's settings".into());
        }
        let count = a.int_arg("count").unwrap_or(1).clamp(1, 10) as usize;
        let interval = a.int_arg("interval_ms").unwrap_or(1000).clamp(100, 60_000) as u64;
        let base = self.shot_base(a.str_arg("name"));
        let mut frames = Vec::new();
        for i in 0..count {
            if i > 0 {
                tokio::time::sleep(Duration::from_millis(interval)).await;
            }
            let name = if count == 1 { base.clone() } else { format!("{base}-{}", i + 1) };
            let (path, rel) = self.shot_path(Some(&name))?;
            super::screen::capture(&path).await?;
            frames.push((path, rel));
        }
        Ok(ToolResult::ok(format!(
            "Saved {} of the whole screen: {}.{}{}",
            if count == 1 { "a screenshot".to_string() } else { format!("{count} screenshots {interval} ms apart") },
            frames.iter().map(|(_, r)| r.as_str()).collect::<Vec<_>>().join(", "),
            frame_changes(&frames),
            self.look_hint()
        )))
    }

    async fn screenshot_sequence(&self, a: &Action) -> Result<ToolResult, String> {
        let count = a.int_arg("count").unwrap_or(4).clamp(2, 10);
        let interval = a.int_arg("interval_ms").unwrap_or(1000).clamp(50, 30_000) as u64;
        let start = a.int_arg("start_ms").unwrap_or(1000).clamp(0, 30_000) as u64;
        let base = self.shot_base(a.str_arg("name"));
        let mut frames = Vec::new();
        for i in 0..count {
            frames.push(self.shot_path(Some(&format!("{base}-{}", i + 1)))?);
        }
        let (url, _server) = self.page_url(a).await?;
        // One page load; the frames are `interval` ms apart in real time.
        let mut opts = self.render_opts(a, start as i64);
        opts.wait_ms = start;
        opts.interval_ms = interval;
        opts.shots = frames.iter().map(|(p, _)| p.clone()).collect();
        let r = self.render_page(&url, &opts).await?;
        Ok(ToolResult::ok(format!(
            "Saved {count} frames {interval} ms apart (from {start} ms): {}.{}{}\n{}",
            frames.iter().map(|(_, r)| r.as_str()).collect::<Vec<_>>().join(", "),
            frame_changes(&frames),
            self.look_hint(),
            page_report(&url, &r, false)
        )))
    }

    fn shot_base(&self, name: Option<&str>) -> String {
        let n = name.unwrap_or("").trim().trim_end_matches(".png");
        if n.is_empty() {
            format!("seq-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0))
        } else {
            n.to_string()
        }
    }

    fn look_hint(&self) -> &'static str {
        if self.vision {
            " They are shown to you with this result."
        } else {
            " You cannot see images: consult a teammate who can, passing them in images."
        }
    }

    fn look_hint_one(&self) -> &'static str {
        if self.vision {
            " It is shown to you with this result."
        } else {
            " You cannot see images: consult a teammate who can, passing it in images."
        }
    }

    async fn view_image(&self, a: &Action) -> Result<ToolResult, String> {
        let paths = image_args(a);
        if paths.is_empty() {
            return Err("view_image needs \"path\" or \"paths\"".into());
        }
        for p in &paths {
            crate::providers::Image::from_file(&self.resolve(p)?)?;
        }
        Ok(ToolResult::ok(format!("{} shown to you below, in this order.", paths.join(", "))))
    }

    /// An image in the worktree, for a model to look at.
    pub fn load_image(&self, raw: &str) -> Result<crate::providers::Image, String> {
        let path = self.resolve(raw)?;
        crate::providers::Image::from_file(&path)
    }

    /// The URL a page tool opens: a web or localhost URL as given, or a repo
    /// file served over loopback http for as long as the returned server lives.
    async fn page_url(&self, a: &Action) -> Result<(Url, Option<super::browser::LocalServer>), String> {
        let raw = a.str_arg("url").or_else(|| a.str_arg("path")).ok_or(format!("{} needs \"url\"", a.tool))?.trim();
        if raw.contains("://") || raw.starts_with("localhost") || raw.starts_with("127.0.0.1") {
            let full = if raw.contains("://") { raw.to_string() } else { format!("http://{raw}") };
            let url = super::browser::parse_web_url(&full)?;
            if self.net < NetAccess::Full && !matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]")) {
                return Err(format!(
                    "{url} is on the internet, and your internet access is \"{}\": you can open only repo files and localhost pages",
                    self.net.as_str()
                ));
            }
            return Ok((url, None));
        }
        let (file, query) = raw.split_once(['?', '#']).map_or((raw, ""), |(f, _)| (f, &raw[f.len()..]));
        let path = self.resolve(file)?;
        if !path.exists() {
            return Err(format!("{file} does not exist in the repo; for a website pass a full https:// URL"));
        }
        let srv = super::browser::LocalServer::start(self.root.clone()).await?;
        let rel = self.rel(&path).replace('\\', "/");
        let u = format!("http://127.0.0.1:{}/{}{query}", srv.port, rel.trim_start_matches('/'));
        Ok((super::browser::parse_web_url(&u)?, Some(srv)))
    }

    fn render_opts(&self, a: &Action, default_wait: i64) -> super::browser::RenderOpts {
        let wait = a.int_arg("wait_ms").unwrap_or(default_wait).clamp(0, 30_000) as u64;
        let size = (
            a.int_arg("width").unwrap_or(1280).clamp(200, 3840) as u32,
            a.int_arg("height").unwrap_or(800).clamp(200, 2160) as u32,
        );
        super::browser::RenderOpts::new(wait, size)
    }

    /// Renders `url`; a page that did not load is an error, with a hint at
    /// the cause.
    async fn render_page(&self, url: &Url, opts: &super::browser::RenderOpts) -> Result<super::browser::Rendered, String> {
        let r = super::browser::render(url, opts).await?;
        let Some(fail) = &r.load_error else { return Ok(r) };
        let local = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
        let mut msg = format!("{url} did not load: {fail}");
        let refused = fail.contains("REFUSED") || fail.contains("CONNECTION_RESET") || fail.contains("ADDRESS_UNREACHABLE");
        if local && refused {
            msg.push_str(&refused_hint(&self.root, url));
        } else if local && fail.starts_with("HTTP 404") {
            msg.push_str("\nThe server answered 404: that path does not exist there. Check the file name, or open a repo file directly by its path (e.g. index.html).");
        }
        Err(msg)
    }

    async fn browser(&self, a: &Action) -> Result<ToolResult, String> {
        let (url, _server) = self.page_url(a).await?;
        let mut opts = self.render_opts(a, 2000);
        if let Some(s) = a.str_arg("screenshot").filter(|s| !s.trim().is_empty()) {
            opts.shots = vec![self.resolve(s)?];
        }
        let r = self.render_page(&url, &opts).await?;
        let mut out = page_report(&url, &r, true);
        if let Some(p) = r.screenshots.first() {
            out.push_str(&format!("\nScreenshot saved to {}.{}", self.rel(p).replace('\\', "/"), self.look_hint_one()));
        }
        Ok(ToolResult::ok(clip(&out, 40_000)))
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

/// What a page tool reports: where it loaded, console, failed requests,
/// facts about canvases, and (with `text`) the rendered text.
pub(super) fn page_report(url: &Url, r: &super::browser::Rendered, text: bool) -> String {
    let mut out = format!("[rendered {url}]\n");
    if r.console.is_empty() {
        out.push_str("Console: (no messages)\n");
    } else {
        out.push_str("Console:\n");
        for l in &r.console {
            out.push_str(&format!("  {l}\n"));
        }
    }
    if !r.failed.is_empty() {
        out.push_str(&format!(
            "Failed requests ({}): the page asked for these and did not get them, so what they provide (textures, models, \
             scripts, styles, data) is missing:\n",
            r.failed.len()
        ));
        for l in &r.failed {
            out.push_str(&format!("  {l}\n"));
        }
    }
    for n in &r.notes {
        out.push_str(&format!("Note: {n}\n"));
    }
    if text {
        out.push_str("\nPage text:\n");
        out.push_str(if r.text.trim().is_empty() { "(the page rendered no text)" } else { r.text.trim() });
        out.push('\n');
    }
    out
}

/// Why a localhost URL refused the connection, and what to do instead. A
/// static page needs no dev server: its file can be opened directly.
fn refused_hint(root: &Path, url: &Url) -> String {
    let procs = processes::list(root);
    let has_index = root.join("index.html").is_file();
    let dev_script = std::fs::read_to_string(root.join("package.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .is_some_and(|p| p["scripts"]["dev"].is_string() || p["scripts"]["start"].is_string());
    let mut msg = format!("\nNothing is listening at {url}.");
    if procs.is_empty() {
        msg.push_str(" No server was started in this run, so no localhost URL works yet.");
    } else {
        msg.push_str(&format!(
            " Background processes in this run: {}. Read their output with process_output for the real URL and port.",
            procs.join("; ")
        ));
    }
    if has_index && !dev_script {
        msg.push_str(" This repo is a static page: open it directly with url \"index.html\" (it is served over http for you, so ES modules work). Do not guess localhost ports.");
    } else if dev_script {
        msg.push_str(" Start the dev server with start_process (e.g. `npm run dev`), read its output with process_output for the real URL, then open that.");
    } else {
        msg.push_str(" Open a repo file directly by its path (e.g. index.html), or start the server the project needs with start_process first.");
    }
    msg
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

/// Why cmd.exe would silently mangle `command`: it runs only the first line
/// and has no heredocs, so `cat > f <<'EOF' ...` models write from habit
/// fails with an error they cannot make sense of.
fn cmd_exe_problem(command: &str) -> Option<&'static str> {
    let heredoc = command.contains("<<");
    let multiline = command.trim().contains('\n');
    (heredoc || multiline).then_some(
        "not run: commands go to cmd.exe, which runs only the first line and has no heredocs (<<EOF) or cat. \
         To create or rewrite a file, use write_file with the complete content; to run several commands, \
         join them on one line with &&.",
    )
}

/// The platform shell running `command`: `cmd /C` on Windows, `sh -c`
/// elsewhere (with `test -f/-d` translated for cmd).
fn shell_command(command: &str) -> Command {
    if cfg!(windows) {
        let mut c = Command::new("cmd");
        let trimmed = command.trim();
        // A lone `test -f path` also works without a Unix `test` on PATH.
        // Only then: in a chain (`test -f a && ...`) the rewrite would break
        // cmd's syntax.
        let lone = |rest: &str| {
            let p = rest.trim().trim_matches('"');
            (!p.is_empty() && !p.contains(|c: char| c.is_whitespace() || "&|<>;()^".contains(c))).then(|| p.to_string())
        };
        let cmd_str = if let Some(p) = trimmed.strip_prefix("test -f ").or_else(|| trimmed.strip_prefix("test -e ")).and_then(lone) {
            format!("if exist {p} (exit 0) else (exit 1)")
        } else if let Some(p) = trimmed.strip_prefix("test -d ").and_then(lone) {
            format!("if exist {p}\\ (exit 0) else (exit 1)")
        } else {
            command.to_string()
        };
        // cmd.exe does not parse MSVC-style argv escapes: pass the line
        // verbatim. `/S` strips exactly the outer quotes and keeps the rest.
        #[cfg(windows)]
        c.raw_arg(format!("/S /C \"{cmd_str}\""));
        #[cfg(not(windows))]
        c.arg("/C").arg(cmd_str);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    }
}

/// Kills a command's whole process tree when dropped while armed: on a
/// timeout, and when the run is killed and the future running the command
/// is dropped. (`kill_on_drop` alone only reaches the shell; on Windows the
/// programs it started would keep running.)
struct TreeGuard(Option<u32>);

impl Drop for TreeGuard {
    fn drop(&mut self) {
        processes::kill_tree(self.0.take());
    }
}

/// Most bytes kept per output stream; beyond it the middle is dropped (the
/// start and the end are what matter, see `clip`).
const MAX_CAPTURE: usize = 8 * 1024 * 1024;

async fn collect<R: tokio::io::AsyncRead + Unpin>(mut r: R, buf: std::sync::Arc<std::sync::Mutex<Vec<u8>>>) {
    use tokio::io::AsyncReadExt;
    let mut tmp = [0u8; 8192];
    loop {
        match r.read(&mut tmp).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let mut b = buf.lock().unwrap();
                b.extend_from_slice(&tmp[..n]);
                if b.len() > MAX_CAPTURE {
                    let (keep_head, keep_tail) = (MAX_CAPTURE / 4, MAX_CAPTURE / 2);
                    let end = b.len() - keep_tail;
                    b.drain(keep_head..end);
                }
            }
        }
    }
}

/// Runs `command` through the platform shell in `cwd`, with stdin closed
/// and a hard timeout. A timeout (or dropping the future) kills everything
/// the command spawned (`cargo test` → test binaries, `npm test` → node),
/// not just the shell: on unix through its own process group, on Windows
/// through the process tree.
pub async fn run_shell(cwd: &Path, command: &str, env: &[(String, String)], timeout: Duration) -> ShellOutput {
    let mut cmd = shell_command(command);
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

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return ShellOutput {
                exit_code: None,
                output: format!("failed to start the shell: {e}"),
                timed_out: false,
            };
        }
    };
    let mut tree = TreeGuard(child.id());
    let (out_buf, err_buf) = (std::sync::Arc::default(), std::sync::Arc::default());
    let mut readers = Vec::new();
    if let Some(o) = child.stdout.take() {
        readers.push(tokio::spawn(collect(o, std::sync::Arc::clone(&out_buf))));
    }
    if let Some(e) = child.stderr.take() {
        readers.push(tokio::spawn(collect(e, std::sync::Arc::clone(&err_buf))));
    }
    let waited = tokio::time::timeout(timeout, child.wait()).await;
    let timed_out = waited.is_err();
    if timed_out {
        // The shell is still alive here, so its whole tree can be found.
        drop(std::mem::replace(&mut tree, TreeGuard(None)));
        let _ = child.kill().await;
    } else {
        // Finished: whatever it left running in the background stays.
        tree.0 = None;
    }
    // A background program may still hold the pipes open; take what came.
    let grace = if timed_out { Duration::from_secs(1) } else { Duration::from_secs(3) };
    let _ = tokio::time::timeout(grace, async {
        for r in &mut readers {
            let _ = r.await;
        }
    })
    .await;
    for r in &readers {
        r.abort();
    }
    let text = |b: &std::sync::Arc<std::sync::Mutex<Vec<u8>>>| String::from_utf8_lossy(&b.lock().unwrap()).into_owned();
    let (stdout, stderr) = (text(&out_buf), text(&err_buf));
    let mut output = match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
        (_, true) => stdout,
        (true, false) => stderr,
        (false, false) => format!("{stdout}\n[stderr]\n{stderr}"),
    };
    match waited {
        Ok(Ok(status)) => ShellOutput { exit_code: status.code(), output, timed_out: false },
        Ok(Err(e)) => {
            ShellOutput { exit_code: None, output: format!("failed waiting for the command: {e}"), timed_out: false }
        }
        Err(_) => {
            let note = format!("command did not finish within {}s and was killed", timeout.as_secs());
            output = if output.trim().is_empty() { note } else { format!("{output}\n[{note}]") };
            ShellOutput { exit_code: None, output, timed_out: true }
        }
    }
}

fn human_size(n: u64) -> String {
    match n {
        0..1024 => format!("{n} B"),
        1024..1_048_576 => format!("{:.1} KB", n as f64 / 1024.0),
        _ => format!("{:.1} MB", n as f64 / 1_048_576.0),
    }
}

/// Background processes (dev servers, watchers) started by agents. They
/// are keyed by the worktree that started them, so a run only sees its own,
/// and are killed with their whole process tree on stop or run deletion.
pub mod processes {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::process::Stdio;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    use tokio::io::AsyncReadExt;

    const KEEP: usize = 200_000;

    struct Proc {
        root: PathBuf,
        command: String,
        started: Instant,
        pid: Option<u32>,
        output: Arc<Mutex<String>>,
        exit: Arc<Mutex<Option<String>>>,
        kill: Option<tokio::sync::oneshot::Sender<()>>,
    }

    fn table() -> &'static Mutex<HashMap<String, Proc>> {
        static T: OnceLock<Mutex<HashMap<String, Proc>>> = OnceLock::new();
        T.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn same_root(a: &Path, b: &Path) -> bool {
        super::is_same_path(a, b)
    }

    pub fn start(root: &Path, cwd: &Path, command: &str) -> Result<String, String> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let mut cmd = super::shell_command(command);
        cmd.current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .env("ORCHOPORK", "1")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("CI", "1")
            .env("BROWSER", "none");
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd.spawn().map_err(|e| format!("cannot start {command:?}: {e}"))?;
        let id = format!("p{}", NEXT.fetch_add(1, Ordering::SeqCst));
        let output = Arc::new(Mutex::new(String::new()));
        let exit = Arc::new(Mutex::new(None));
        let pid = child.id();
        if let Some(o) = child.stdout.take() {
            tokio::spawn(pump(o, output.clone()));
        }
        if let Some(e) = child.stderr.take() {
            tokio::spawn(pump(e, output.clone()));
        }
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let exit2 = exit.clone();
        tokio::spawn(async move {
            tokio::select! {
                r = child.wait() => {
                    *exit2.lock().unwrap() = Some(match r {
                        Ok(s) => s.code().map_or("killed by a signal".into(), |c| format!("exited with code {c}")),
                        Err(e) => format!("failed: {e}"),
                    });
                }
                _ = rx => {
                    kill_tree(pid);
                    let _ = child.kill().await;
                    *exit2.lock().unwrap() = Some("stopped".into());
                }
            }
        });
        table().lock().unwrap().insert(
            id.clone(),
            Proc {
                root: root.to_path_buf(),
                command: command.to_string(),
                started: Instant::now(),
                pid,
                output,
                exit,
                kill: Some(tx),
            },
        );
        Ok(id)
    }

    async fn pump<R: tokio::io::AsyncRead + Unpin>(mut r: R, out: Arc<Mutex<String>>) {
        let mut buf = [0u8; 8192];
        loop {
            match r.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let mut o = out.lock().unwrap();
                    o.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if o.len() > KEEP * 2 {
                        let cut = o.len() - KEEP;
                        let cut = (cut..o.len()).find(|&i| o.is_char_boundary(i)).unwrap_or(o.len());
                        o.drain(..cut);
                    }
                }
            }
        }
    }

    /// (status, last `tail` chars of output).
    pub fn output(root: &Path, id: &str, tail: usize) -> Option<(String, String)> {
        let t = table().lock().unwrap();
        let p = t.get(id).filter(|p| same_root(&p.root, root))?;
        let out = p.output.lock().unwrap();
        let start = out.len().saturating_sub(tail);
        let start = (start..out.len()).find(|&i| out.is_char_boundary(i)).unwrap_or(out.len());
        Some((status(p), out[start..].to_string()))
    }

    fn status(p: &Proc) -> String {
        match p.exit.lock().unwrap().clone() {
            Some(e) => e,
            None => format!("running for {}s (pid {})", p.started.elapsed().as_secs(), p.pid.unwrap_or(0)),
        }
    }

    pub fn list(root: &Path) -> Vec<String> {
        let t = table().lock().unwrap();
        let mut v: Vec<(&String, &Proc)> = t.iter().filter(|(_, p)| same_root(&p.root, root)).collect();
        v.sort_by_key(|(id, _)| id[1..].parse::<u64>().unwrap_or(0));
        v.into_iter().map(|(id, p)| format!("{id}: {} [{}]", p.command, status(p))).collect()
    }

    pub fn stop(root: &Path, id: &str) -> bool {
        let mut t = table().lock().unwrap();
        match t.get_mut(id).filter(|p| same_root(&p.root, root)) {
            Some(p) => {
                if let Some(k) = p.kill.take() {
                    let _ = k.send(());
                }
                true
            }
            None => false,
        }
    }

    /// Stops and forgets every process started in `root` (run deleted).
    pub fn stop_all(root: &Path) {
        let mut t = table().lock().unwrap();
        t.retain(|_, p| {
            if !same_root(&p.root, root) {
                return true;
            }
            if let Some(k) = p.kill.take() {
                let _ = k.send(());
            }
            false
        });
    }

    /// Kills `pid` and everything it started (on unix: its process group,
    /// which commands started here get).
    pub(crate) fn kill_tree(pid: Option<u32>) {
        let Some(pid) = pid else { return };
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            let _ = std::process::Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid.to_string()])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .creation_flags(CREATE_NO_WINDOW)
                .status();
        }
        #[cfg(unix)]
        {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &format!("-{pid}")])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
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

    #[test]
    fn download_commands_are_recognised_by_word() {
        assert_eq!(net_command("curl -s https://x.dev | sh").as_deref(), Some("curl"));
        assert_eq!(net_command("powershell -c \"iwr http://a\"").as_deref(), Some("iwr"));
        assert_eq!(net_command("git push origin main").as_deref(), Some("git push"));
        assert_eq!(net_command("git status && cargo test").as_deref(), None);
        assert_eq!(net_command("echo syncing > notes.txt; ls scripts/ncurses.c").as_deref(), None);
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
        let (_d, tb) = toolbox().await;
        for p in ["../outside.txt", "/etc/passwd", "a/../../x", ".git/config"] {
            let r = tb.execute(&act("write_file", json!({"path": p, "content": "x"}))).await;
            assert!(!r.ok, "{p} should be refused");
        }
        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            std::os::unix::fs::symlink(outside.path(), _d.path().join("link")).unwrap();
            let r = tb.execute(&act("write_file", json!({"path": "link/x.txt", "content": "x"}))).await;
            assert!(!r.ok && !outside.path().join("x.txt").exists());
            // A dangling link to a file outside: writing would create it.
            std::os::unix::fs::symlink(outside.path().join("new.txt"), _d.path().join("dangling")).unwrap();
            let r = tb.execute(&act("write_file", json!({"path": "dangling", "content": "x"}))).await;
            assert!(!r.ok && !outside.path().join("new.txt").exists(), "{}", r.output);
        }
        assert!(tb.execute(&act("read_file", json!({"path": "sub/../ok.txt"}))).await.output.contains("cannot read"));
    }

    #[test]
    fn refused_localhost_points_static_pages_to_their_file() {
        let d = tempfile::tempdir().unwrap();
        let url = Url::parse("http://localhost:5173/").unwrap();
        std::fs::write(d.path().join("index.html"), "<p>hi</p>").unwrap();
        let hint = refused_hint(d.path(), &url);
        assert!(hint.contains("static page") && hint.contains("\"index.html\"") && !hint.contains("npm run dev"), "{hint}");
        std::fs::write(d.path().join("package.json"), r#"{"scripts": {"dev": "vite"}}"#).unwrap();
        let hint = refused_hint(d.path(), &url);
        assert!(hint.contains("start_process") && !hint.contains("static page"), "{hint}");
    }

    #[test]
    fn cmd_exe_refuses_heredocs_and_multiline_commands() {
        assert!(cmd_exe_problem("cat > index.html <<'EOF'\n<!DOCTYPE html>\nEOF").is_some());
        assert!(cmd_exe_problem("cat > index.html <<EOF").is_some());
        assert!(cmd_exe_problem("npm install\nnpm test").is_some());
        assert!(cmd_exe_problem("npm install && npm test\n").is_none());
        assert!(cmd_exe_problem("git status").is_none());
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

    #[test]
    fn apply_edit_handles_crlf_and_lf() {
        let text = "<html>\r\n  <head>\r\n    <title>Test</title>\r\n  </head>\r\n</html>\r\n";
        let old = "  <head>\n    <title>Test</title>\n  </head>";
        let new = "  <head>\n    <title>Water Sim</title>\n  </head>";
        let res = apply_edit(text, old, new).expect("should match despite CRLF/LF");
        assert!(res.contains("<title>Water Sim</title>"));
        assert!(res.contains("\r\n"), "should preserve CRLF");
    }

    #[test]
    fn apply_edit_handles_whitespace_and_indentation() {
        let text = "function init() {\n    const a = 1;\n    const b = 2;\n    return a + b;\n}\n";
        // Agent provided 2 spaces instead of 4 spaces
        let old = "  const b = 2;\n  return a + b;";
        let new = "    const b = 42;\n    return a * b;";
        let res = apply_edit(text, old, new).expect("should match with relaxed indentation");
        assert!(res.contains("const b = 42;"));
    }

    #[test]
    fn apply_edit_handles_read_file_line_numbers() {
        let text = "line one\nline two\nline three\n";
        // Agent accidentally copied line numbers from read_file
        let old = "    2| line two";
        let new = "line two updated";
        let res = apply_edit(text, old, new).expect("should strip read_file line numbers");
        assert!(res.contains("line two updated"));
    }

    #[tokio::test]
    async fn move_delete_list_dir_and_replace_all() {
        let (d, tb) = toolbox().await;
        assert!(tb.execute(&act("write_file", json!({"path": "a/x.txt", "content": "foo foo\n"}))).await.ok);
        let e = tb.execute(&act("edit_file", json!({"path": "a/x.txt", "old": "foo", "new": "bar", "replace_all": true}))).await;
        assert!(e.ok && e.output.contains("2 replacements"), "{}", e.output);
        let m = tb.execute(&act("move_file", json!({"from": "a/x.txt", "to": "b/y.txt"}))).await;
        assert!(m.ok, "{}", m.output);
        assert_eq!(std::fs::read_to_string(d.path().join("b/y.txt")).unwrap(), "bar bar\n");
        let l = tb.execute(&act("list_dir", json!({"path": "."}))).await;
        assert!(l.output.contains("a/") && l.output.contains("b/") && !l.output.contains(".git"), "{}", l.output);
        let refused = tb.execute(&act("delete_file", json!({"path": "b"}))).await;
        assert!(!refused.ok && refused.output.contains("recursive"));
        assert!(tb.execute(&act("delete_file", json!({"path": "b", "recursive": true}))).await.ok);
        assert!(!d.path().join("b").exists());
        assert!(!tb.execute(&act("delete_file", json!({"path": "."}))).await.ok, "the root is never deleted");
        assert!(!tb.execute(&act("move_file", json!({"from": "a", "to": "../escape"}))).await.ok);
    }

    /// cmd.exe does not follow the MSVC argv quoting rules, so the command
    /// line must reach it verbatim: quotes and `&&` keep their meaning.
    #[tokio::test]
    async fn commands_with_quotes_reach_the_shell_verbatim() {
        let (_d, tb) = toolbox().await;
        let r = tb.execute(&act("run_command", json!({"command": r#"echo "a  b"&& echo second"#}))).await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.contains(r#""a  b""#) && r.output.contains("second"), "{}", r.output);
        assert!(!r.output.contains(r#"\""#), "no MSVC-style escapes: {}", r.output);
    }

    /// A timed-out command keeps the output it wrote before the timeout, and
    /// the programs it started die with it.
    #[tokio::test]
    async fn timed_out_commands_keep_partial_output_and_are_killed() {
        let (d, tb) = toolbox().await;
        let marker = d.path().join("still-running.txt");
        let cmd = if cfg!(windows) {
            // The grandchild itself writes the marker (not cmd).
            format!(
                "echo started&& powershell -NoProfile -Command \"Start-Sleep -Seconds 5; Set-Content -Path '{}' -Value x\"",
                marker.display()
            )
        } else {
            format!("echo started; sleep 5; echo x > '{}'", marker.display())
        };
        let t = std::time::Instant::now();
        let r = tb.execute(&act("run_command", json!({"command": cmd}))).await;
        assert!(!r.ok && r.output.contains("timed out") && r.output.contains("started"), "{}", r.output);
        assert!(t.elapsed() < Duration::from_secs(5), "returned at the timeout, took {:?}", t.elapsed());
        tokio::time::sleep(Duration::from_secs(5)).await;
        assert!(!marker.exists(), "the command's children were killed with it");
    }

    #[tokio::test]
    async fn commands_take_a_working_directory() {
        let (d, tb) = toolbox().await;
        std::fs::create_dir(d.path().join("sub")).unwrap();
        let cmd = if cfg!(windows) { "cd" } else { "pwd" };
        let r = tb.execute(&act("run_command", json!({"command": cmd, "cwd": "sub"}))).await;
        assert!(r.ok && r.output.trim_end().ends_with("sub"), "{}", r.output);
        assert!(!tb.execute(&act("run_command", json!({"command": cmd, "cwd": "../.."}))).await.ok);
    }

    #[tokio::test]
    async fn background_processes_are_listed_read_and_scoped_to_their_run() {
        let (_d, tb) = toolbox().await;
        let s = tb.execute(&act("start_process", json!({"command": "echo started-ok", "wait_secs": 2}))).await;
        assert!(s.ok && s.output.contains("started-ok"), "{}", s.output);
        let id = s.output.split_whitespace().nth(2).unwrap().trim_end_matches(':').to_string();
        let o = tb.execute(&act("process_output", json!({"id": id}))).await;
        assert!(o.output.contains("exited with code 0"), "{}", o.output);
        assert!(tb.execute(&act("process_output", json!({}))).await.output.contains(&id));
        let (_d2, other) = toolbox().await;
        assert!(!other.execute(&act("process_output", json!({"id": id}))).await.ok, "other runs cannot see it");
        assert!(tb.execute(&act("stop_process", json!({"id": id}))).await.ok);
    }

    #[tokio::test]
    async fn tool_table_is_consistent() {
        let (_d, tb) = toolbox().await;
        let schemas = tb.schemas_with_finish();
        let mut names: Vec<&str> = schemas.iter().map(|t| t.name.as_str()).collect();
        let desc = tb.describe_with_finish();
        for n in &names {
            assert!(desc.contains(&format!("- {n} ")), "{n} missing from the prompt");
        }
        names.sort();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len, "duplicate tool names");
        assert_eq!(tb.class("read_file"), ToolClass::Read);
        assert_eq!(tb.class("edit_file"), ToolClass::Write);
        assert_eq!(tb.class("run_command"), ToolClass::Exec);
        assert_eq!(tb.class("some_skill_tool"), ToolClass::Exec);
        assert_eq!(tb.class("finish"), ToolClass::Control);
        for t in BUILTIN {
            assert!(crate::skills::model::RESERVED_TOOLS.contains(&t.name), "{} must be reserved for skills", t.name);
        }
        assert!(!ToolClass::Exec.needs_approval(Autonomy::Full));
        assert!(!ToolClass::Write.needs_approval(Autonomy::Semi) && ToolClass::Exec.needs_approval(Autonomy::Semi));
        assert!(ToolClass::Write.needs_approval(Autonomy::Manual) && !ToolClass::Read.needs_approval(Autonomy::Manual));

        let limits = Limits { allow_commands: false, allow_network: false, ..Limits::default() };
        let locked = ToolBox::new(_d.path(), &limits, vec![]).unwrap();
        let names: Vec<String> = locked.schemas().into_iter().map(|t| t.name).collect();
        assert!(!names.iter().any(|n| n == "run_command" || n == "browser" || n == "web_fetch"), "{names:?}");
    }

    #[tokio::test]
    async fn internet_levels_decide_which_web_tools_exist_and_work() {
        let d = tempfile::tempdir().unwrap();
        let names = |limits: &Limits, own: Option<NetAccess>| -> Vec<String> {
            ToolBox::new(d.path(), limits, vec![]).unwrap().with_network(own).schemas().into_iter().map(|t| t.name).collect()
        };
        let has = |v: &[String], n: &str| v.iter().any(|x| x == n);
        let web = ["web_search", "web_fetch"];
        let pages = ["browser", "screenshot_page", "screenshot_sequence"];
        let l = Limits::default();
        let off = names(&l, Some(NetAccess::Off));
        assert!(web.iter().chain(&pages).all(|n| !has(&off, n)), "{off:?}");
        let local = names(&l, Some(NetAccess::Local));
        assert!(pages.iter().all(|n| has(&local, n)) && web.iter().all(|n| !has(&local, n)), "{local:?}");
        let docs = names(&l, Some(NetAccess::Docs));
        assert!(pages.iter().chain(&web).all(|n| has(&docs, n)), "{docs:?}");
        assert_eq!(names(&l, None), names(&l, Some(NetAccess::Full)));
        // The workspace is the ceiling: an agent cannot ask for more.
        let capped = Limits { network: NetAccess::Local, ..Limits::default() };
        let v = names(&capped, Some(NetAccess::Full));
        assert!(has(&v, "browser") && !has(&v, "web_fetch"), "{v:?}");

        // A model that names a tool it was not offered is refused.
        let tb = ToolBox::new(d.path(), &l, vec![]).unwrap().with_network(Some(NetAccess::Off));
        let r = tb.execute(&act("web_fetch", json!({"url": "https://example.com"}))).await;
        assert!(!r.ok && r.output.contains("internet access"), "{}", r.output);
        let r = tb.execute(&act("browser", json!({"url": "index.html"}))).await;
        assert!(!r.ok && r.output.contains("internet access"), "{}", r.output);
        // `docs` reads documentation but opens no outside page in the browser.
        let tb = ToolBox::new(d.path(), &l, vec![]).unwrap().with_network(Some(NetAccess::Docs));
        let r = tb.execute(&act("browser", json!({"url": "https://example.com/"}))).await;
        assert!(!r.ok && r.output.contains("on the internet"), "{}", r.output);
        // Downloads by shell are refused without internet.
        let tb = ToolBox::new(d.path(), &l, vec![]).unwrap().with_network(Some(NetAccess::Off));
        let r = tb.execute(&act("run_command", json!({"command": "curl -s https://example.com"}))).await;
        assert!(!r.ok && r.output.contains("curl"), "{}", r.output);
        let r = tb.execute(&act("start_process", json!({"command": "wget https://example.com/x"}))).await;
        assert!(!r.ok && r.output.contains("wget"), "{}", r.output);
    }

    /// Needs Edge or Chrome: `cargo test headless -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn headless_browser_renders_repo_pages_with_modules() {
        let (d, tb) = toolbox().await;
        std::fs::write(d.path().join("index.html"), r#"<html><body><p id="s">static</p><script type="module" src="main.js"></script></body></html>"#).unwrap();
        std::fs::write(d.path().join("main.js"), r#"document.getElementById("s").textContent = "rendered by module"; console.error("boom from module");"#).unwrap();
        let r = tb.execute(&act("browser", json!({"url": "index.html", "screenshot": "shot.png", "wait_ms": 1000}))).await;
        assert!(r.ok, "{}", r.output);
        assert!(r.output.contains("rendered by module"), "{}", r.output);
        assert!(r.output.contains("boom from module"), "{}", r.output);
        assert!(d.path().join("shot.png").is_file(), "{}", r.output);
    }

    /// Needs Edge or Chrome (skips otherwise).
    #[tokio::test]
    async fn screenshot_sequences_tell_moving_pages_from_still_ones() {
        if super::super::browser::find_browser().is_none() {
            eprintln!("skipped: no Edge/Chrome here");
            return;
        }
        let (d, tb) = toolbox().await;
        std::fs::write(
            d.path().join("moving.html"),
            r#"<body style="margin:0"><div id="b" style="position:absolute;top:50px;width:80px;height:80px;background:red"></div>
<script>const t0 = performance.now(); (function f(){ document.getElementById("b").style.left = ((performance.now() - t0) / 5 % 600) + "px"; requestAnimationFrame(f); })();</script></body>"#,
        )
        .unwrap();
        std::fs::write(d.path().join("still.html"), "<h1>still</h1>").unwrap();
        let moving = tb.execute(&act("screenshot_sequence", json!({"url": "moving.html", "count": 3, "interval_ms": 700, "name": "m"}))).await;
        assert!(moving.ok, "{}", moving.output);
        assert!(moving.output.contains("every frame differs"), "{}", moving.output);
        for i in 1..=3 {
            assert!(d.path().join(format!(".orchopork-shots/m-{i}.png")).is_file());
        }
        let still = tb.execute(&act("screenshot_sequence", json!({"url": "still.html", "count": 3, "interval_ms": 300, "name": "s"}))).await;
        assert!(still.output.contains("nothing changed"), "{}", still.output);
    }
}
