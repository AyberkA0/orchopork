//! The actor's action protocol: one JSON object per reply,
//! `{"tool": "<name>", "args": {...}}`.
//!
//! A text protocol instead of each provider's native tool-calling API:
//! it behaves the same on every backend (Ollama models without a tool
//! template, llama.cpp, Claude, OpenAI-compatible APIs), keeps the
//! transcript plain text, and is parsed leniently because small local
//! models get the details wrong (fences or not, `name`/`arguments` instead
//! of `tool`/`args`, raw newlines inside JSON strings, trailing commas,
//! single quotes, comments, or unescaped quotes in code snippets).

use serde_json::{Map, Value, json};

#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub tool: String,
    pub args: Map<String, Value>,
}

impl Action {
    pub fn str_arg(&self, key: &str) -> Option<&str> {
        self.args.get(key).and_then(Value::as_str)
    }

    pub fn int_arg(&self, key: &str) -> Option<i64> {
        self.args.get(key).and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())))
    }
}

const TOOL_KEYS: &[&str] = &["tool", "name", "action", "tool_name"];
const ARG_KEYS: &[&str] = &["args", "arguments", "parameters", "input", "params"];

/// Finds the *last* JSON object or block in `text` that names a tool.
/// Reasoning before it may contain other braces or example snippets.
/// The tool name is canonicalized (see `canonical_tool`).
pub fn parse_action(text: &str) -> Result<Action, String> {
    parse_action_raw(text).map(|mut a| {
        a.tool = canonical_tool(&a.tool);
        a
    })
}

/// Maps the tool names models actually produce onto ours: namespaced names
/// from their training (`repo_browser.write_file`, `functions.read_file`,
/// `container.exec`), other harnesses' names (`open_file`, `str_replace`,
/// `bash`), and case/space/dash variants. Names we do not recognize (skill
/// tools) are returned trimmed but otherwise unchanged.
pub fn canonical_tool(raw: &str) -> String {
    let t = raw.trim().trim_matches(|c| c == '"' || c == '\'' || c == '`').trim();
    let base = t.rsplit(['.', ':', '/']).next().unwrap_or(t).trim();
    let norm: String = base.to_lowercase().chars().map(|c| if c == ' ' || c == '-' { '_' } else { c }).collect();
    let alias = match norm.as_str() {
        "open_file" | "open" | "cat" | "view" | "view_file" | "readfile" | "read" | "get_file" | "file_read" => "read_file",
        "create_file" | "writefile" | "write" | "save_file" | "file_write" | "create" => "write_file",
        "str_replace" | "str_replace_editor" | "replace" | "replace_in_file" | "editfile" | "edit" | "modify_file" => "edit_file",
        "print_tree" | "tree" | "listfiles" | "list" | "find_files" | "glob" => "list_files",
        "ls" | "listdir" | "list_directory" => "list_dir",
        "grep" | "find" | "search_files" | "code_search" | "ripgrep" | "rg" => "search",
        "exec" | "shell" | "bash" | "run" | "execute" | "terminal" | "run_shell" | "cmd" | "powershell" | "command" => "run_command",
        "rm" | "remove_file" | "delete" | "unlink" => "delete_file",
        "mv" | "rename" | "rename_file" | "move" => "move_file",
        "fetch" | "http_get" | "open_url" | "get_url" => "web_fetch",
        "websearch" | "google" | "search_web" | "internet_search" => "web_search",
        "browse" | "open_browser" | "navigate" | "screenshot" => "browser",
        "done" | "complete" | "final" | "final_answer" | "task_complete" | "submit" => "finish",
        _ => "",
    };
    if !alias.is_empty() {
        return alias.to_string();
    }
    if KNOWN_TOOLS.contains(&norm.as_str()) {
        return norm;
    }
    base.to_string()
}

fn parse_action_raw(text: &str) -> Result<Action, String> {
    let objects = json_objects(text);
    for obj in objects.into_iter().rev() {
        if let Some(a) = to_action(obj) {
            return Ok(a);
        }
    }

    // Try fenced blocks with heuristic action extraction
    for fence in extract_fenced_blocks(text).into_iter().rev() {
        if let Some(a) = heuristic_action(&fence) {
            return Ok(a);
        }
    }

    // Try full text heuristic extraction (handles unescaped internal quotes like <script src="x">)
    if let Some(a) = heuristic_action(text) {
        return Ok(a);
    }

    // Plain command lines for read-only tools: `read_file src/a.rs`,
    // `read file water_shader.frag`, `list_files(path="src")`.
    if let Some(a) = loose_read_call(text) {
        return Ok(a);
    }

    if !text.contains('{') {
        Err("no JSON tool call found in the reply. Every turn requires a tool call JSON object, e.g. {\"tool\": \"read_file\", \"args\": {\"path\": \"README.md\"}}. Do not reply with text only.".into())
    } else {
        Err("found JSON or braces in reply, but could not parse a valid tool call. Ensure JSON is well-formed with a \"tool\" field and valid arguments.".into())
    }
}

/// The critic's verdict: `{"verdict": "approve"|"revise", "feedback": "..."}`.
/// Falls back to keywords, and treats anything unclear as "revise" with the
/// whole reply as feedback (never silently approves).
pub fn parse_verdict(text: &str) -> (bool, String) {
    for obj in json_objects(text).into_iter().rev() {
        if let Some(v) = obj.get("verdict").and_then(Value::as_str) {
            let feedback = obj.get("feedback").and_then(Value::as_str).unwrap_or("").trim().to_string();
            return (v.trim().eq_ignore_ascii_case("approve"), feedback);
        }
    }
    // Also check heuristic
    if let Some(v) = extract_field_value(text, "verdict", &["feedback"]) {
        let feedback = extract_field_value(text, "feedback", &[]).unwrap_or_default();
        return (v.trim().eq_ignore_ascii_case("approve"), feedback);
    }
    let upper = text.to_uppercase();
    let approve = upper.contains("VERDICT: APPROVE") || upper.trim() == "APPROVE";
    (approve, text.trim().to_string())
}

fn to_action(obj: Map<String, Value>) -> Option<Action> {
    let (tool_key, tool) = TOOL_KEYS.iter().find_map(|k| obj.get(*k).and_then(Value::as_str).map(|t| (*k, t)))?;
    let tool = tool.trim().to_string();
    if tool.is_empty() {
        return None;
    }
    let explicit = ARG_KEYS.iter().find_map(|k| obj.get(*k));
    let args = match explicit {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        },
        _ => {
            // Flat form: {"tool": "read_file", "path": "x"}.
            obj.iter().filter(|(k, _)| k.as_str() != tool_key).map(|(k, v)| (k.clone(), v.clone())).collect()
        }
    };
    Some(Action { tool, args })
}

/// Every top-level `{...}` span that parses as a JSON object, in order.
pub(crate) fn json_objects(text: &str) -> Vec<Map<String, Value>> {
    let mut list = Vec::new();

    // 1. Spans from balanced_spans
    for span in balanced_spans(text) {
        if let Some(m) = parse_json_object(span) {
            list.push(m);
        }
    }

    // 2. Fenced blocks
    for block in extract_fenced_blocks(text) {
        if let Some(m) = parse_json_object(&block)
            && !list.contains(&m) {
                list.push(m);
            }
    }

    // 3. Outermost braces fallback if empty
    if list.is_empty()
        && let (Some(s), Some(e)) = (text.find('{'), text.rfind('}'))
            && s < e
                && let Some(m) = parse_json_object(&text[s..=e]) {
                    list.push(m);
                }

    list
}

fn parse_json_object(span: &str) -> Option<Map<String, Value>> {
    let span = span.trim();
    if span.is_empty() {
        return None;
    }
    let target = if !span.starts_with('{') || !span.ends_with('}') {
        if let (Some(s), Some(e)) = (span.find('{'), span.rfind('}')) {
            if s < e { &span[s..=e] } else { span }
        } else {
            span
        }
    } else {
        span
    };

    if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(target) {
        return Some(m);
    }
    let rep = repair(target);
    if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&rep) {
        return Some(m);
    }
    let san = sanitize_json(target);
    if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&san) {
        return Some(m);
    }
    let rep_san = repair(&san);
    if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&rep_san) {
        return Some(m);
    }
    if let Ok(y) = serde_yaml::from_str::<serde_yaml::Value>(target)
        && let Value::Object(m) = yaml_to_json(y) {
            return Some(m);
        }
    if let Ok(y) = serde_yaml::from_str::<serde_yaml::Value>(&rep)
        && let Value::Object(m) = yaml_to_json(y) {
            return Some(m);
        }
    None
}

fn yaml_to_json(v: serde_yaml::Value) -> Value {
    match v {
        serde_yaml::Value::Null => Value::Null,
        serde_yaml::Value::Bool(b) => Value::Bool(b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(i.into())
            } else if let Some(f) = n.as_f64() {
                serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
        serde_yaml::Value::String(s) => Value::String(s),
        serde_yaml::Value::Sequence(seq) => Value::Array(seq.into_iter().map(yaml_to_json).collect()),
        serde_yaml::Value::Mapping(map) => {
            let mut m = Map::new();
            for (k, v) in map {
                if let serde_yaml::Value::String(ks) = k {
                    m.insert(ks, yaml_to_json(v));
                }
            }
            Value::Object(m)
        }
        serde_yaml::Value::Tagged(tagged) => yaml_to_json(tagged.value),
    }
}

pub(crate) fn extract_fenced_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut pos = 0;
    while let Some(start_idx) = text[pos..].find("```") {
        let abs_start = pos + start_idx;
        let rest = &text[abs_start + 3..];
        let line_end = rest.find('\n').unwrap_or(rest.len());
        let content_start = abs_start + 3 + line_end;
        if content_start >= text.len() {
            break;
        }
        if let Some(end_idx) = text[content_start..].find("```") {
            let content = &text[content_start..content_start + end_idx];
            blocks.push(content.trim().to_string());
            pos = content_start + end_idx + 3;
        } else {
            let content = &text[content_start..];
            blocks.push(content.trim().to_string());
            break;
        }
    }
    blocks
}

/// Top-level brace-balanced spans, respecting JSON string quoting so braces
/// inside strings (code in a `write_file`) do not end a span early.
fn balanced_spans(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let start = i;
        let (mut depth, mut in_str, mut esc) = (0usize, false, false);
        let mut end = None;
        while i < bytes.len() {
            let b = bytes[i];
            if in_str {
                if esc {
                    esc = false;
                } else if b == b'\\' {
                    esc = true;
                } else if b == b'"' {
                    in_str = false;
                }
            } else if b == b'"' {
                in_str = true;
            } else if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            i += 1;
        }
        match end {
            Some(e) => {
                spans.push(&text[start..=e]);
                i = e + 1;
            }
            None => i = start + 1,
        }
    }
    spans
}

/// Strips comments (`//`, `/* */`), trailing commas before `}` or `]`,
/// and normalizes single quotes to double quotes outside strings.
fn sanitize_json(span: &str) -> String {
    let mut out = String::with_capacity(span.len());
    let chars: Vec<char> = span.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut quote_char = '"';

    while i < n {
        let c = chars[i];
        if in_str {
            if esc {
                esc = false;
                if c == '\'' {
                    // `\'` is not a JSON escape: keep just the quote.
                    out.pop();
                }
                out.push(c);
            } else if c == '"' && quote_char == '\'' {
                // A double quote inside a single-quoted string.
                out.push_str("\\\"");
            } else if c == '\\' {
                esc = true;
                out.push(c);
            } else if c == quote_char {
                in_str = false;
                out.push('"');
            } else {
                out.push(c);
            }
            i += 1;
            continue;
        }

        // Outside strings:
        if c == '"' || c == '\'' {
            in_str = true;
            quote_char = c;
            out.push('"');
            i += 1;
            continue;
        }

        // Single line comment //
        if c == '/' && i + 1 < n && chars[i + 1] == '/' {
            i += 2;
            while i < n && chars[i] != '\n' && chars[i] != '\r' {
                i += 1;
            }
            continue;
        }

        // Multi line comment /* ... */
        if c == '/' && i + 1 < n && chars[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(chars[i] == '*' && chars[i + 1] == '/') {
                i += 1;
            }
            i += 2;
            continue;
        }

        // Trailing commas: `,` followed only by whitespace and then `}` or `]`
        if c == ',' {
            let mut j = i + 1;
            while j < n && chars[j].is_whitespace() {
                j += 1;
            }
            if j < n && (chars[j] == '}' || chars[j] == ']') {
                i += 1;
                continue;
            }
        }

        out.push(c);
        i += 1;
    }

    out
}

/// Escapes raw control characters inside JSON strings — the most common
/// way local models break JSON when writing multi-line file content.
fn repair(span: &str) -> String {
    let mut out = String::with_capacity(span.len() + 16);
    let (mut in_str, mut esc) = (false, false);
    for c in span.chars() {
        if in_str {
            if esc {
                esc = false;
                out.push(c);
                continue;
            }
            match c {
                '\\' => {
                    esc = true;
                    out.push(c);
                }
                '"' => {
                    in_str = false;
                    out.push(c);
                }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        } else {
            if c == '"' {
                in_str = true;
            }
            out.push(c);
        }
    }
    out
}

fn unescape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

const KNOWN_TOOLS: &[&str] = &[
    "read_file",
    "edit_file",
    "write_file",
    "list_files",
    "list_dir",
    "move_file",
    "delete_file",
    "search",
    "run_command",
    "start_process",
    "process_output",
    "stop_process",
    "web_search",
    "web_fetch",
    "browser",
    "finish",
    "delegate",
];

/// Read-only calls written as a command line instead of JSON. Only reads:
/// guessing the arguments of a write from prose would be unsafe.
fn loose_read_call(text: &str) -> Option<Action> {
    for line in text.lines().rev() {
        let l = line.trim().trim_matches(|c| c == '`' || c == '*' || c == '>').trim();
        let lower = l.to_ascii_lowercase();
        for (tool, spellings) in [
            ("read_file", &["read_file", "read file", "readfile", "read"][..]),
            ("list_files", &["list_files", "list files", "listfiles"][..]),
            ("list_dir", &["list_dir", "list dir", "ls"][..]),
        ] {
            let Some(sp) = spellings.iter().find(|sp| {
                lower.starts_with(**sp)
                    && lower[sp.len()..].chars().next().is_none_or(|c| c.is_whitespace() || c == ':' || c == '(')
            }) else {
                continue;
            };
            let rest = l[sp.len()..]
                .trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == '(')
                .trim_end_matches(|c: char| c.is_whitespace() || c == ')' || c == ';' || c == '.')
                .trim();
            let rest = rest.strip_prefix("path=").or_else(|| rest.strip_prefix("path =")).unwrap_or(rest).trim();
            let path = rest.trim_matches(|c| c == '"' || c == '\'' || c == '`');
            // One path-looking token, nothing else.
            if path.is_empty() && tool == "read_file" {
                continue;
            }
            if path.contains(char::is_whitespace) || (tool == "read_file" && !path.contains(['.', '/', '\\'])) {
                continue;
            }
            let mut args = Map::new();
            if !path.is_empty() {
                args.insert("path".into(), Value::String(path.to_string()));
            }
            return Some(Action { tool: tool.into(), args });
        }
    }
    None
}

fn find_tool_name(text: &str) -> Option<String> {
    for key in TOOL_KEYS {
        for quote in &["\"", "'"] {
            let pattern = format!("{quote}{key}{quote}");
            if let Some(pos) = text.find(&pattern) {
                let after = &text[pos + pattern.len()..];
                let mut chars = after.char_indices();
                let mut found_colon = false;
                for (_, c) in chars.by_ref() {
                    if c == ':' {
                        found_colon = true;
                        break;
                    } else if !c.is_whitespace() {
                        break;
                    }
                }
                if found_colon {
                    let after_colon = &after[chars.next().map(|(i, _)| i).unwrap_or(after.len())..];
                    let trimmed = after_colon.trim_start();
                    let val = if trimmed.starts_with('"') || trimmed.starts_with('\'') {
                        let q = trimmed.chars().next().unwrap();
                        let rest = &trimmed[1..];
                        if let Some(end) = rest.find(q) {
                            &rest[..end]
                        } else {
                            rest.split_whitespace().next().unwrap_or("")
                        }
                    } else {
                        trimmed.split(|c: char| c.is_whitespace() || c == ',' || c == '}').next().unwrap_or("")
                    };
                    let val = val.trim().trim_matches(|c| c == '"' || c == '\'');
                    if !val.is_empty() {
                        return Some(val.to_string());
                    }
                }
            }
        }
    }

    // Known tools fallback
    for known in KNOWN_TOOLS {
        if text.contains(&format!("\"{known}\"")) || text.contains(&format!("'{known}'")) {
            return Some((*known).to_string());
        }
    }
    None
}

fn extract_field_value(text: &str, key: &str, boundary_keys: &[&str]) -> Option<String> {
    for quote in &["\"", "'", ""] {
        let pattern = if quote.is_empty() {
            format!("{key}:")
        } else {
            format!("{quote}{key}{quote}")
        };
        let Some(pos) = text.find(&pattern) else { continue };
        let after = &text[pos + pattern.len()..];
        let mut chars = after.char_indices();
        let mut found_colon = quote.is_empty();
        if !found_colon {
            for (_, c) in chars.by_ref() {
                if c == ':' {
                    found_colon = true;
                    break;
                } else if !c.is_whitespace() {
                    break;
                }
            }
        }
        if !found_colon {
            continue;
        }
        let after_colon = after[chars.next().map(|(i, _)| i).unwrap_or(0)..].trim_start();
        if after_colon.is_empty() {
            continue;
        }

        let (mut raw_val, is_quoted) = if after_colon.starts_with('"') || after_colon.starts_with('\'') {
            let q = after_colon.chars().next().unwrap();
            let content_after_q = &after_colon[1..];
            let mut end_pos = None;
            for bk in boundary_keys {
                for bq in &["\"", "'"] {
                    let bk_pat = format!("{bq}{bk}{bq}");
                    if let Some(bk_idx) = content_after_q.find(&bk_pat) {
                        let before_bk = &content_after_q[..bk_idx];
                        let cand_end = before_bk.rfind(',').unwrap_or(bk_idx);
                        end_pos = Some(end_pos.map_or(cand_end, |cur: usize| cur.min(cand_end)));
                    }
                }
            }

            let slice: &str = if let Some(e) = end_pos {
                let s = content_after_q[..e].trim_end();
                if s.ends_with(q) {
                    &s[..s.len() - 1]
                } else {
                    s
                }
            } else {
                // Find closing quote before `}`:
                // Scan content_after_q for a quote `q` whose subsequent characters
                // are only whitespace, newlines, and/or closing braces `}` or fences.
                let mut found_slice = None;
                for (idx, c) in content_after_q.char_indices() {
                    if c == q {
                        let remainder = &content_after_q[idx + 1..];
                        let is_end = remainder
                            .chars()
                            .take_while(|rc| rc.is_whitespace() || *rc == '}' || *rc == '`')
                            .any(|rc| rc == '}' || rc == '`')
                            || remainder.trim().is_empty();
                        if is_end {
                            found_slice = Some(&content_after_q[..idx]);
                            break;
                        }
                    }
                }
                if let Some(s) = found_slice {
                    s
                } else if let Some(rbrace) = content_after_q.find('}') {
                    let s = content_after_q[..rbrace].trim_end();
                    if s.ends_with(q) {
                        &s[..s.len() - 1]
                    } else {
                        s
                    }
                } else {
                    content_after_q
                }
            };
            (slice.to_string(), true)
        } else {
            let end = after_colon
                .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
                .unwrap_or(after_colon.len());
            (after_colon[..end].to_string(), false)
        };

        if is_quoted {
            raw_val = unescape_string(&raw_val);
        }
        return Some(raw_val);
    }
    None
}

/// Fallback extractor that recovers an action even when models output unescaped
/// quotes inside code or HTML strings (e.g. `<script src="main.js"></script>`).
pub(crate) fn heuristic_action(text: &str) -> Option<Action> {
    let tool = find_tool_name(text)?;
    let mut args = Map::new();

    match tool.as_str() {
        "edit_file" => {
            let path = extract_field_value(text, "path", &["old", "old_text", "search", "new", "new_text", "replace"])
                .or_else(|| extract_field_value(text, "file", &["old", "new"]));
            let old = extract_field_value(text, "old", &["new", "new_text", "replace"])
                .or_else(|| extract_field_value(text, "old_text", &["new"]))
                .or_else(|| extract_field_value(text, "search", &["replace", "new"]));
            let new = extract_field_value(text, "new", &[])
                .or_else(|| extract_field_value(text, "new_text", &[]))
                .or_else(|| extract_field_value(text, "replace", &[]));

            if let Some(p) = path { args.insert("path".into(), Value::String(p)); }
            if let Some(o) = old { args.insert("old".into(), Value::String(o)); }
            if let Some(n) = new { args.insert("new".into(), Value::String(n)); }
        }
        "write_file" => {
            let path = extract_field_value(text, "path", &["content", "text", "body"])
                .or_else(|| extract_field_value(text, "file", &["content"]));
            let content = extract_field_value(text, "content", &[])
                .or_else(|| extract_field_value(text, "text", &[]))
                .or_else(|| extract_field_value(text, "body", &[]));

            if let Some(p) = path { args.insert("path".into(), Value::String(p)); }
            if let Some(c) = content { args.insert("content".into(), Value::String(c)); }
        }
        "read_file" => {
            let path = extract_field_value(text, "path", &["start_line", "limit", "offset"])
                .or_else(|| extract_field_value(text, "file", &["start_line"]));
            let start = extract_field_value(text, "start_line", &["limit"]);
            let limit = extract_field_value(text, "limit", &[]);

            if let Some(p) = path { args.insert("path".into(), Value::String(p)); }
            if let Some(s) = start.and_then(|v| v.trim().parse::<i64>().ok()) {
                args.insert("start_line".into(), json!(s));
            }
            if let Some(l) = limit.and_then(|v| v.trim().parse::<i64>().ok()) {
                args.insert("limit".into(), json!(l));
            }
        }
        "list_files" => {
            let path = extract_field_value(text, "path", &[]).or_else(|| extract_field_value(text, "dir", &[]));
            if let Some(p) = path { args.insert("path".into(), Value::String(p)); }
        }
        "search" => {
            let pat = extract_field_value(text, "pattern", &["path", "dir"]).or_else(|| extract_field_value(text, "query", &["path"]));
            let path = extract_field_value(text, "path", &[]).or_else(|| extract_field_value(text, "dir", &[]));
            if let Some(p) = pat { args.insert("pattern".into(), Value::String(p)); }
            if let Some(d) = path { args.insert("path".into(), Value::String(d)); }
        }
        "run_command" | "command" => {
            let cmd = extract_field_value(text, "cmd", &[]).or_else(|| extract_field_value(text, "command", &[]));
            if let Some(c) = cmd { args.insert("cmd".into(), Value::String(c)); }
        }
        "finish" => {
            let summary = extract_field_value(text, "summary", &[]).or_else(|| extract_field_value(text, "report", &[]));
            if let Some(s) = summary { args.insert("summary".into(), Value::String(s)); }
        }
        "delegate" => {
            let agent = extract_field_value(text, "agent", &["instruction", "goal"]).or_else(|| extract_field_value(text, "to", &["instruction"]));
            let inst = extract_field_value(text, "instruction", &[]).or_else(|| extract_field_value(text, "goal", &[]));
            if let Some(a) = agent { args.insert("agent".into(), Value::String(a)); }
            if let Some(i) = inst { args.insert("instruction".into(), Value::String(i)); }
        }
        _ => {
            // Newer tools take short scalar arguments; pick up whichever appear.
            for key in ["path", "url", "query", "id", "from", "to", "command", "cwd", "screenshot"] {
                if let Some(v) = extract_field_value(text, key, &["path", "url", "query", "id", "from", "to", "command", "cwd", "wait_ms", "screenshot", "recursive"]) {
                    args.insert(key.into(), Value::String(v));
                }
            }
        }
    }

    Some(Action { tool, args })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaced_and_foreign_tool_names_are_canonicalized() {
        for (raw, want) in [
            ("repo_browser.write_file", "write_file"),
            ("functions.read_file", "read_file"),
            ("repo_browser.open_file", "read_file"),
            ("repo_browser.print_tree", "list_files"),
            ("container.exec", "run_command"),
            ("Write File", "write_file"),
            ("edit-file", "edit_file"),
            ("str_replace", "edit_file"),
            ("my_skill_tool", "my_skill_tool"),
            ("Deploy.Preview", "Preview"),
        ] {
            assert_eq!(canonical_tool(raw), want, "{raw}");
        }
        let a = parse_action(r#"{"tool": "repo_browser.write_file", "args": {"path": "a.js", "content": "x"}}"#).unwrap();
        assert_eq!((a.tool.as_str(), a.str_arg("path")), ("write_file", Some("a.js")));
    }

    #[test]
    fn plain_read_commands_are_understood() {
        for t in ["read_file water_shader.frag", "Let me look.\nread file water_shader.frag", "`read_file(path=\"water_shader.frag\")`"] {
            let a = parse_action(t).unwrap_or_else(|e| panic!("{t}: {e}"));
            assert_eq!((a.tool.as_str(), a.str_arg("path")), ("read_file", Some("water_shader.frag")), "{t}");
        }
        assert_eq!(parse_action("list_files").unwrap().tool, "list_files");
        // Prose that merely mentions reading is not a call.
        assert!(parse_action("I will read the shader next and then fix it").is_err());
        assert!(parse_action("readme.md looks fine").is_err());
    }

    #[test]
    fn new_tools_are_recovered_from_broken_json() {
        let a = parse_action(r#"{"tool": "web_fetch", "args": {"url": "https://example.com/a?b="c""}}"#).unwrap();
        assert_eq!(a.tool, "web_fetch");
        assert!(a.str_arg("url").unwrap().starts_with("https://example.com/a"));
    }

    #[test]
    fn fenced_action_after_reasoning() {
        let t = "I'll read it first {not json}.\n```json\n{\"tool\": \"read_file\", \"args\": {\"path\": \"src/main.rs\"}}\n```";
        let a = parse_action(t).unwrap();
        assert_eq!(a.tool, "read_file");
        assert_eq!(a.str_arg("path"), Some("src/main.rs"));
    }

    #[test]
    fn last_object_wins_and_braces_in_strings_are_fine() {
        let t = r#"Example: {"tool": "search", "args": {"pattern": "x"}}
Now: {"tool": "write_file", "args": {"path": "a.rs", "content": "fn main() { println!(\"}\"); }"}}"#;
        let a = parse_action(t).unwrap();
        assert_eq!(a.tool, "write_file");
        assert_eq!(a.str_arg("content"), Some("fn main() { println!(\"}\"); }"));
    }

    #[test]
    fn aliases_flat_args_and_raw_newlines_are_accepted() {
        let a = parse_action(r#"{"name": "read_file", "arguments": "{\"path\": \"a\"}"}"#).unwrap();
        assert_eq!((a.tool.as_str(), a.str_arg("path")), ("read_file", Some("a")));

        let a = parse_action(r#"{"tool": "read_file", "path": "b", "start_line": "3"}"#).unwrap();
        assert_eq!((a.str_arg("path"), a.int_arg("start_line")), (Some("b"), Some(3)));

        let a = parse_action("{\"tool\": \"write_file\", \"args\": {\"path\": \"x\", \"content\": \"line1\nline2\"}}")
            .unwrap();
        assert_eq!(a.str_arg("content"), Some("line1\nline2"));
    }

    #[test]
    fn call_quoted_by_an_ollama_parse_error_is_recovered() {
        // Args object closed early, tool name after it, JS single quotes.
        let t = r#"{"path":"src/main.js","content":"import * as THREE from 'three';\nconst s = `\n  uniform float time;\n`;\nanimate();\n"},"tool":"write_file"}"#;
        let a = parse_action(t).unwrap();
        assert_eq!((a.tool.as_str(), a.str_arg("path")), ("write_file", Some("src/main.js")));
        let content = a.str_arg("content").unwrap();
        assert!(content.starts_with("import * as THREE from 'three';\n"), "{content}");
        assert!(content.ends_with("animate();\n"), "{content}");
    }

    #[test]
    fn missing_or_toolless_json_is_an_error() {
        assert!(parse_action("just prose").is_err());
        assert!(parse_action(r#"{"path": "x"}"#).is_err());
    }

    #[test]
    fn verdicts_default_to_revise() {
        assert!(parse_verdict(r#"{"verdict": "approve", "feedback": ""}"#).0);
        let (ok, fb) = parse_verdict(r#"Issues. {"verdict": "revise", "feedback": "missing test"}"#);
        assert_eq!((ok, fb.as_str()), (false, "missing test"));
        assert!(!parse_verdict("looks fine I guess").0);
    }

    #[test]
    fn handles_trailing_commas_and_comments() {
        let t = r#"{
            // Let's edit the file
            "tool": "edit_file",
            "args": {
                "path": "index.html",
                "old": "hello",
                "new": "world",
            },
        }"#;
        let a = parse_action(t).unwrap();
        assert_eq!(a.tool, "edit_file");
        assert_eq!(a.str_arg("path"), Some("index.html"));
        assert_eq!(a.str_arg("old"), Some("hello"));
        assert_eq!(a.str_arg("new"), Some("world"));
    }

    #[test]
    fn handles_single_quotes() {
        let t = "{ 'tool': 'read_file', 'args': { 'path': 'index.html' } }";
        let a = parse_action(t).unwrap();
        assert_eq!(a.tool, "read_file");
        assert_eq!(a.str_arg("path"), Some("index.html"));
        let t = r#"{ 'tool': 'write_file', 'args': { 'path': 'a.html', 'content': '<p class="x">it\'s</p>' } }"#;
        let a = parse_action(t).unwrap();
        assert_eq!(a.str_arg("content"), Some(r#"<p class="x">it's</p>"#));
    }

    #[test]
    fn handles_unescaped_html_quotes_in_edit_file() {
        let t = r#"I will integrate water.js now:
```json
{
  "tool": "edit_file",
  "args": {
    "path": "index.html",
    "old": "<script src="main.js"></script>",
    "new": "<script src="water.js"></script>\n<script src="main.js"></script>"
  }
}
```"#;
        let a = parse_action(t).unwrap();
        assert_eq!(a.tool, "edit_file");
        assert_eq!(a.str_arg("path"), Some("index.html"));
        assert_eq!(a.str_arg("old"), Some("<script src=\"main.js\"></script>"));
        assert_eq!(a.str_arg("new"), Some("<script src=\"water.js\"></script>\n<script src=\"main.js\"></script>"));
    }
}
