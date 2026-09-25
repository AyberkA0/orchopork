//! The actor's action protocol: one JSON object per reply,
//! `{"tool": "<name>", "args": {...}}`.
//!
//! A text protocol instead of each provider's native tool-calling API:
//! it behaves the same on every backend (Ollama models without a tool
//! template, llama.cpp, Claude, OpenAI-compatible APIs), keeps the
//! transcript plain text, and is parsed leniently because small local
//! models get the details wrong (fences or not, `name`/`arguments` instead
//! of `tool`/`args`, raw newlines inside JSON strings).

use serde_json::{Map, Value};

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

/// Finds the *last* JSON object in `text` that names a tool. Reasoning
/// before it may contain other braces or example snippets.
pub fn parse_action(text: &str) -> Result<Action, String> {
    let objects = json_objects(text);
    if objects.is_empty() {
        return Err("no JSON object found in the reply".into());
    }
    for obj in objects.into_iter().rev() {
        if let Some(a) = to_action(obj) {
            return Ok(a);
        }
    }
    Err("found JSON, but no object with a \"tool\" field".into())
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
    balanced_spans(text)
        .into_iter()
        .filter_map(|span| {
            let parsed = serde_json::from_str::<Value>(span).or_else(|_| serde_json::from_str(&repair(span)));
            match parsed {
                Ok(Value::Object(m)) => Some(m),
                _ => None,
            }
        })
        .collect()
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
            // Unbalanced: retry from the next byte so a stray `{` in prose
            // does not hide a valid object after it.
            None => i = start + 1,
        }
    }
    spans
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
