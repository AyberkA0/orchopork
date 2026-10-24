//! The agent's browser: plain HTTP fetches, a web search, and a real
//! headless Chromium (Edge or Chrome, whichever is installed) for pages
//! that need JavaScript, console errors, or a screenshot.
//!
//! Files of the run's worktree are served from a throwaway loopback HTTP
//! server for the duration of one call, so `index.html` with ES modules
//! works the way it would on a dev server (file:// URLs block modules).
//!
//! The browser is driven over the DevTools protocol, so one page load
//! yields its console, uncaught exceptions, failed requests and every
//! screenshot of a sequence.

use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use reqwest::Url;
use serde_json::json;
use tokio::process::Command;

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36 orchopork";
const MAX_FETCH_BYTES: usize = 3_000_000;

fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(UA)
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client")
}

/// A URL the agent may open: http(s) only.
pub fn parse_web_url(raw: &str) -> Result<Url, String> {
    let raw = raw.trim();
    let with_scheme = if raw.contains("://") { raw.to_string() } else { format!("https://{raw}") };
    let url = Url::parse(&with_scheme).map_err(|e| format!("invalid URL {raw:?}: {e}"))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        s => Err(format!("only http(s) URLs can be opened, not {s}:")),
    }
}

/// GET a page and return it as readable text (HTML is flattened, links
/// kept as `text <url>`).
pub async fn fetch(url: &Url) -> Result<String, String> {
    let resp = http().get(url.clone()).send().await.map_err(|e| format!("cannot fetch {url}: {e}"))?;
    let status = resp.status();
    let ctype = resp.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
    let final_url = resp.url().clone();
    let bytes = resp.bytes().await.map_err(|e| format!("cannot read {url}: {e}"))?;
    let bytes = &bytes[..bytes.len().min(MAX_FETCH_BYTES)];
    let textual = ["text", "json", "xml", "javascript", "ecmascript", "css", "yaml", "toml", "x-sh", "graphql", "csv"];
    if !(ctype.is_empty() || textual.iter().any(|t| ctype.contains(t))) {
        return Err(format!("{final_url} returned {ctype} ({} bytes), not text", bytes.len()));
    }
    let body = String::from_utf8_lossy(bytes);
    let text = if ctype.contains("html") || body.trim_start().starts_with('<') { html_to_text(&body, &final_url) } else { body.into_owned() };
    Ok(format!("[{} {}] {final_url}\n\n{}", status.as_u16(), status.canonical_reason().unwrap_or(""), text.trim()))
}

/// Web search through DuckDuckGo's HTML endpoint (no API key needed).
pub async fn search(query: &str) -> Result<String, String> {
    let url = Url::parse_with_params("https://html.duckduckgo.com/html/", &[("q", query)]).map_err(|e| e.to_string())?;
    let html = http()
        .get(url)
        .send()
        .await
        .map_err(|e| format!("search failed: {e}"))?
        .text()
        .await
        .map_err(|e| format!("search failed: {e}"))?;
    let mut out = String::new();
    let mut n = 0;
    let mut rest = html.as_str();
    while let Some(i) = rest.find("class=\"result__a\"") {
        rest = &rest[i..];
        let href = attr(rest, "href").unwrap_or_default();
        let title = inner_text(rest).unwrap_or_default();
        let snippet = rest
            .find("class=\"result__snippet\"")
            .filter(|&j| rest[..j].matches("class=\"result__a\"").count() <= 1)
            .and_then(|j| inner_text(&rest[j..]))
            .unwrap_or_default();
        rest = &rest[1..];
        let link = ddg_target(&href);
        if link.is_empty() || link.contains("duckduckgo.com/y.js") {
            continue;
        }
        n += 1;
        out.push_str(&format!("{n}. {}\n   {link}\n", decode_entities(title.trim())));
        if !snippet.trim().is_empty() {
            out.push_str(&format!("   {}\n", decode_entities(snippet.trim())));
        }
        if n >= 10 {
            break;
        }
    }
    if n == 0 {
        return Ok(format!("no results for {query:?}"));
    }
    Ok(out)
}

fn ddg_target(href: &str) -> String {
    let href = decode_entities(href);
    let abs = if href.starts_with("//") { format!("https:{href}") } else { href.clone() };
    if let Ok(u) = Url::parse(&abs)
        && u.path().starts_with("/l/")
        && let Some((_, v)) = u.query_pairs().find(|(k, _)| k == "uddg")
    {
        return v.into_owned();
    }
    abs
}

fn attr(tag_start: &str, name: &str) -> Option<String> {
    let end = tag_start.find('>')?;
    let tag = &tag_start[..end];
    let key = format!("{name}=\"");
    let i = tag.find(&key)? + key.len();
    let j = tag[i..].find('"')?;
    Some(tag[i..i + j].to_string())
}

/// Text between the end of the tag at the start of `s` and its closing tag.
fn inner_text(s: &str) -> Option<String> {
    let start = s.find('>')? + 1;
    let end = s[start..].find("</")? + start;
    Some(strip_tags(&s[start..end]))
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let Some(end) = rest[..rest.len().min(12)].find(';') else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let ent = &rest[1..end];
        let rep = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            _ if ent.starts_with("#x") || ent.starts_with("#X") => {
                u32::from_str_radix(&ent[2..], 16).ok().and_then(char::from_u32)
            }
            _ if ent.starts_with('#') => ent[1..].parse::<u32>().ok().and_then(char::from_u32),
            _ => None,
        };
        match rep {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Flattens HTML to readable text: drops scripts/styles, turns block
/// elements into line breaks and keeps link targets.
pub fn html_to_text(html: &str, base: &Url) -> String {
    let lower = html.to_ascii_lowercase();
    let mut out = String::new();
    let mut title = String::new();
    let mut i = 0;
    let bytes = html.as_bytes();
    let mut pending_href: Option<String> = None;
    while i < html.len() {
        if bytes[i] != b'<' {
            let next = html[i..].find('<').map_or(html.len(), |n| i + n);
            out.push_str(&decode_entities(&html[i..next]));
            i = next;
            continue;
        }
        if html[i..].starts_with("<!--") {
            i = html[i..].find("-->").map_or(html.len(), |n| i + n + 3);
            continue;
        }
        let close = match html[i..].find('>') {
            Some(n) => i + n,
            None => break,
        };
        let tag_raw = &html[i + 1..close];
        let tag_lower = &lower[i + 1..close];
        let name: String =
            tag_lower.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '!').collect();
        let closing = tag_lower.starts_with('/');
        i = close + 1;
        match name.as_str() {
            "script" | "style" | "noscript" | "svg" | "template" | "head" if !closing => {
                let end_tag = format!("</{name}");
                let end = lower[i..].find(&end_tag).map_or(html.len(), |n| i + n);
                if name == "head" {
                    // Keep only the title from <head>.
                    if let Some(t) = lower[i..end].find("<title") {
                        let ts = i + t;
                        if let Some(te) = lower[ts..end].find("</title") {
                            // `<title ...>text` -> text
                            let inner = html[ts..ts + te].split_once('>').map_or("", |(_, t)| t);
                            title = decode_entities(inner).trim().to_string();
                        }
                    }
                }
                i = lower[end..].find('>').map_or(html.len(), |n| end + n + 1);
            }
            "a" if !closing => {
                pending_href = attr_ci(tag_raw, "href").and_then(|h| {
                    let h = decode_entities(&h);
                    (!h.starts_with('#') && !h.starts_with("javascript:"))
                        .then(|| base.join(&h).map(|u| u.to_string()).unwrap_or(h))
                });
            }
            "a" => {
                if let Some(h) = pending_href.take() {
                    out.push_str(&format!(" <{h}>"));
                }
            }
            "br" | "p" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "section" | "article"
            | "header" | "footer" | "nav" | "ul" | "ol" | "table" | "pre" | "blockquote" | "main" | "form" => {
                out.push('\n');
                if !closing && name == "li" {
                    out.push_str("- ");
                }
                if !closing && name.len() == 2 && name.starts_with('h') {
                    out.push_str(&"#".repeat(name[1..].parse().unwrap_or(1)));
                    out.push(' ');
                }
            }
            "td" | "th" => out.push_str(" | "),
            "img" => {
                if let Some(alt) = attr_ci(tag_raw, "alt").filter(|a| !a.trim().is_empty()) {
                    out.push_str(&format!("[image: {}]", decode_entities(&alt)));
                }
            }
            "input" | "button" | "textarea" | "select" if !closing => {
                let label = attr_ci(tag_raw, "value")
                    .or_else(|| attr_ci(tag_raw, "placeholder"))
                    .or_else(|| attr_ci(tag_raw, "name"))
                    .unwrap_or_default();
                if name != "button" {
                    out.push_str(&format!("[{name} {}]", decode_entities(&label)));
                }
            }
            _ => {}
        }
    }
    // Collapse whitespace: trim lines, at most one blank line in a row.
    let mut clean = String::new();
    if !title.is_empty() {
        clean.push_str(&format!("Title: {title}\n\n"));
    }
    let mut blank = 0;
    for line in out.lines() {
        let l = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if l.is_empty() {
            blank += 1;
            if blank == 1 && !clean.is_empty() {
                clean.push('\n');
            }
            continue;
        }
        blank = 0;
        clean.push_str(&l);
        clean.push('\n');
    }
    clean
}

fn attr_ci(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let mut from = 0;
    while let Some(n) = lower[from..].find(name) {
        let at = from + n;
        from = at + name.len();
        let before_ok = at == 0 || lower.as_bytes()[at - 1].is_ascii_whitespace();
        let rest = tag[from..].trim_start();
        if !before_ok || !rest.starts_with('=') {
            continue;
        }
        let v = rest[1..].trim_start();
        return Some(match v.chars().next() {
            Some(q @ ('"' | '\'')) => v[1..].split(q).next().unwrap_or("").to_string(),
            _ => v.split(|c: char| c.is_whitespace() || c == '>').next().unwrap_or("").to_string(),
        });
    }
    None
}

// ---- headless Chromium -----------------------------------------------------------

/// Edge or Chrome, if one is installed (`ORCHOPORK_BROWSER` overrides).
pub fn find_browser() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ORCHOPORK_BROWSER")
        && !p.trim().is_empty()
    {
        return Some(PathBuf::from(p));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        for var in ["ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"] {
            if let Ok(base) = std::env::var(var) {
                let b = PathBuf::from(base);
                candidates.push(b.join(r"Microsoft\Edge\Application\msedge.exe"));
                candidates.push(b.join(r"Google\Chrome\Application\chrome.exe"));
                candidates.push(b.join(r"Chromium\Application\chrome.exe"));
                candidates.push(b.join(r"BraveSoftware\Brave-Browser\Application\brave.exe"));
            }
        }
    } else if cfg!(target_os = "macos") {
        for app in [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ] {
            candidates.push(PathBuf::from(app));
        }
    } else if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            for exe in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"] {
                candidates.push(dir.join(exe));
            }
        }
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// What to do with a loaded page.
pub struct RenderOpts {
    /// Milliseconds to let scripts run after the page loaded.
    pub wait_ms: u64,
    pub size: (u32, u32),
    /// Screenshots to save: the first after `wait_ms`, then one every
    /// `interval_ms`, all from the same page load.
    pub shots: Vec<PathBuf>,
    pub interval_ms: u64,
}

impl RenderOpts {
    pub fn new(wait_ms: u64, size: (u32, u32)) -> Self {
        Self { wait_ms, size, shots: Vec::new(), interval_ms: 1000 }
    }
}

#[derive(Debug, Default)]
pub struct Rendered {
    pub text: String,
    /// Console messages, uncaught exceptions and browser warnings, in order.
    pub console: Vec<String>,
    /// Requests that failed or got an HTTP error, as
    /// `Failed to load resource: 404 <url>`.
    pub failed: Vec<String>,
    /// Why the page itself did not load (`net::ERR_CONNECTION_REFUSED`,
    /// `HTTP 404`, ...).
    pub load_error: Option<String>,
    pub screenshots: Vec<PathBuf>,
    /// Facts that matter when a page shows little text: canvases, WebGL.
    pub notes: Vec<String>,
}

impl Rendered {
    /// Uncaught exceptions and failed loads: the page is broken.
    pub fn errors(&self) -> Vec<String> {
        self.console
            .iter()
            .filter(|l| l.contains("Uncaught") || l.contains("net::ERR_"))
            .chain(self.failed.iter())
            .cloned()
            .collect()
    }
}

/// Most console lines kept from one page.
const MAX_CONSOLE: usize = 60;

/// Loads `url` in headless Chromium, lets scripts run, and reports what a
/// developer would see in DevTools: console output, uncaught exceptions,
/// failed requests (404s included), the rendered text and screenshots.
///
/// Driven over the DevTools protocol so everything comes from one page
/// load; falls back to plain command-line flags (console only, one load
/// per screenshot) when DevTools cannot be reached.
pub async fn render(url: &Url, opts: &RenderOpts) -> Result<Rendered, String> {
    let exe = find_browser().ok_or(
        "no Chromium-based browser (Edge/Chrome) found for rendering; set ORCHOPORK_BROWSER to its path, or use web_fetch",
    )?;
    for p in &opts.shots {
        if let Some(parent) = p.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
    }
    let budget = Duration::from_millis(90_000 + opts.wait_ms + opts.interval_ms * opts.shots.len() as u64);
    match tokio::time::timeout(budget, devtools_render(&exe, url, opts)).await {
        Ok(Ok(r)) => Ok(r),
        Ok(Err(DevtoolsError::Unavailable(e))) => {
            tracing::warn!("DevTools unavailable ({e}); rendering with command-line flags");
            cli_render(&exe, url, opts).await
        }
        Ok(Err(DevtoolsError::Failed(e))) => Err(e),
        Err(_) => Err(format!("the browser did not finish within {}s", budget.as_secs())),
    }
}

/// Flags shared by both ways of driving the browser. WebGL uses the GPU
/// when there is one, as the user's browser would, and falls back to the
/// SwiftShader software renderer otherwise.
fn browser_flags(profile: &Path, size: (u32, u32)) -> Vec<String> {
    vec![
        "--headless=new".into(),
        "--enable-unsafe-swiftshader".into(),
        "--no-first-run".into(),
        "--no-default-browser-check".into(),
        "--disable-extensions".into(),
        "--disable-background-networking".into(),
        "--disable-sync".into(),
        "--mute-audio".into(),
        "--hide-scrollbars".into(),
        format!("--user-data-dir={}", profile.display()),
        format!("--window-size={},{}", size.0, size.1),
    ]
}

enum DevtoolsError {
    /// DevTools could not be reached at all: try the fallback.
    Unavailable(String),
    /// The page was opened but something went wrong: report it.
    Failed(String),
}

/// Removes the throwaway profile directory when dropped.
struct Profile(PathBuf);

impl Drop for Profile {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn devtools_render(exe: &Path, url: &Url, opts: &RenderOpts) -> Result<Rendered, DevtoolsError> {
    use DevtoolsError::{Failed, Unavailable};
    let profile = Profile(std::env::temp_dir().join(format!("orchopork-browser-{}", uuid::Uuid::new_v4().simple())));
    let mut args = browser_flags(&profile.0, opts.size);
    args.push("--remote-debugging-port=0".into());
    args.push("about:blank".into());
    let mut child = Command::new(exe)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Failed(format!("cannot start {}: {e}", exe.display())))?;
    let result = async {
        let port = devtools_port(&profile.0, &mut child).await.map_err(Unavailable)?;
        let ws_url = page_socket(port).await.map_err(Unavailable)?;
        let mut cdp = Cdp { ws: Ws::connect(&ws_url).await.map_err(Unavailable)?, next: 0, events: Vec::new() };
        drive(&mut cdp, url, opts).await.map_err(Failed)
    }
    .await;
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
    drop(profile);
    result
}

/// The port Chromium picked, from the `DevToolsActivePort` file it writes
/// into the profile.
async fn devtools_port(profile: &Path, child: &mut tokio::process::Child) -> Result<u16, String> {
    let file = profile.join("DevToolsActivePort");
    for _ in 0..400 {
        if let Ok(text) = tokio::fs::read_to_string(&file).await
            && let Some(port) = text.lines().next().and_then(|l| l.trim().parse::<u16>().ok())
        {
            return Ok(port);
        }
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("the browser exited at startup ({status})"));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Err("the browser did not open its DevTools port within 20s".into())
}

/// The WebSocket of the browser's first page.
async fn page_socket(port: u16) -> Result<String, String> {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build().map_err(|e| e.to_string())?;
    for _ in 0..50 {
        if let Ok(resp) = client.get(format!("http://127.0.0.1:{port}/json/list")).send().await
            && let Ok(list) = resp.json::<serde_json::Value>().await
            && let Some(ws) = list
                .as_array()
                .into_iter()
                .flatten()
                .find(|t| t["type"] == "page")
                .and_then(|t| t["webSocketDebuggerUrl"].as_str())
        {
            return Ok(ws.to_string());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("the browser has no page to drive".into())
}

/// Loads the page, waits, takes the screenshots and collects what happened.
async fn drive(cdp: &mut Cdp, url: &Url, opts: &RenderOpts) -> Result<Rendered, String> {
    for m in ["Page.enable", "Runtime.enable", "Log.enable", "Network.enable"] {
        cdp.call(m, json!({})).await?;
    }
    cdp.call(
        "Emulation.setDeviceMetricsOverride",
        json!({ "width": opts.size.0, "height": opts.size.1, "deviceScaleFactor": 1, "mobile": false }),
    )
    .await?;
    let nav = cdp.call("Page.navigate", json!({ "url": url.as_str() })).await?;
    let frame = nav["frameId"].as_str().unwrap_or("").to_string();
    let mut notes = Vec::new();
    if let Some(e) = nav["errorText"].as_str().filter(|e| !e.is_empty()) {
        let mut r = Events::new(&frame).absorb(std::mem::take(&mut cdp.events));
        r.load_error = Some(e.to_string());
        return Ok(r);
    }
    if !cdp.wait_for("Page.loadEventFired", Duration::from_secs(30)).await {
        notes.push("the page's load event did not fire within 30 s (a script or request may hang)".to_string());
    }
    cdp.pump(Duration::from_millis(opts.wait_ms)).await;
    let mut screenshots = Vec::new();
    for (i, path) in opts.shots.iter().enumerate() {
        if i > 0 {
            cdp.pump(Duration::from_millis(opts.interval_ms)).await;
        }
        let shot = cdp.call("Page.captureScreenshot", json!({ "format": "png" })).await?;
        let png = shot["data"].as_str().and_then(crate::fsutil::base64_decode).ok_or("the screenshot came back empty")?;
        tokio::fs::write(path, png).await.map_err(|e| format!("cannot save {}: {e}", path.display()))?;
        screenshots.push(path.clone());
    }
    let diag = cdp
        .call("Runtime.evaluate", json!({ "expression": PAGE_FACTS, "returnByValue": true }))
        .await
        .map(|v| v["result"]["value"].clone())
        .unwrap_or_default();
    let _ = cdp.ws.send(&json!({ "id": 0, "method": "Browser.close" }).to_string()).await;
    let mut r = Events::new(&frame).absorb(std::mem::take(&mut cdp.events));
    r.text = html_to_text(diag["html"].as_str().unwrap_or(""), url);
    r.screenshots = screenshots;
    notes.extend(page_notes(&diag));
    r.notes = notes;
    Ok(r)
}

/// The rendered DOM plus facts about canvases (a WebGL or 2D app has no
/// text, so "no text" says nothing about whether it works).
const PAGE_FACTS: &str = r#"(() => {
  const canvases = [...document.querySelectorAll('canvas')].map(c => {
    const r = c.getBoundingClientRect();
    return [c.width, c.height, Math.round(r.width), Math.round(r.height)];
  });
  let webgl = null, renderer = '';
  if (canvases.length) {
    try {
      const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl');
      webgl = !!gl;
      const info = gl && gl.getExtension('WEBGL_debug_renderer_info');
      renderer = gl ? String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER)) : '';
    } catch (e) { webgl = false; }
  }
  return { html: document.documentElement ? document.documentElement.outerHTML : '', canvases, webgl, renderer };
})()"#;

fn page_notes(diag: &serde_json::Value) -> Vec<String> {
    let mut notes = Vec::new();
    let canvases: Vec<Vec<i64>> = diag["canvases"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| c.as_array().into_iter().flatten().filter_map(|n| n.as_i64()).collect())
        .collect();
    if !canvases.is_empty() {
        let list: Vec<String> = canvases
            .iter()
            .filter(|c| c.len() == 4)
            .map(|c| format!("{}x{} px shown at {}x{}", c[0], c[1], c[2], c[3]))
            .collect();
        notes.push(format!("{} canvas element(s): {}", canvases.len(), list.join("; ")));
        if canvases.iter().all(|c| c.len() == 4 && (c[2] == 0 || c[3] == 0)) {
            notes.push("every canvas is shown at zero size, so nothing drawn on it is visible".into());
        }
    }
    if diag["webgl"] == false {
        notes.push("WebGL is not available in this headless browser: WebGL content cannot render here".into());
    }
    let renderer = diag["renderer"].as_str().unwrap_or("");
    if ["swiftshader", "llvmpipe", "basic render", "software"].iter().any(|s| renderer.to_lowercase().contains(s)) {
        notes.push(format!(
            "WebGL runs on a software renderer here ({renderer}), not a GPU: shaders that rely on undefined behaviour (normalizing a zero vector, dividing by zero, uninitialized values) can come out black here while looking fine on a real GPU. Treat black areas as a possible shader bug, not proof that nothing is drawn."
        ));
    }
    notes
}

/// Turns DevTools events into console lines and failed requests.
struct Events {
    frame: String,
    urls: std::collections::HashMap<String, (String, String)>,
    out: Rendered,
}

impl Events {
    fn new(frame: &str) -> Self {
        Self { frame: frame.to_string(), urls: Default::default(), out: Rendered::default() }
    }

    fn absorb(mut self, events: Vec<serde_json::Value>) -> Rendered {
        for e in &events {
            self.event(e);
        }
        self.out.console.truncate(MAX_CONSOLE);
        self.out.failed.dedup();
        self.out.failed.truncate(MAX_CONSOLE);
        self.out
    }

    fn console(&mut self, line: String) {
        if self.out.console.len() < MAX_CONSOLE * 2 && !self.out.console.contains(&line) {
            self.out.console.push(line);
        }
    }

    fn event(&mut self, e: &serde_json::Value) {
        let p = &e["params"];
        match e["method"].as_str().unwrap_or("") {
            "Network.requestWillBeSent" => {
                let id = p["requestId"].as_str().unwrap_or("").to_string();
                let url = p["request"]["url"].as_str().unwrap_or("").to_string();
                self.urls.insert(id, (url, p["type"].as_str().unwrap_or("").to_string()));
            }
            "Network.responseReceived" => {
                let status = p["response"]["status"].as_i64().unwrap_or(0);
                let url = p["response"]["url"].as_str().unwrap_or("");
                if status >= 400 && !ignored_request(url) {
                    self.out.failed.push(format!("Failed to load resource: {status} {url}"));
                    if p["type"] == "Document" && p["frameId"] == self.frame.as_str() && self.out.load_error.is_none() {
                        self.out.load_error = Some(format!("HTTP {status}"));
                    }
                }
            }
            "Network.loadingFailed" => {
                if p["canceled"] == true {
                    return;
                }
                let id = p["requestId"].as_str().unwrap_or("");
                let (url, kind) = self.urls.get(id).cloned().unwrap_or_default();
                if ignored_request(&url) {
                    return;
                }
                let why = p["blockedReason"].as_str().map_or_else(
                    || p["errorText"].as_str().unwrap_or("failed").to_string(),
                    |b| format!("blocked ({b})"),
                );
                self.out.failed.push(format!("Failed to load resource: {why} {url}"));
                if kind == "Document" && self.out.load_error.is_none() && url.starts_with("http") {
                    self.out.load_error = Some(why);
                }
            }
            "Runtime.consoleAPICalled" => {
                let kind = p["type"].as_str().unwrap_or("log");
                let text: Vec<String> = p["args"].as_array().into_iter().flatten().map(remote_value).collect();
                let at = p["stackTrace"]["callFrames"][0].clone();
                let line = format!("console.{kind}: {}{}", text.join(" "), source_of(&at["url"], &at["lineNumber"]));
                self.console(line);
            }
            "Runtime.exceptionThrown" => {
                let d = &p["exceptionDetails"];
                let desc = d["exception"]["description"].as_str().and_then(|s| s.lines().next()).unwrap_or("");
                let text = d["text"].as_str().unwrap_or("Uncaught");
                let msg = if desc.is_empty() || text.contains(desc) {
                    text.to_string()
                } else {
                    format!("{} {desc}", text.trim_end_matches(':'))
                };
                let msg = if msg.starts_with("Uncaught") { msg } else { format!("Uncaught {msg}") };
                self.console(format!("{msg}{}", source_of(&d["url"], &d["lineNumber"])));
            }
            "Log.entryAdded" => {
                let entry = &p["entry"];
                // Failed requests are reported from the network events.
                if entry["source"] == "network" || !matches!(entry["level"].as_str(), Some("error" | "warning")) {
                    return;
                }
                let text = entry["text"].as_str().unwrap_or("");
                self.console(format!("{}: {text}{}", entry["level"].as_str().unwrap_or(""), source_of(&entry["url"], &entry["lineNumber"])));
            }
            _ => {}
        }
    }
}

/// Requests the browser makes on its own.
fn ignored_request(url: &str) -> bool {
    url.ends_with("/favicon.ico") || url.starts_with("data:") || url.starts_with("chrome-extension:")
}

fn remote_value(v: &serde_json::Value) -> String {
    match &v["value"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => {
            v["description"].as_str().or_else(|| v["unserializableValue"].as_str()).unwrap_or(v["type"].as_str().unwrap_or("")).to_string()
        }
        other => other.to_string(),
    }
}

fn source_of(url: &serde_json::Value, line: &serde_json::Value) -> String {
    match url.as_str().filter(|u| !u.is_empty()) {
        Some(u) => format!(" (source: {u}:{})", line.as_i64().map_or(0, |l| l + 1)),
        None => String::new(),
    }
}

/// A DevTools connection to one page.
struct Cdp {
    ws: Ws,
    next: u64,
    /// Events received so far, in order.
    events: Vec<serde_json::Value>,
}

impl Cdp {
    async fn call(&mut self, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
        self.next += 1;
        let id = self.next;
        self.ws.send(&json!({ "id": id, "method": method, "params": params }).to_string()).await.map_err(|e| format!("{method}: {e}"))?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            let msg = tokio::time::timeout_at(deadline, self.ws.recv())
                .await
                .map_err(|_| format!("{method}: no answer from the browser within 30s"))??;
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg) else { continue };
            if v["id"].as_u64() == Some(id) {
                if let Some(e) = v.get("error") {
                    return Err(format!("{method}: {}", e["message"].as_str().unwrap_or("failed")));
                }
                return Ok(v["result"].clone());
            }
            if v.get("method").is_some() {
                self.events.push(v);
            }
        }
    }

    /// Collects events for `d`.
    async fn pump(&mut self, d: Duration) {
        let deadline = tokio::time::Instant::now() + d;
        while let Ok(Ok(msg)) = tokio::time::timeout_at(deadline, self.ws.recv()).await {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg)
                && v.get("method").is_some()
            {
                self.events.push(v);
            }
        }
    }

    /// Collects events until `method` arrives (true) or `d` passes.
    async fn wait_for(&mut self, method: &str, d: Duration) -> bool {
        if self.events.iter().any(|e| e["method"] == method) {
            return true;
        }
        let deadline = tokio::time::Instant::now() + d;
        while let Ok(Ok(msg)) = tokio::time::timeout_at(deadline, self.ws.recv()).await {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg)
                && v.get("method").is_some()
            {
                let hit = v["method"] == method;
                self.events.push(v);
                if hit {
                    return true;
                }
            }
        }
        false
    }
}

/// A minimal WebSocket client: enough for DevTools on loopback (text
/// messages, fragmentation, ping/pong, close).
struct Ws {
    stream: tokio::net::TcpStream,
    buf: Vec<u8>,
}

impl Ws {
    async fn connect(ws_url: &str) -> Result<Self, String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let u = Url::parse(ws_url).map_err(|e| format!("bad DevTools URL {ws_url}: {e}"))?;
        let host = u.host_str().unwrap_or("127.0.0.1").to_string();
        let port = u.port().unwrap_or(80);
        let mut stream = tokio::net::TcpStream::connect((host.as_str(), port)).await.map_err(|e| format!("cannot reach DevTools: {e}"))?;
        let key = crate::fsutil::base64(uuid::Uuid::new_v4().as_bytes());
        let path = match u.query() {
            Some(q) => format!("{}?{q}", u.path()),
            None => u.path().to_string(),
        };
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        );
        stream.write_all(req.as_bytes()).await.map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        let end = loop {
            let mut tmp = [0u8; 4096];
            let n = tokio::time::timeout(Duration::from_secs(10), stream.read(&mut tmp))
                .await
                .map_err(|_| "DevTools handshake timed out".to_string())?
                .map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("DevTools closed the connection during the handshake".into());
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break i + 4;
            }
            if buf.len() > 64 * 1024 {
                return Err("DevTools sent an oversized handshake".into());
            }
        };
        let head = String::from_utf8_lossy(&buf[..end]).to_string();
        if !head.starts_with("HTTP/1.1 101") {
            return Err(format!("DevTools refused the WebSocket: {}", head.lines().next().unwrap_or("")));
        }
        buf.drain(..end);
        Ok(Self { stream, buf })
    }

    async fn write_frame(&mut self, opcode: u8, payload: &[u8]) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        let mask: [u8; 4] = uuid::Uuid::new_v4().as_bytes()[..4].try_into().unwrap_or([7, 3, 5, 1]);
        let mut f = Vec::with_capacity(payload.len() + 14);
        f.push(0x80 | opcode);
        match payload.len() {
            n if n < 126 => f.push(0x80 | n as u8),
            n if n <= 0xFFFF => {
                f.push(0x80 | 126);
                f.extend_from_slice(&(n as u16).to_be_bytes());
            }
            n => {
                f.push(0x80 | 127);
                f.extend_from_slice(&(n as u64).to_be_bytes());
            }
        }
        f.extend_from_slice(&mask);
        f.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        self.stream.write_all(&f).await
    }

    async fn send(&mut self, text: &str) -> std::io::Result<()> {
        self.write_frame(0x1, text.as_bytes()).await
    }

    async fn fill(&mut self, need: usize) -> Result<(), String> {
        use tokio::io::AsyncReadExt;
        while self.buf.len() < need {
            let mut tmp = vec![0u8; 256 * 1024];
            let n = self.stream.read(&mut tmp).await.map_err(|e| format!("DevTools connection broke: {e}"))?;
            if n == 0 {
                return Err("the browser closed the DevTools connection".into());
            }
            self.buf.extend_from_slice(&tmp[..n]);
        }
        Ok(())
    }

    /// The next complete text message.
    async fn recv(&mut self) -> Result<String, String> {
        let mut msg = Vec::new();
        loop {
            self.fill(2).await?;
            let fin = self.buf[0] & 0x80 != 0;
            let opcode = self.buf[0] & 0x0F;
            let masked = self.buf[1] & 0x80 != 0;
            let (len, head) = match self.buf[1] & 0x7F {
                126 => {
                    self.fill(4).await?;
                    (u16::from_be_bytes([self.buf[2], self.buf[3]]) as usize, 4)
                }
                127 => {
                    self.fill(10).await?;
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&self.buf[2..10]);
                    (u64::from_be_bytes(b) as usize, 10)
                }
                n => (n as usize, 2),
            };
            let start = head + if masked { 4 } else { 0 };
            self.fill(start + len).await?;
            let mut payload = self.buf[start..start + len].to_vec();
            if masked {
                let key = [self.buf[head], self.buf[head + 1], self.buf[head + 2], self.buf[head + 3]];
                for (i, b) in payload.iter_mut().enumerate() {
                    *b ^= key[i % 4];
                }
            }
            self.buf.drain(..start + len);
            match opcode {
                0x8 => return Err("the browser closed the DevTools connection".into()),
                0x9 => self.write_frame(0xA, &payload).await.map_err(|e| e.to_string())?,
                0x0..=0x2 => {
                    msg.extend_from_slice(&payload);
                    if fin {
                        return Ok(String::from_utf8_lossy(&msg).into_owned());
                    }
                }
                _ => {}
            }
        }
    }
}

/// The fallback: one browser launch for the DOM and console, one per
/// screenshot. It cannot see failed requests or exceptions that are not
/// logged to the console.
async fn cli_render(exe: &Path, url: &Url, opts: &RenderOpts) -> Result<Rendered, String> {
    let run = |extra: Vec<String>, wait: u64| {
        let profile = Profile(std::env::temp_dir().join(format!("orchopork-browser-{}", uuid::Uuid::new_v4().simple())));
        let mut a = browser_flags(&profile.0, opts.size);
        a.extend([
            "--enable-logging=stderr".into(),
            "--v=0".into(),
            format!("--virtual-time-budget={}", wait.clamp(0, 30_000)),
            "--run-all-compositor-stages-before-draw".into(),
        ]);
        a.extend(extra);
        a.push(url.to_string());
        async move {
            let r = run_browser(exe, a).await;
            drop(profile);
            r
        }
    };
    let (stdout, stderr) = run(vec!["--dump-dom".into()], opts.wait_ms).await?;
    let mut screenshots = Vec::new();
    for (i, p) in opts.shots.iter().enumerate() {
        let r = run(vec![format!("--screenshot={}", p.display())], opts.wait_ms + i as u64 * opts.interval_ms).await;
        if p.is_file() {
            screenshots.push(p.clone());
        } else if let Err(e) = r {
            return Err(format!("screenshot failed: {e}"));
        }
    }
    let console: Vec<String> = stderr
        .lines()
        .filter(|l| l.contains(":CONSOLE") || l.contains("Uncaught") || l.contains("ERR_"))
        // The browser's own built-in extensions log too; only the page matters.
        .filter(|l| !l.contains("chrome-extension://") && !l.contains("edge://"))
        .map(|l| {
            // "[pid:tid:date:INFO:CONSOLE(12)] "msg", source: url (12)" -> "CONSOLE(12) "msg" ..."
            l.split_once("] ").map_or(l, |(_, m)| m).trim().to_string()
        })
        .take(MAX_CONSOLE)
        .collect();
    let load_error = console
        .iter()
        .find(|l| l.contains("Page load failed") || l.contains("ERR_CONNECTION_REFUSED") || l.contains("ERR_NAME_NOT_RESOLVED"))
        .cloned();
    Ok(Rendered {
        text: html_to_text(&stdout, url),
        console,
        load_error,
        screenshots,
        notes: vec!["DevTools was unavailable: failed requests (404s) are not reported".into()],
        ..Default::default()
    })
}

async fn run_browser(exe: &Path, args: Vec<String>) -> Result<(String, String), String> {
    let child = Command::new(exe)
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
    match tokio::time::timeout(Duration::from_secs(60), child.wait_with_output()).await {
        Ok(Ok(out)) => Ok((String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("the browser did not finish within 60s".into()),
    }
}

// ---- serving worktree files ------------------------------------------------------

/// A loopback HTTP server over `root` that lives as long as the value.
pub struct LocalServer {
    pub port: u16,
    handle: tokio::task::JoinHandle<()>,
}

impl Drop for LocalServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl LocalServer {
    pub async fn start(root: PathBuf) -> Result<Self, String> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.map_err(|e| e.to_string())?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let app = axum::Router::new().fallback(move |uri: axum::http::Uri| {
            let root = root.clone();
            async move { serve_file(&root, uri.path(), false).await }
        });
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self { port, handle })
    }
}

/// Serves `rel` (a URL path) from `root`: directories fall back to their
/// `index.html`; nothing outside `root` or inside `.git` is reachable.
///
/// With `sandbox`, the page runs in an opaque origin (CSP `sandbox`), so
/// agent-written JavaScript shown on the dashboard's own origin cannot call
/// the dashboard API; CORS is opened so its module scripts still load.
pub async fn serve_file(root: &Path, rel: &str, sandbox: bool) -> Response {
    let decoded = percent_decode(rel.split('?').next().unwrap_or(""));
    let mut path = root.to_path_buf();
    for c in Path::new(decoded.trim_start_matches('/')).components() {
        match c {
            Component::Normal(p) if p != ".git" => path.push(p),
            Component::CurDir => {}
            _ => return (StatusCode::NOT_FOUND, "not found").into_response(),
        }
    }
    if path.is_dir() {
        path.push("index.html");
    }
    match crate::fsutil::canonical(&path) {
        Ok(real) if real.starts_with(crate::fsutil::canonical(root).unwrap_or_else(|_| root.to_path_buf())) => {}
        _ => return (StatusCode::NOT_FOUND, "not found").into_response(),
    }
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let mut resp =
                ([(header::CONTENT_TYPE, mime(&path)), (header::CACHE_CONTROL, "no-store")], Body::from(bytes))
                    .into_response();
            if sandbox {
                let h = resp.headers_mut();
                h.insert(
                    header::CONTENT_SECURITY_POLICY,
                    header::HeaderValue::from_static(
                        "sandbox allow-scripts allow-forms allow-popups allow-modals allow-downloads allow-pointer-lock",
                    ),
                );
                h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, header::HeaderValue::from_static("*"));
                h.insert(header::REFERRER_POLICY, header::HeaderValue::from_static("no-referrer"));
            }
            resp
        }
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

fn percent_decode(s: &str) -> String {
    // Byte-wise: slicing the `str` could split a multi-byte character, and a
    // panic here would take the whole server down (release builds abort).
    let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(hi), Some(lo)) = (hex(b[i + 1]), hex(b[i + 2]))
        {
            out.push(hi * 16 + lo);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub fn mime(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "wasm" => "application/wasm",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "glsl" | "vert" | "frag" | "txt" | "md" | "csv" => "text/plain; charset=utf-8",
        "xml" => "application/xml",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_becomes_readable_text() {
        let base = Url::parse("https://example.com/docs/").unwrap();
        let html = r#"<html><head><title>T &amp; Q</title><style>.x{}</style></head>
            <body><h1>Hello</h1><script>var a = "<p>";</script><p>One <a href="page">link</a></p>
            <ul><li>a</li><li>b</li></ul><!-- hidden --></body></html>"#;
        let t = html_to_text(html, &base);
        assert!(t.starts_with("Title: T & Q"), "{t}");
        assert!(t.contains("# Hello"), "{t}");
        assert!(t.contains("link <https://example.com/docs/page>"), "{t}");
        assert!(t.contains("- a") && t.contains("- b"), "{t}");
        assert!(!t.contains("var a") && !t.contains("hidden") && !t.contains(".x{}"), "{t}");
    }

    #[test]
    fn entities_and_percent_decoding() {
        assert_eq!(decode_entities("a &lt;b&gt; &#65;&#x42; &unknown; &"), "a <b> AB &unknown; &");
        assert_eq!(percent_decode("/a%20b/c.js"), "/a b/c.js");
        assert_eq!(percent_decode("/100%"), "/100%");
        assert_eq!(percent_decode("/%é%zz"), "/%é%zz", "never panics on odd input");
    }

    #[test]
    fn only_web_urls_are_accepted() {
        assert!(parse_web_url("example.com").is_ok());
        assert!(parse_web_url("file:///etc/passwd").is_err());
    }

    #[tokio::test]
    async fn local_server_serves_worktree_files_only() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("index.html"), "<p>hi</p>").unwrap();
        std::fs::create_dir(d.path().join(".git")).unwrap();
        std::fs::write(d.path().join(".git").join("config"), "secret").unwrap();
        let r = serve_file(d.path(), "/", true).await;
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(r.headers()[header::CONTENT_TYPE], "text/html; charset=utf-8");
        assert!(r.headers()[header::CONTENT_SECURITY_POLICY].to_str().unwrap().starts_with("sandbox"));
        assert_eq!(serve_file(d.path(), "/.git/config", false).await.status(), StatusCode::NOT_FOUND);
        assert_eq!(serve_file(d.path(), "/../x", false).await.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn pages_report_console_exceptions_failed_requests_and_frames() {
        if find_browser().is_none() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        std::fs::write(
            d.path().join("index.html"),
            r#"<html><head><title>T</title></head><body><h1>Hi</h1><img src="missing.png"><canvas width="300" height="150"></canvas>
            <script>console.error("boom"); setTimeout(() => { throw new Error("late"); }, 10); fetch("nope.json");</script></body></html>"#,
        )
        .unwrap();
        let srv = LocalServer::start(d.path().to_path_buf()).await.unwrap();
        let url = Url::parse(&format!("http://127.0.0.1:{}/index.html", srv.port)).unwrap();
        let mut o = RenderOpts::new(500, (800, 600));
        o.shots = vec![d.path().join("a.png"), d.path().join("b.png")];
        o.interval_ms = 200;
        let r = render(&url, &o).await.unwrap();
        assert!(r.load_error.is_none(), "{r:?}");
        assert!(r.console.iter().any(|l| l.contains("console.error: boom")), "{r:?}");
        assert!(r.console.iter().any(|l| l.contains("Uncaught") && l.contains("late")), "{r:?}");
        assert!(r.failed.iter().any(|l| l.contains("404") && l.contains("missing.png")), "{r:?}");
        assert!(r.failed.iter().any(|l| l.contains("nope.json")), "{r:?}");
        assert!(!r.failed.iter().any(|l| l.contains("favicon")), "{r:?}");
        assert!(r.text.contains("Hi"), "{r:?}");
        assert!(r.notes.iter().any(|n| n.contains("canvas")), "{r:?}");
        assert_eq!(r.screenshots.len(), 2);
        assert!(o.shots.iter().all(|p| p.is_file()));
        assert!(r.errors().iter().any(|e| e.contains("missing.png")));

        let ghost = Url::parse(&format!("http://127.0.0.1:{}/ghost.html", srv.port)).unwrap();
        let r = render(&ghost, &RenderOpts::new(0, (800, 600))).await.unwrap();
        assert_eq!(r.load_error.as_deref(), Some("HTTP 404"), "{r:?}");

        let closed = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let dead = Url::parse(&format!("http://127.0.0.1:{}/", closed.local_addr().unwrap().port())).unwrap();
        drop(closed);
        let r = render(&dead, &RenderOpts::new(0, (800, 600))).await.unwrap();
        assert!(r.load_error.as_deref().is_some_and(|e| e.contains("REFUSED")), "{r:?}");
    }

    #[test]
    fn software_webgl_is_flagged() {
        let d = json!({ "canvases": [[800, 600, 800, 600]], "webgl": true, "renderer": "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)))" });
        let notes = page_notes(&d);
        assert!(notes.iter().any(|n| n.contains("software renderer")), "{notes:?}");
        let gpu = json!({ "canvases": [[800, 600, 800, 600]], "webgl": true, "renderer": "ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Ti Direct3D11)" });
        assert!(!page_notes(&gpu).iter().any(|n| n.contains("software")));
    }

    /// Hits the real internet: `cargo test web_tools -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn web_tools_reach_the_internet() {
        let page = fetch(&parse_web_url("https://example.com").unwrap()).await.unwrap();
        assert!(page.contains("Example Domain"), "{page}");
        let hits = search("rust programming language").await.unwrap();
        assert!(hits.contains("1. ") && hits.contains("http"), "{hits}");
    }
}
