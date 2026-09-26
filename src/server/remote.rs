//! Optional access from other devices (Models & settings → Access from
//! other devices). Off by default: the server then binds loopback only.
//!
//! When it is on, the server binds every interface and each request from a
//! non-loopback peer must carry the access token: the `orchopork_token`
//! cookie (set after the login page or a `?token=` link) or an
//! `Authorization: Bearer` header. Requests from this machine keep the
//! loopback `Host`/`Origin` rules in `local_only`.
//!
//! The setting is machine-wide, not per workspace, because the listener is:
//! `$XDG_CONFIG_HOME/orchopork/server.yaml` (`~/.config/...`; `%APPDATA%` on
//! Windows; `$ORCHOPORK_CONFIG_DIR` overrides), created `0600`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::Duration;

use axum::Json;
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{Html, IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::error::Result;

const COOKIE: &str = "orchopork_token";
/// 30 days; the token itself never expires until it is regenerated.
const COOKIE_MAX_AGE: u32 = 30 * 24 * 3600;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    enabled: bool,
    token: String,
}

pub struct Remote {
    /// `None` when no config directory could be found: the setting then
    /// lasts until the server stops.
    path: Option<PathBuf>,
    settings: RwLock<Settings>,
    /// Carries `enabled`; the serve loop rebinds when it changes.
    tx: watch::Sender<bool>,
    /// What the listener is bound to, and why remote binding failed if it did.
    listening: RwLock<(Option<SocketAddr>, Option<String>)>,
}

pub fn default_path() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("ORCHOPORK_CONFIG_DIR") {
        return Some(PathBuf::from(d).join("server.yaml"));
    }
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")));
    base.map(|b| b.join("orchopork").join("server.yaml"))
}

/// 244 random bits from two v4 UUIDs (the OS RNG), as 64 hex chars.
fn new_token() -> String {
    format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())
}

impl Remote {
    /// An unreadable file means "off": failing closed is the safe default.
    pub fn open(path: Option<PathBuf>) -> Self {
        let settings: Settings = path
            .as_ref()
            .and_then(|p| match std::fs::read_to_string(p) {
                Ok(text) => {
                    serde_yaml::from_str(&text).map_err(|e| tracing::warn!("ignoring {}: {e}", p.display())).ok()
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => {
                    tracing::warn!("ignoring {}: {e}", p.display());
                    None
                }
            })
            .unwrap_or_default();
        let enabled = settings.enabled && !settings.token.is_empty();
        let (tx, _) = watch::channel(enabled);
        Self { path, settings: RwLock::new(Settings { enabled, ..settings }), tx, listening: RwLock::new((None, None)) }
    }

    pub fn enabled(&self) -> bool {
        self.settings.read().unwrap().enabled
    }

    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.tx.subscribe()
    }

    pub fn set_listening(&self, addr: SocketAddr, error: Option<String>) {
        *self.listening.write().unwrap() = (Some(addr), error);
    }

    fn save(&self, s: &Settings) -> Result<()> {
        if let Some(p) = &self.path {
            crate::fsutil::write_atomic(p, serde_yaml::to_string(s)?.as_bytes(), true)?;
        }
        Ok(())
    }

    /// Turning it on creates a token if there is none yet.
    pub fn set_enabled(&self, on: bool) -> Result<()> {
        let mut next = self.settings.read().unwrap().clone();
        next.enabled = on;
        if on && next.token.is_empty() {
            next.token = new_token();
        }
        self.save(&next)?;
        *self.settings.write().unwrap() = next;
        self.tx.send_replace(on);
        Ok(())
    }

    /// Replaces the token: every device has to log in again.
    pub fn regenerate(&self) -> Result<String> {
        let mut next = self.settings.read().unwrap().clone();
        next.token = new_token();
        self.save(&next)?;
        *self.settings.write().unwrap() = next.clone();
        Ok(next.token)
    }

    fn check(&self, presented: &str) -> bool {
        let s = self.settings.read().unwrap();
        s.enabled && !s.token.is_empty() && constant_time_eq(presented.trim().as_bytes(), s.token.as_bytes())
    }

    pub fn status(&self) -> Value {
        let s = self.settings.read().unwrap();
        let (addr, error) = self.listening.read().unwrap().clone();
        json!({
            "enabled": s.enabled,
            "token": s.token,
            "port": addr.map(|a| a.port()),
            "all_interfaces": addr.is_some_and(|a| a.ip().is_unspecified()),
            "error": error,
            "lan_ip": lan_ip(),
            "persisted": self.path.is_some(),
        })
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// This machine's address on its local network: the source address the OS
/// would pick to reach the internet. A UDP "connect" sends no packet.
fn lan_ip() -> Option<String> {
    let sock = std::net::UdpSocket::bind(("0.0.0.0", 0)).ok()?;
    sock.connect(("192.0.2.1", 80)).ok()?;
    let ip = sock.local_addr().ok()?.ip();
    (!ip.is_unspecified() && !ip.is_loopback()).then(|| ip.to_string())
}

fn cookie_token(headers: &HeaderMap) -> Option<String> {
    headers.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).flat_map(|v| v.split(';')).find_map(|kv| {
        let (k, v) = kv.trim().split_once('=')?;
        (k == COOKIE).then(|| v.to_string())
    })
}

fn bearer_token(headers: &HeaderMap) -> Option<String> {
    let v = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, tok) = v.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| tok.trim().to_string())
}

fn query_token(req: &Request) -> Option<String> {
    req.uri().query()?.split('&').find_map(|kv| kv.strip_prefix("token=").map(str::to_string))
}

fn set_cookie(token: &str) -> HeaderValue {
    HeaderValue::from_str(&format!("{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={COOKIE_MAX_AGE}"))
        .expect("token is hex")
}

fn deny(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({ "error": msg }))).into_response()
}

/// Gate for requests from another machine (see the module docs).
pub async fn guard(remote: &Remote, req: Request, next: Next) -> Response {
    if !remote.enabled() {
        return deny(StatusCode::FORBIDDEN, "access from other devices is turned off in orchopork's settings");
    }
    let headers = req.headers();
    let from_cookie = cookie_token(headers).filter(|t| remote.check(t));
    let presented = bearer_token(headers).or_else(|| query_token(&req));
    let from_other = presented.as_deref().filter(|t| remote.check(t)).map(str::to_string);
    let Some(token) = from_cookie.clone().or(from_other) else {
        if presented.is_some() {
            // Tokens are unguessable; this only makes scripted guessing slower.
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let is_api = req.uri().path().starts_with("/api/");
        return if *req.method() == Method::GET && !is_api {
            (StatusCode::UNAUTHORIZED, [(header::CACHE_CONTROL, "no-store")], Html(LOGIN_HTML)).into_response()
        } else {
            deny(StatusCode::UNAUTHORIZED, "access token required")
        };
    };
    // A `?token=` link: keep the token out of the address bar and history.
    if from_cookie.is_none() && *req.method() == Method::GET && query_token(&req).is_some() {
        let mut resp = (StatusCode::SEE_OTHER, [(header::LOCATION, req.uri().path().to_string())]).into_response();
        resp.headers_mut().insert(header::SET_COOKIE, set_cookie(&token));
        resp.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        return resp;
    }
    // CSRF: a state-changing request must come from this same origin.
    let state_changing = !matches!(*req.method(), Method::GET | Method::HEAD);
    if state_changing {
        let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
        let same = match headers.get(header::ORIGIN).and_then(|o| o.to_str().ok()) {
            None => true,
            Some(o) => {
                o.split_once("://").is_some_and(|(_, rest)| rest.trim_end_matches('/').eq_ignore_ascii_case(host))
            }
        };
        if !same {
            return deny(StatusCode::FORBIDDEN, "cross-origin request refused");
        }
    }
    let regenerating = req.uri().path() == "/api/remote/token";
    let mut resp = next.run(req).await;
    if regenerating && resp.status().is_success() {
        // Keep the device that asked for a new token signed in.
        let fresh = remote.settings.read().unwrap().token.clone();
        resp.headers_mut().append(header::SET_COOKIE, set_cookie(&fresh));
    } else if from_cookie.is_none() {
        resp.headers_mut().append(header::SET_COOKIE, set_cookie(&token));
    }
    resp
}

const LOGIN_HTML: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>orchopork · sign in</title>
<style>
:root { --bg: #fbfaf8; --panel: #fff; --line: #e4e2dc; --text: #17181a; --muted: #6e7076; --accent: #5b5bd6; --bad: #c2362b; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d0e11; --panel: #17191e; --line: #262930; --text: #ececf0; --muted: #9194a0; --accent: #8b8cff; --bad: #f87171; } }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 16px; background: var(--bg); color: var(--text); font: 14.5px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
form { width: min(420px, 100%); background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 22px; display: grid; gap: 12px; }
h1 { margin: 0; font: 800 17px ui-monospace, Menlo, Consolas, monospace; }
p { margin: 0; color: var(--muted); font-size: 13px; }
input { width: 100%; font: 13px ui-monospace, Menlo, Consolas, monospace; color: var(--text); background: var(--bg); border: 1px solid var(--line); border-radius: 9px; padding: 10px 11px; }
button { font: 600 14px inherit; padding: 10px; border-radius: 9px; border: 0; background: var(--accent); color: #fff; cursor: pointer; }
.err { color: var(--bad); min-height: 1em; }
</style></head><body>
<form id="f"><h1>orchopork</h1>
<p>Enter the access token. You find it on the machine running orchopork, under Models &amp; settings → Access from other devices.</p>
<input id="t" type="password" autocomplete="current-password" placeholder="Access token" autofocus required>
<button>Sign in</button><p class="err" id="e"></p></form>
<script>
document.getElementById("f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const e = document.getElementById("e"); e.textContent = "";
  const r = await fetch("/api/remote/check", { headers: { authorization: "Bearer " + document.getElementById("t").value.trim() } }).catch(() => null);
  if (r && r.ok) location.reload(); else e.textContent = r && r.status === 401 ? "Wrong token." : "Could not reach orchopork.";
});
</script></body></html>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabling_creates_a_persisted_private_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.yaml");
        let r = Remote::open(Some(path.clone()));
        assert!(!r.enabled());
        r.set_enabled(true).unwrap();
        let tok = r.status()["token"].as_str().unwrap().to_string();
        assert_eq!(tok.len(), 64);
        assert!(r.check(&tok) && !r.check("nope") && !r.check(""));
        let again = Remote::open(Some(path.clone()));
        assert!(again.enabled() && again.check(&tok));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let fresh = again.regenerate().unwrap();
        assert!(!again.check(&tok) && again.check(&fresh));
        again.set_enabled(false).unwrap();
        assert!(!again.check(&fresh), "a token is worthless while access is off");
    }

    #[test]
    fn tokens_are_read_from_cookie_and_bearer() {
        let mut h = HeaderMap::new();
        h.insert(header::COOKIE, HeaderValue::from_static("a=1; orchopork_token=abc; b=2"));
        h.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer xyz"));
        assert_eq!(cookie_token(&h).as_deref(), Some("abc"));
        assert_eq!(bearer_token(&h).as_deref(), Some("xyz"));
    }
}
