//! Who may talk to the server.
//!
//! By default only this machine: the server binds loopback, and a request
//! must come from a loopback address with a local `Host` (and, when it
//! changes state, a local `Origin`). That last part is DNS-rebinding
//! protection: this API can run shell commands through the agents, so a
//! web page must never be able to reach it.
//!
//! `orchopork serve --host 0.0.0.0` opens it to other machines. What they
//! need is a setting (Settings → Remote access, kept in
//! `~/.orchopork/server.json`), off by default:
//!
//! - **Token off:** any machine that can reach the port gets in, as long
//!   as it addresses the server by IP (or `localhost`). A DNS name in
//!   `Host` is refused, which is what stops DNS rebinding: a web page cannot
//!   point its own domain at this machine and talk to the API.
//! - **Token on:** every request that is not local must carry the access
//!   token: once as `?token=…` in the link printed at startup (which sets a
//!   cookie and redirects), then by that cookie, or as
//!   `Authorization: Bearer …` for scripts.
//!
//! Either way state-changing requests must be same-origin, so another site
//! cannot drive a browser that can reach the server.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::Json;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use serde_json::json;

const COOKIE: &str = "orchopork_token";

/// Who may reach the server. `token: None` keeps it local-only (no
/// `--host`); otherwise `require` says whether other machines need it.
#[derive(Clone, Default)]
pub struct Access {
    token: Option<Arc<str>>,
    require: Arc<AtomicBool>,
    /// Where the `require` choice is saved (`None` in tests).
    settings: Option<Arc<PathBuf>>,
    /// The link printed at startup, shown in Settings.
    link_base: Option<Arc<str>>,
}

impl Access {
    pub fn local_only() -> Self {
        Self::default()
    }

    /// Open to other machines; `require` is the saved token setting.
    pub fn remote(token: impl Into<String>, require: bool) -> Self {
        Self { token: Some(Arc::from(token.into())), require: Arc::new(AtomicBool::new(require)), ..Self::default() }
    }

    /// Saves later changes of the token setting to `path`.
    pub fn saved_in(mut self, path: Option<PathBuf>) -> Self {
        self.settings = path.map(Arc::new);
        self
    }

    /// The base URL other machines use (e.g. `http://192.168.1.16:7878`).
    pub fn reachable_at(mut self, base: impl Into<String>) -> Self {
        self.link_base = Some(Arc::from(base.into()));
        self
    }

    pub fn requires_token(&self) -> bool {
        self.token.is_some() && self.require.load(Ordering::SeqCst)
    }
}

/// Server-wide settings, kept next to the token.
pub fn settings_file() -> Option<PathBuf> {
    token_file().map(|t| t.with_file_name("server.json"))
}

/// Whether other machines need the token (default: no).
pub fn load_require_token(path: Option<&std::path::Path>) -> bool {
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["require_token"].as_bool())
        .unwrap_or(false)
}

/// A fresh random token (two v4 UUIDs: 244 random bits).
pub fn new_token() -> String {
    format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())
}

/// Where the token is kept between runs, so the link stays the same.
pub fn token_file() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".orchopork").join("access-token"))
}

/// The token to use: `--token`, else `ORCHOPORK_TOKEN`, else the saved
/// one, else a new one (saved for next time).
pub fn load_or_create(explicit: Option<String>) -> std::io::Result<String> {
    if let Some(t) = explicit.or_else(|| std::env::var("ORCHOPORK_TOKEN").ok()).filter(|t| !t.trim().is_empty()) {
        return Ok(t.trim().to_string());
    }
    let Some(path) = token_file() else { return Ok(new_token()) };
    if let Ok(t) = std::fs::read_to_string(&path)
        && t.trim().len() >= 32
    {
        return Ok(t.trim().to_string());
    }
    let t = new_token();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::fsutil::write_atomic(&path, t.as_bytes(), true).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(t)
}

/// The host part of a `Host` header (`[::1]:80` → `::1`).
fn host_name(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.rsplit_once(':').map_or(host, |(h, p)| if p.bytes().all(|b| b.is_ascii_digit()) { h } else { host })
    }
}

pub(super) fn is_local_host(host: &str) -> bool {
    matches!(host_name(host).to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1")
}

/// An IP address or `localhost`: never a name a web page could point at
/// this machine.
fn is_ip_host(host: &str) -> bool {
    is_local_host(host) || host_name(host).parse::<std::net::IpAddr>().is_ok()
}

/// Compares without an early exit, so timing does not leak the token.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn query_token(req: &Request) -> Option<String> {
    req.uri().query()?.split('&').find_map(|kv| kv.strip_prefix("token=")).map(str::to_string)
}

fn presented(req: &Request) -> Option<String> {
    let h = req.headers();
    let bearer = h
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|t| t.trim().to_string());
    let cookie =
        h.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).find_map(|c| {
            c.split(';').find_map(|kv| kv.trim().strip_prefix(&format!("{COOKIE}=")).map(str::to_string))
        });
    bearer.or(cookie).or_else(|| query_token(req))
}

fn forbidden(msg: &str) -> Response {
    (StatusCode::FORBIDDEN, Json(json!({ "error": msg }))).into_response()
}

pub async fn guard(State(access): State<Access>, req: Request, next: Next) -> Response {
    let headers = req.headers();
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("").to_string();
    let origin = headers.get(header::ORIGIN).and_then(|h| h.to_str().ok()).map(|o| o.trim_end_matches('/').to_string());
    let state_changing = !matches!(*req.method(), Method::GET | Method::HEAD);
    // Without connection info (the router used in tests) the peer counts
    // as local; the real server always records it.
    let peer_local =
        req.extensions().get::<ConnectInfo<SocketAddr>>().is_none_or(|ConnectInfo(addr)| addr.ip().is_loopback());

    let origin_local = origin.as_deref().is_none_or(|o| o.split_once("://").is_some_and(|(_, r)| is_local_host(r)));
    if peer_local && is_local_host(&host) && (!state_changing || origin_local) {
        return next.run(req).await;
    }
    let Some(token) = access.token.clone() else {
        return forbidden("orchopork only accepts local requests (start it with --host to allow others)");
    };
    let has_token = presented(&req).is_some_and(|t| same(&t, &token));
    if access.requires_token() {
        if !has_token {
            return forbidden("orchopork needs its access token: open the link printed when it started");
        }
    } else if !is_ip_host(&host) {
        return forbidden("open orchopork by its IP address (or turn on the access token in Settings)");
    }
    // Same-origin only for changes: another site must not act through a
    // browser that can reach this server.
    let same_origin = origin.as_deref().is_none_or(|o| o.split_once("://").is_some_and(|(_, r)| r == host));
    if state_changing && !same_origin {
        return forbidden("cross-origin requests are not accepted");
    }
    // The startup link: keep the token in a cookie and drop it from the URL.
    if has_token && query_token(&req).is_some() && req.method() == Method::GET {
        let mut resp = Redirect::to(req.uri().path()).into_response();
        set_cookie(&mut resp, &token);
        return resp;
    }
    next.run(req).await
}

fn set_cookie(resp: &mut Response, token: &str) {
    let cookie = format!("{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000");
    if let Ok(v) = cookie.parse() {
        resp.headers_mut().insert(header::SET_COOKIE, v);
    }
}

/// `GET /api/access`: what Settings shows.
pub async fn get_access(State(access): State<Access>) -> Response {
    Json(describe(&access)).into_response()
}

fn describe(a: &Access) -> serde_json::Value {
    json!({
        "remote": a.token.is_some(),
        "require_token": a.requires_token(),
        "url": a.link_base.as_deref(),
        "link": match (&a.link_base, &a.token) {
            (Some(b), Some(t)) => Some(format!("{b}/?token={t}")),
            _ => None,
        },
        "token_file": token_file(),
    })
}

#[derive(serde::Deserialize)]
pub struct AccessReq {
    require_token: bool,
}

/// `PUT /api/access`: turns the token check on or off, and saves it.
pub async fn put_access(State(access): State<Access>, Json(req): Json<AccessReq>) -> Response {
    let Some(token) = access.token.clone() else {
        return (StatusCode::CONFLICT, Json(json!({ "error": "start orchopork with --host to allow other machines" })))
            .into_response();
    };
    if let Some(path) = access.settings.as_deref() {
        let body = json!({ "require_token": req.require_token }).to_string();
        if let Err(e) = crate::fsutil::write_atomic(path, body.as_bytes(), true) {
            return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))).into_response();
        }
    }
    access.require.store(req.require_token, Ordering::SeqCst);
    let mut resp = Json(describe(&access)).into_response();
    // Whoever turned it on keeps their own access.
    if req.require_token {
        set_cookie(&mut resp, &token);
    }
    resp
}

/// This machine's address on the local network, for the startup link
/// (no packet is sent: connecting a UDP socket only picks a route).
pub fn lan_ip() -> Option<std::net::IpAddr> {
    let s = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("192.0.2.1:9").ok()?;
    s.local_addr().ok().map(|a| a.ip()).filter(|ip| !ip.is_loopback() && !ip.is_unspecified())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_ip_hosts_pass_without_a_token() {
        for h in ["192.168.1.16:7878", "10.0.0.2", "[fe80::1]:7878", "localhost:7878"] {
            assert!(is_ip_host(h), "{h}");
        }
        for h in ["evil.example:7878", "mybox.local", ""] {
            assert!(!is_ip_host(h), "{h}");
        }
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("server.json");
        assert!(!load_require_token(Some(&f)), "off by default");
        std::fs::write(&f, r#"{"require_token": true}"#).unwrap();
        assert!(load_require_token(Some(&f)));
    }

    #[test]
    fn tokens_compare_exactly() {
        assert!(same("abc", "abc"));
        assert!(!same("abc", "abd") && !same("abc", "ab") && !same("", "a"));
        let t = new_token();
        assert_eq!(t.len(), 64);
        assert_ne!(t, new_token());
    }
}
