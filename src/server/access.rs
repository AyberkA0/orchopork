//! Who may talk to the server.
//!
//! By default only this machine: the server binds loopback, and a request
//! must come from a loopback address with a local `Host` (and, when it
//! changes state, a local `Origin`). That last part is DNS-rebinding
//! protection: this API can run shell commands through the agents, so a
//! web page must never be able to reach it.
//!
//! `orchopork serve --host 0.0.0.0` opens it to other machines, and then
//! every request that is not local must carry the access token: once as
//! `?token=…` in the link printed at startup (which sets a cookie and
//! redirects), then by that cookie, or as `Authorization: Bearer …` for
//! scripts. State-changing requests must also be same-origin, so another
//! site cannot drive a browser that holds the cookie.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Json;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};
use serde_json::json;

const COOKIE: &str = "orchopork_token";

/// The access token for non-local requests; `None` keeps the server
/// local-only.
#[derive(Clone, Default)]
pub struct Access(pub Option<Arc<str>>);

impl Access {
    pub fn local_only() -> Self {
        Self(None)
    }

    pub fn with_token(token: impl Into<String>) -> Self {
        Self(Some(Arc::from(token.into())))
    }
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

pub(super) fn is_local_host(host: &str) -> bool {
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.rsplit_once(':').map_or(host, |(h, p)| if p.bytes().all(|b| b.is_ascii_digit()) { h } else { host })
    };
    matches!(name.to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1")
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
    let Some(token) = access.0.clone() else {
        return forbidden("orchopork only accepts local requests (start it with --host to allow others)");
    };
    if !presented(&req).is_some_and(|t| same(&t, &token)) {
        return forbidden("orchopork needs its access token: open the link printed when it started");
    }
    // Same-origin only for changes: the cookie must not let another site
    // act through this browser.
    let same_origin = origin.as_deref().is_none_or(|o| o.split_once("://").is_some_and(|(_, r)| r == host));
    if state_changing && !same_origin {
        return forbidden("cross-origin requests are not accepted");
    }
    // The startup link: keep the token in a cookie and drop it from the URL.
    if query_token(&req).is_some() && req.method() == Method::GET {
        let cookie = format!("{COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000");
        let mut resp = Redirect::to(req.uri().path()).into_response();
        if let Ok(v) = cookie.parse() {
            resp.headers_mut().insert(header::SET_COOKIE, v);
        }
        return resp;
    }
    next.run(req).await
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
    fn tokens_compare_exactly() {
        assert!(same("abc", "abc"));
        assert!(!same("abc", "abd") && !same("abc", "ab") && !same("", "a"));
        let t = new_token();
        assert_eq!(t.len(), 64);
        assert_ne!(t, new_token());
    }
}
