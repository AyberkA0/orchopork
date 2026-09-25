//! Guided setup helpers: a folder browser for picking the project, and
//! detection of what is already on this machine (API keys in the
//! environment, local model servers, agent CLIs) so the UI can offer
//! one-click connections instead of forms.

use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::Json;
use axum::extract::{Query, State};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiResult, Shared, expand_home};
use crate::error::Error;
use crate::git::{GitRepo, STATE_DIR};
use crate::providers::ProviderId;

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

#[derive(Deserialize)]
pub(super) struct BrowseReq {
    #[serde(default)]
    path: String,
}

/// Lists the sub-folders of `path` (default: home) for the project picker.
/// Hidden folders are skipped; each entry says whether it is a git repo
/// root or an orchopork project already.
pub(super) async fn browse(Query(q): Query<BrowseReq>) -> ApiResult {
    let raw =
        if q.path.trim().is_empty() { home().unwrap_or_else(|| PathBuf::from("/")) } else { expand_home(&q.path) };
    let path = crate::fsutil::canonical(&raw).map_err(|e| Error::InvalidRequest(format!("{}: {e}", raw.display())))?;
    if !path.is_dir() {
        return Err(Error::InvalidRequest(format!("{} is not a folder", path.display())).into());
    }
    let mut dirs = Vec::new();
    let mut rd = tokio::fs::read_dir(&path).await.map_err(Error::from)?;
    while let Some(e) = rd.next_entry().await.map_err(Error::from)? {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || !e.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let p = e.path();
        dirs.push(json!({
            "name": name,
            "path": p,
            "repo": p.join(".git").exists(),
            "project": p.join(STATE_DIR).join("config.yaml").is_file(),
        }));
    }
    dirs.sort_by_key(|d| d["name"].as_str().unwrap_or("").to_lowercase());
    dirs.truncate(500);
    Ok(Json(json!({
        "path": path,
        "parent": path.parent(),
        "home": home(),
        "repo": GitRepo::new(&path).is_repo().await,
        "dirs": dirs,
        "roots": drive_roots(),
    })))
}

fn drive_roots() -> Vec<String> {
    if cfg!(windows) {
        (b'A'..=b'Z').map(|c| format!("{}:\\", c as char)).filter(|d| Path::new(d).exists()).collect()
    } else {
        vec!["/".into()]
    }
}

#[derive(Deserialize)]
pub(super) struct MkdirReq {
    parent: String,
    name: String,
}

pub(super) async fn mkdir(Json(req): Json<MkdirReq>) -> ApiResult {
    let name = req.name.trim();
    if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
        return Err(Error::InvalidRequest("enter a plain folder name".into()).into());
    }
    let parent = crate::fsutil::canonical(&expand_home(&req.parent))
        .map_err(|e| Error::InvalidRequest(format!("{}: {e}", req.parent.trim())))?;
    let path = parent.join(name);
    if path.exists() {
        return Err(Error::Conflict(format!("{} already exists", path.display())).into());
    }
    tokio::fs::create_dir(&path).await.map_err(Error::from)?;
    Ok(Json(json!({ "path": path })))
}

const KEY_ENVS: &[(ProviderId, &[&str])] = &[
    (ProviderId::Claude, &["ANTHROPIC_API_KEY"]),
    (ProviderId::Gemini, &["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    (ProviderId::DeepSeek, &["DEEPSEEK_API_KEY"]),
];

fn env_key(p: ProviderId) -> Option<(&'static str, String)> {
    KEY_ENVS
        .iter()
        .find(|(id, _)| *id == p)?
        .1
        .iter()
        .find_map(|v| std::env::var(v).ok().map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).map(|k| (*v, k)))
}

/// Well-known local OpenAI-compatible servers, probed on `/v1/models`.
const LOCAL_SERVERS: &[(&str, &str)] = &[
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
    ("vLLM", "http://127.0.0.1:8000/v1"),
    ("Jan", "http://127.0.0.1:1337/v1"),
];

async fn reachable(url: &str) -> bool {
    let c = reqwest::Client::builder().timeout(Duration::from_millis(800)).build();
    match c {
        Ok(c) => c.get(url).send().await.map(|r| r.status().is_success()).unwrap_or(false),
        Err(_) => false,
    }
}

/// Everything on this machine that could be connected with one click.
pub(super) async fn detect(State(s): State<Shared>) -> ApiResult {
    let engine = s.engine()?;
    let ws = engine.workspace();
    let cfg = ws.config();
    let configured = ws.gateway.configured();

    let keys: Vec<Value> = KEY_ENVS
        .iter()
        .filter(|(p, _)| !configured.contains(p))
        .filter_map(|(p, _)| env_key(*p).map(|(var, _)| json!({ "provider": p, "env": var })))
        .collect();

    let ollama_models = ws.gateway.provider(ProviderId::Ollama).ok();
    let ollama = match ollama_models {
        Some(p) => tokio::time::timeout(Duration::from_secs(2), p.list_models()).await.ok().and_then(|r| r.ok()),
        None => None,
    };

    let known: Vec<String> = cfg
        .endpoints
        .iter()
        .map(|e| e.base_url.trim_end_matches('/').to_string())
        .chain(cfg.llamacpp_url.iter().map(|u| format!("{}/v1", u.trim_end_matches('/'))))
        .collect();
    let probes = LOCAL_SERVERS.iter().filter(|(_, u)| !known.iter().any(|k| k == u)).map(|(name, url)| async move {
        reachable(&format!("{url}/models")).await.then(|| json!({ "name": name, "base_url": url }))
    });
    let servers: Vec<Value> = futures_join(probes).await.into_iter().flatten().collect();

    let node = crate::acp::resolve_command("npx").is_some();
    let agents: Vec<Value> = cfg
        .external_agents
        .iter()
        .map(|a| {
            let setup = crate::acp::setup_for(&a.id);
            let launcher = crate::acp::resolve_command(&a.command).is_some();
            let cli = setup.as_ref().map(|s| crate::acp::resolve_command(s.cli).is_some());
            // npx-launched adapters also need the tool itself for its login.
            let ready = launcher && cli.unwrap_or(true);
            json!({
                "id": a.id,
                "name": a.name,
                "ready": ready,
                "launcher": launcher,
                "cli_installed": cli,
                "node": node,
                "install": setup.as_ref().map(|s| s.install.join(" ")),
                "login": setup.as_ref().map(|s| s.login),
                "key_env_set": setup.as_ref().is_some_and(|s| std::env::var_os(s.key_env).is_some()),
            })
        })
        .collect();

    Ok(Json(json!({
        "keys": keys,
        "ollama": { "url": cfg.ollama_url, "running": ollama.is_some(), "models": ollama.unwrap_or_default() },
        "servers": servers,
        "agents": agents,
        "node": node,
    })))
}

async fn futures_join<F: std::future::Future<Output = T> + Send + 'static, T: Send + 'static>(
    fs: impl Iterator<Item = F>,
) -> Vec<T> {
    let handles: Vec<_> = fs.map(tokio::spawn).collect();
    let mut out = Vec::new();
    for h in handles {
        if let Ok(v) = h.await {
            out.push(v);
        }
    }
    out
}

#[derive(Deserialize)]
pub(super) struct ImportKeyReq {
    provider: ProviderId,
}

/// Connects a cloud provider with the key found in the environment, so the
/// key never has to pass through the browser.
pub(super) async fn import_env_key(State(s): State<Shared>, Json(req): Json<ImportKeyReq>) -> ApiResult {
    let (_, key) = env_key(req.provider)
        .ok_or_else(|| Error::InvalidRequest(format!("no API key for {} in the environment", req.provider.as_str())))?;
    super::add_cloud_key(&s, req.provider, key).await
}

#[derive(Deserialize)]
pub(super) struct InstallReq {
    id: String,
}

/// Installs a preset agent's CLI with its fixed npm command.
pub(super) async fn install_agent(State(s): State<Shared>, Json(req): Json<InstallReq>) -> ApiResult {
    let log = crate::acp::install(&req.id).await?;
    s.models_changed(); // the agent's launcher is on PATH now
    let tail: Vec<&str> = log.lines().rev().take(6).collect();
    Ok(Json(json!({ "ok": true, "log": tail.into_iter().rev().collect::<Vec<_>>().join("\n") })))
}
