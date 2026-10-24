//! Local HTTP API + embedded dashboard. Binds loopback only, and rejects
//! requests whose `Host`/`Origin` is not local (DNS-rebinding protection:
//! this API can run shell commands through the agent, so a web page must
//! never be able to reach it).

mod outputs;
mod setup;
pub mod tokens;
pub mod wizard;

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use axum::extract::{Path as UrlPath, Query, Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::{Stream, StreamExt};
use tower_http::trace::TraceLayer;

use crate::bootstrap::Workspace;
use crate::config::{Config, RemoteAuth, Routing};
use crate::error::{Error, Result};
use crate::graph::Engine;
use crate::providers::{OllamaProvider, OpenAiCompatProvider, Provider, ProviderId, pricing};
use crate::secrets::GITHUB;
use crate::storage::RunSpec;
use tokens::TokenStore;
use wizard::{Event, VcsStatus, Wizard};

/// Single-file embedded SPA; client JS routes off `wizard.step`.
const INDEX_HTML: &str = include_str!("../../ui/index.html");

pub struct AppState {
    /// Live model lists of cloud providers, refreshed every 10 minutes.
    cloud_models: tokio::sync::Mutex<std::collections::HashMap<ProviderId, (std::time::Instant, Vec<String>)>>,
    /// Discovered ACP session options per agent id (discovery spawns the
    /// agent, so it is done once per server run unless refreshed).
    agent_options: tokio::sync::Mutex<std::collections::HashMap<String, Value>>,
    /// The last `/api/models` answer: (when, fingerprint of what it depends
    /// on, body). Pickers ask for it often; rescanning every provider each
    /// time is what made them slow.
    models_cache: Mutex<Option<(std::time::Instant, String, Value)>>,
    engine: RwLock<Option<Engine>>,
    wizard: Mutex<Wizard>,
    /// Pre-filled in the workspace picker (the `--workspace` argument).
    default_root: Option<PathBuf>,
    pub tokens: Arc<TokenStore>,
    /// Whether other machines need an access token (Settings → Remote
    /// access; off by default). Saved in `access_file`.
    require_token: std::sync::atomic::AtomicBool,
    access_file: PathBuf,
    /// The base URL other machines use (`http://192.168.1.16:7878`), when
    /// the server listens beyond this machine.
    remote_base: Mutex<Option<String>>,
}

pub type Shared = Arc<AppState>;

/// The cookie that keeps the access token after the startup link was opened.
const TOKEN_COOKIE: &str = "orchopork_token";

/// Reads the saved "require an access token" setting (default: off).
fn load_require_token(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v["require_token"].as_bool())
        .unwrap_or(false)
}

impl AppState {
    /// If `default_root` is an already-onboarded workspace, bind it and go
    /// straight to the dashboard; otherwise start the wizard and touch
    /// nothing on disk until the user picks a directory.
    /// Records that other machines reach the server at `base`.
    pub fn set_remote_base(&self, base: String) {
        *self.remote_base.lock().unwrap() = Some(base);
    }

    /// The token the Remote access card links with: the first one, made on
    /// first use.
    fn link_token(&self) -> Result<String> {
        match self.tokens.list().into_iter().next() {
            Some(t) => Ok(t.token),
            None => Ok(self.tokens.create("default")?.token),
        }
    }

    pub async fn new(default_root: Option<PathBuf>) -> Result<Shared> {
        // Tests must never mint real remote-access tokens in the user's home.
        let tokens_path = if cfg!(test) {
            std::env::temp_dir().join(format!("orchopork-test-tokens-{}.json", uuid::Uuid::new_v4().simple()))
        } else {
            expand_home("~/.orchopork/tokens.json")
        };
        let access_file = if cfg!(test) {
            std::env::temp_dir().join(format!("orchopork-test-server-{}.json", uuid::Uuid::new_v4().simple()))
        } else {
            expand_home("~/.orchopork/server.json")
        };
        let state = Arc::new(Self {
            cloud_models: Default::default(),
            models_cache: Mutex::new(None),
            agent_options: Default::default(),
            engine: RwLock::new(None),
            wizard: Mutex::new(Wizard::default()),
            default_root: default_root.clone(),
            tokens: Arc::new(TokenStore::open(tokens_path)?),
            require_token: std::sync::atomic::AtomicBool::new(load_require_token(&access_file)),
            access_file,
            remote_base: Mutex::new(None),
        });
        if let Some(ref root) = default_root
            && Workspace::is_initialized(root) {
                let ws = Workspace::open(root).await?;
                if ws.config().onboarded || ws.repo.is_repo().await {
                    let engine = Engine::new(ws).await?;
                    *state.engine.write().unwrap() = Some(engine);
                    state.wizard.lock().unwrap().apply(Event::Resume)?;
                }
            }
        Ok(state)
    }

    /// Known models first (curated order), then anything else the
    /// provider's API lists that looks like a chat model.
    async fn cloud_models(&self, ws: &Workspace, id: ProviderId) -> Vec<String> {
        let mut out: Vec<String> = pricing::suggested_models(id).into_iter().map(String::from).collect();
        // The lock is not held across the network call, so one slow provider
        // never blocks the others.
        let fresh = self
            .cloud_models
            .lock()
            .await
            .get(&id)
            .filter(|(t, _)| t.elapsed() < std::time::Duration::from_secs(600))
            .map(|(_, v)| v.clone());
        let live = match fresh {
            Some(v) => v,
            None => {
                let listed = match ws.gateway.provider(id) {
                    Ok(p) => tokio::time::timeout(std::time::Duration::from_secs(5), p.list_models())
                        .await
                        .ok()
                        .and_then(|r| r.ok()),
                    Err(_) => None,
                };
                let v: Vec<String> = listed
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|m| match id {
                        ProviderId::Claude => m.starts_with("claude-"),
                        ProviderId::Gemini => m.contains("gemini") && !m.contains("embedding") && !m.contains("image"),
                        _ => true,
                    })
                    .collect();
                self.cloud_models.lock().await.insert(id, (std::time::Instant::now(), v.clone()));
                v
            }
        };
        for m in live {
            if !out.contains(&m) {
                out.push(m);
            }
        }
        out
    }

    /// Forget the cached model list (after models were added or removed).
    fn models_changed(&self) {
        *self.models_cache.lock().unwrap() = None;
    }

    fn engine(&self) -> Result<Engine> {
        self.engine
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| Error::Conflict("no workspace is selected yet; finish setup first".into()))
    }

    /// Binds `root` as the active workspace, reusing the current engine
    /// when it is the same directory.
    async fn bind(&self, root: &Path) -> Result<Engine> {
        let current = self.engine.read().unwrap().clone();
        if let Some(e) = current {
            if e.workspace().root == root {
                return Ok(e);
            }
            for r in e.list_runs().await? {
                if e.is_active(&r.id) {
                    return Err(Error::Conflict(
                        "runs are executing in the current workspace; pause them first".into(),
                    ));
                }
            }
        }
        let engine = Engine::new(Workspace::open(root).await?).await?;
        *self.engine.write().unwrap() = Some(engine.clone());
        Ok(engine)
    }
}

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/api/state", get(get_state))
        .route("/api/wizard/initialize", post(wizard_initialize))
        .route("/api/wizard/workspace", post(wizard_workspace))
        .route("/api/wizard/vcs", post(wizard_vcs))
        .route("/api/wizard/finish", post(wizard_finish))
        .route("/api/wizard/back", post(wizard_back))
        .route("/api/config", axum::routing::put(put_config))
        .route("/api/workspace", post(open_workspace))
        .route("/api/fs/browse", get(setup::browse))
        .route("/api/fs/mkdir", post(setup::mkdir))
        .route("/api/setup/detect", get(setup::detect))
        .route("/api/setup/import-key", post(setup::import_env_key))
        .route("/api/agents/install", post(setup::install_agent))
        .route("/api/agents/add-preset", post(setup::add_preset))
        .route("/api/models", get(available_models))
        .route("/api/agents/probe", post(probe_agent))
        .route("/api/agents/options", post(agent_options))
        .route("/api/models/add", post(add_model))
        .route("/api/models/remove", post(remove_model))
        .route("/api/ollama/pull", post(ollama_pull))
        .route("/api/orchestra/propose", post(propose_team))
        .route("/api/providers/keys", post(set_provider_key))
        .route("/api/providers/{id}/models", get(provider_models))
        .route("/api/skills", get(list_skills))
        .route("/api/skills/reload", post(reload_skills))
        .route("/api/skills/toggle", post(toggle_skill))
        .route("/api/runs", get(list_runs).post(create_run))
        .route("/api/runs/{id}", get(get_run).delete(delete_run))
        .route("/api/runs/{id}/pause", post(pause_run))
        .route("/api/runs/{id}/resume", post(resume_run))
        .route("/api/runs/{id}/inject", post(inject_run))
        .route("/api/runs/{id}/rewind", post(rewind_run))
        .route("/api/runs/{id}/diff", get(diff_run))
        .route("/api/runs/{id}/answer", post(answer_run))
        .route("/api/runs/{id}/files", get(outputs::files))
        .route("/api/runs/{id}/file", get(outputs::file))
        .route("/api/runs/{id}/raw/{*path}", get(outputs::raw))
        .route("/api/runs/{id}/metrics", get(outputs::metrics))
        .route("/api/metrics", get(outputs::compare))
        .route("/api/runs/{id}/push", post(push_run))
        .route("/api/runs/{id}/merge", post(merge_run))
        .route("/api/runs/{id}/approval", post(decide_approval))
        .route("/api/runs/{id}/kill", post(kill_run))
        .route("/api/runs/{id}/steps/{seq}/diff", get(step_diff))
        .route("/api/runs/{id}/lessons", post(suggest_lessons))
        .route("/api/lessons", post(save_lessons))
        .route("/api/orchestra/preflight", post(preflight))
        .route("/api/runs/{id}/reorchestrate", post(reorchestrate))
        .route("/api/runs/{id}/flow", axum::routing::put(edit_flow))
        .route("/api/orchestra/criteria", post(detail_criteria))
        .route("/api/runs/{id}/autonomy", post(set_autonomy))
        .route("/api/runs/{id}/questions", post(set_questions))
        .route("/api/browser/status", get(browser_status))
        .route("/preview/{id}", get(preview_root))
        .route("/preview/{id}/", get(preview_index))
        .route("/preview/{id}/{*path}", get(preview_file))
        .route("/pvt/{token}/{id}", get(preview_root_t))
        .route("/pvt/{token}/{id}/", get(preview_index_t))
        .route("/pvt/{token}/{id}/{*path}", get(preview_file_t))
        .route("/api/tokens", get(list_tokens).post(create_token))
        .route("/api/access", get(get_access).put(put_access))
        .route("/api/tokens/{id}", delete(delete_token))
        .route("/api/events", get(events))
        // Client-side-routed SPA: every other GET gets the same shell.
        .fallback(get(spa_index))
        .layer(middleware::from_fn_with_state(state.clone(), local_or_token_auth))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// The host part of a `Host` header (`[::1]:80` → `::1`).
fn host_name(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.rsplit_once(':').map_or(host, |(h, p)| if p.bytes().all(|b| b.is_ascii_digit()) { h } else { host })
    }
}

fn is_local_host(host: &str) -> bool {
    matches!(host_name(host).to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1")
}

/// An IP address or `localhost`: never a name a web page could point at
/// this machine (DNS rebinding), so it is safe to serve without a token.
fn is_ip_host(host: &str) -> bool {
    is_local_host(host) || host_name(host).parse::<std::net::IpAddr>().is_ok()
}

async fn local_or_token_auth(State(s): State<Shared>, req: Request, next: Next) -> Response {
    let headers = req.headers();
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    // A reverse proxy or port forward on this machine may rewrite `Host` to
    // localhost, so the socket peer must be loopback too. (No peer info, as
    // in unit tests, counts as loopback.)
    let peer_loopback = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .is_none_or(|c| c.0.ip().to_canonical().is_loopback());
    let is_local = is_local_host(host) && peer_loopback;

    // 1. Extract token from header or query parameter
    let mut token_val = None;
    if let Some(auth) = headers.get(header::AUTHORIZATION).and_then(|a| a.to_str().ok())
        && let Some(bearer) = auth.strip_prefix("Bearer ").or_else(|| auth.strip_prefix("bearer ")) {
            token_val = Some(bearer.trim().to_string());
        }
    if token_val.is_none()
        && let Some(t) = headers.get("x-orchopork-token").and_then(|a| a.to_str().ok()) {
            token_val = Some(t.trim().to_string());
        }
    if token_val.is_none() {
        token_val = headers.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).find_map(|c| {
            c.split(';').find_map(|kv| kv.trim().strip_prefix(&format!("{TOKEN_COOKIE}=")).map(str::to_string))
        });
    }
    let from_link = req.uri().query().is_some_and(|q| q.split('&').any(|kv| kv.starts_with("token=")));
    // Remote previews carry the token in the path (`/pvt/<token>/<run>/…`),
    // so relative subresources of a sandboxed page stay authenticated.
    if token_val.is_none()
        && let Some(rest) = req.uri().path().strip_prefix("/pvt/")
    {
        token_val = rest.split('/').next().map(str::to_string);
    }
    if token_val.is_none()
        && let Some(q) = req.uri().query() {
            for pair in q.split('&') {
                if let Some((k, v)) = pair.split_once('=')
                    && k == "token" {
                        token_val = Some(v.to_string());
                        break;
                    }
            }
        }

    let is_authenticated = token_val.as_ref().is_some_and(|t| s.tokens.validate(t));

    // Run previews are sandboxed pages with an opaque origin, which the
    // browser reports as `Origin: null`. A remote preview's URL carries the
    // token (`/pvt/<token>/…`), so agent-written JavaScript could read it and
    // call the API with it. The dashboard itself never sends `null`.
    let opaque_origin = headers.get(header::ORIGIN).is_some_and(|o| o.as_bytes().eq_ignore_ascii_case(b"null"));
    if opaque_origin && req.uri().path().starts_with("/api/") {
        return (StatusCode::FORBIDDEN, Json(json!({ "error": "sandboxed pages cannot call the API" }))).into_response();
    }

    // Token management, and whether tokens are needed at all, is for the
    // machine owner only: a remote client must not be able to list, mint
    // or revoke tokens, or lock others (or itself) out.
    let owner_only = req.uri().path().starts_with("/api/tokens")
        || (req.uri().path() == "/api/access" && *req.method() != Method::GET);
    if !is_local && owner_only {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "access tokens can only be managed from the local machine (localhost)" })),
        )
            .into_response();
    }

    // If request has valid token: allow access from anywhere. The startup
    // link (`/?token=…`) leaves the token in a cookie and drops it from the URL.
    if is_authenticated {
        let path = req.uri().path();
        if from_link && *req.method() == Method::GET && !path.starts_with("/api/") && !path.starts_with("/pvt/") {
            let token = token_val.unwrap_or_default();
            let mut resp = axum::response::Redirect::to(path).into_response();
            let cookie = format!("{TOKEN_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000");
            if let Ok(v) = cookie.parse() {
                resp.headers_mut().insert(header::SET_COOKIE, v);
            }
            return resp;
        }
        return next.run(req).await;
    }

    // If local and not authenticated:
    if is_local {
        let state_changing = !matches!(*req.method(), Method::GET | Method::HEAD);
        let origin_ok = match headers.get(header::ORIGIN).and_then(|h| h.to_str().ok()) {
            None => true,
            Some(o) => o.split_once("://").is_some_and(|(_, rest)| is_local_host(rest.trim_end_matches('/'))),
        };
        if !state_changing || origin_ok {
            return next.run(req).await;
        }
    }

    // Token check off (the default): other machines get in by IP address.
    // A host name is refused, which keeps DNS rebinding out, and changes
    // must come from orchopork's own page.
    if !is_local && !s.require_token.load(std::sync::atomic::Ordering::SeqCst) && is_ip_host(host) {
        let state_changing = !matches!(*req.method(), Method::GET | Method::HEAD);
        let same_origin = headers
            .get(header::ORIGIN)
            .and_then(|h| h.to_str().ok())
            .is_none_or(|o| o.split_once("://").is_some_and(|(_, rest)| rest.trim_end_matches('/') == host));
        if !state_changing || same_origin {
            return next.run(req).await;
        }
    }

    // Allow serving the HTML shell / assets for non-API GET requests so browser can prompt user for token
    let path = req.uri().path();
    let protected = ["/api/", "/preview/", "/pvt/"].iter().any(|p| path.starts_with(p));
    if !protected && matches!(*req.method(), Method::GET | Method::HEAD) {
        return next.run(req).await;
    }

    (
        StatusCode::FORBIDDEN,
        Json(json!({ "error": "orchopork only accepts local requests or requests with a valid access token" })),
    )
        .into_response()
}

struct ApiError(Error);

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        Self(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = match self.0 {
            Error::Wizard(_) | Error::Skill(_) | Error::SkillFile { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Error::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            Error::NotFound(_) => StatusCode::NOT_FOUND,
            Error::Conflict(_) => StatusCode::CONFLICT,
            Error::Budget(_) => StatusCode::PAYMENT_REQUIRED,
            Error::Provider(_) => StatusCode::BAD_GATEWAY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (code, Json(json!({ "error": self.0.to_string() }))).into_response()
    }
}

type ApiResult<T = Json<Value>> = std::result::Result<T, ApiError>;

// ---- overall state -----------------------------------------------------------

async fn vcs_status(ws: &Workspace) -> VcsStatus {
    let is_repo = ws.repo.is_repo().await;
    VcsStatus {
        is_repo,
        has_commits: is_repo && ws.repo.head().await.ok().flatten().is_some(),
        branch: if is_repo { ws.repo.current_branch().await } else { None },
        origin: if is_repo { ws.repo.remote_url("origin").await } else { None },
        remote_auth: ws.config().remote_auth,
        has_github_token: ws.secrets.has(GITHUB),
    }
}

fn providers_json(ws: &Workspace) -> Vec<Value> {
    let registered = ws.gateway.configured();
    ProviderId::ALL
        .iter()
        .map(|p| {
            json!({
                "id": p.as_str(),
                "local": p.is_local(),
                "configured": registered.contains(p),
                "suggested_models": pricing::suggested_models(*p),
            })
        })
        .collect()
}

async fn budget_json(ws: &Workspace) -> Result<Value> {
    Ok(json!({
        "cap_usd": ws.gateway.cap(),
        "spent_usd": ws.gateway.spent_this_month().await?,
        "remaining_usd": ws.gateway.budget_remaining().await?,
    }))
}

/// Everything the UI needs to render any page, in one call.
async fn get_state(State(s): State<Shared>) -> ApiResult {
    let wizard = s.wizard.lock().unwrap().clone();
    let engine = s.engine.read().unwrap().clone();
    let def_root = s.default_root.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
    let Some(engine) = engine else {
        return Ok(Json(json!({ "wizard": wizard, "default_root": def_root, "workspace": null })));
    };
    let ws = engine.workspace();
    Ok(Json(json!({
        "wizard": wizard,
        "default_root": def_root,
        "workspace": {
            "root": ws.root,
            "vcs": vcs_status(ws).await,
            "config": ws.config(),
            "providers": providers_json(ws),
            "budget": budget_json(ws).await?,
            "skills": ws.skills.list(),
            "skill_errors": skill_errors(ws),
        },
    })))
}

fn skill_errors(ws: &Workspace) -> Vec<Value> {
    ws.skills.load_errors().into_iter().map(|(p, e)| json!({ "path": p, "error": e })).collect()
}

// ---- wizard -------------------------------------------------------------------

fn apply(s: &Shared, ev: Event) -> ApiResult<Json<Wizard>> {
    let mut w = s.wizard.lock().unwrap();
    w.apply(ev)?;
    Ok(Json(w.clone()))
}

async fn wizard_initialize(State(s): State<Shared>) -> ApiResult<Json<Wizard>> {
    apply(&s, Event::Initialize)
}

async fn wizard_back(State(s): State<Shared>) -> ApiResult<Json<Wizard>> {
    apply(&s, Event::Back)
}

#[derive(Deserialize)]
struct WorkspaceReq {
    path: String,
}

fn expand_home(p: &str) -> PathBuf {
    let p = p.trim();
    if let Some(rest) = p.strip_prefix("~/").or_else(|| (p == "~").then_some(""))
        && let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(p)
}

async fn wizard_workspace(State(s): State<Shared>, Json(req): Json<WorkspaceReq>) -> ApiResult<Json<Wizard>> {
    if s.wizard.lock().unwrap().step != wizard::Step::Permissions {
        return Err(Error::Wizard("not at the workspace step".into()).into());
    }
    let path = crate::fsutil::canonical(&expand_home(&req.path))
        .map_err(|e| Error::InvalidRequest(format!("{}: {e}", req.path.trim())))?;
    if !path.is_dir() {
        return Err(Error::InvalidRequest(format!("{} is not a directory", path.display())).into());
    }
    let report = wizard::check_permissions(&path).await;
    if !report.ok() {
        return apply(&s, Event::PermissionsChecked { report });
    }
    let engine = s.bind(&path).await?;
    let ws = engine.workspace();
    let status = vcs_status(ws).await;
    let mut w = s.wizard.lock().unwrap();
    w.apply(Event::PermissionsChecked { report })?;
    w.vcs = Some(status);
    if ws.config().onboarded {
        w.apply(Event::Resume)?;
    }
    Ok(Json(w.clone()))
}

#[derive(Deserialize)]
struct VcsReq {
    #[serde(default)]
    init_if_missing: bool,
    #[serde(default)]
    remote_auth: RemoteAuth,
    /// Only sent when the user typed one; empty leaves a stored token alone.
    #[serde(default)]
    github_token: Option<String>,
}

async fn wizard_vcs(State(s): State<Shared>, Json(req): Json<VcsReq>) -> ApiResult<Json<Wizard>> {
    let engine = s.engine()?;
    let ws = engine.workspace();
    if req.init_if_missing && !ws.repo.is_repo().await {
        ws.repo.init().await?;
    }
    if ws.repo.is_repo().await {
        ws.repo.ensure_state_dir_excluded().await?;
    }
    if let Some(t) = req.github_token.as_deref().filter(|t| !t.trim().is_empty()) {
        ws.secrets.set(GITHUB, t)?;
    }
    if req.remote_auth == RemoteAuth::Pat && !ws.secrets.has(GITHUB) {
        return Err(Error::InvalidRequest("enter a GitHub token to use PAT authentication".into()).into());
    }
    ws.update_config(|c| c.remote_auth = req.remote_auth)?;
    let status = vcs_status(ws).await;
    apply(&s, Event::VcsConfigured { status })
}

#[derive(Deserialize)]
struct FinishReq {
    routing: Routing,
    ollama_url: String,
    #[serde(default)]
    llamacpp_url: Option<String>,
    monthly_cap_usd: f64,
    enabled_skills: Vec<String>,
    /// Provider → API key; empty values are ignored.
    #[serde(default)]
    keys: BTreeMap<ProviderId, String>,
}

/// Every model the routing names must be usable: cloud providers need a
/// key, llama.cpp needs its URL.
fn validate_routing(ws: &Workspace, cfg: &Config) -> Result<()> {
    let r = &cfg.routing;
    for (role, m) in
        [("planner", &r.planner), ("actor", &r.actor), ("critic", &r.critic), ("escalation", &r.escalation)]
    {
        let Some(m) = m else { continue };
        if m.model.trim().is_empty() {
            return Err(Error::InvalidRequest(format!("{role}: model name is empty")));
        }
        let usable = match m.provider {
            ProviderId::Ollama => true,
            ProviderId::LlamaCpp => cfg.llamacpp_url.as_deref().is_some_and(|u| !u.trim().is_empty()),
            ProviderId::Acp => cfg.external_agents.iter().any(|a| a.id == m.model),
            ProviderId::Compat => {
                let ep = m.model.split('/').next().unwrap_or("");
                cfg.endpoints.iter().any(|e| e.id == ep)
            }
            p => ws.secrets.provider_key(p).is_some(),
        };
        if !usable {
            return Err(Error::InvalidRequest(format!(
                "{role} uses {} but it is not configured ({})",
                m.provider.as_str(),
                if m.provider.is_local() { "set its URL" } else { "add an API key" }
            )));
        }
    }
    Ok(())
}

async fn wizard_finish(State(s): State<Shared>, Json(req): Json<FinishReq>) -> ApiResult<Json<Wizard>> {
    if s.wizard.lock().unwrap().step != wizard::Step::Skills {
        return Err(Error::Wizard("not at the providers & skills step".into()).into());
    }
    let engine = s.engine()?;
    let ws = engine.workspace();
    for (p, k) in &req.keys {
        if !k.trim().is_empty() {
            ws.set_provider_key(*p, k)?;
        }
    }
    let mut next = ws.config();
    next.routing = req.routing;
    next.ollama_url = req.ollama_url.trim().to_string();
    next.llamacpp_url = req.llamacpp_url.map(|u| u.trim().to_string()).filter(|u| !u.is_empty());
    next.monthly_cap_usd = req.monthly_cap_usd;
    next.onboarded = true;
    if next.routing.actor.is_none() {
        return Err(Error::InvalidRequest("choose an actor model (the one that does most of the work)".into()).into());
    }
    validate_routing(ws, &next)?;
    next.validate()?;
    ws.skills.set_enabled_exact(&req.enabled_skills)?;
    ws.update_config(|c| *c = next)?;
    apply(&s, Event::SkillsConfirmed)
}

// ---- chat-first flow ------------------------------------------------------

#[derive(Deserialize)]
struct OpenWorkspaceReq {
    path: String,
    #[serde(default)]
    git_init: bool,
}

/// One-step replacement for the wizard: bind a directory as the workspace.
/// A non-repository is refused with `needs_git_init` unless `git_init`.
async fn open_workspace(State(s): State<Shared>, Json(req): Json<OpenWorkspaceReq>) -> ApiResult {
    s.models_changed();
    let path = crate::fsutil::canonical(&expand_home(&req.path))
        .map_err(|e| Error::InvalidRequest(format!("{}: {e}", req.path.trim())))?;
    if !path.is_dir() {
        return Err(Error::InvalidRequest(format!("{} is not a directory", path.display())).into());
    }
    let repo = crate::git::GitRepo::new(&path);
    if !repo.is_repo().await {
        if !req.git_init {
            let inside = repo.enclosing_repo().await;
            return Ok(Json(json!({ "needs_git_init": true, "path": path, "enclosing_repo": inside })));
        }
        repo.init().await?;
    }
    let engine = s.bind(&path).await?;
    engine.workspace().update_config(|c| c.onboarded = true)?;
    {
        let mut w = s.wizard.lock().unwrap();
        w.apply(Event::Resume)?;
    }
    Ok(Json(json!({ "ok": true, "path": path })))
}

/// Every model the user can pick right now: live local models plus the
/// known models of cloud providers that have a key.
/// How long a model list is reused. Short enough that a model pulled in a
/// terminal shows up quickly; long enough that opening pickers is instant.
const MODELS_TTL: std::time::Duration = std::time::Duration::from_secs(20);

async fn available_models(State(s): State<Shared>) -> ApiResult {
    let engine = s.engine()?;
    let ws = engine.workspace();
    let cfg = ws.config();
    // Anything the answer depends on: a different workspace, config or set
    // of configured providers makes the cached answer stale.
    let fingerprint = format!(
        "{}|{:?}|{}",
        ws.root.display(),
        ws.gateway.configured(),
        serde_json::to_string(&cfg).unwrap_or_default()
    );
    if let Some((_, _, v)) =
        s.models_cache.lock().unwrap().as_ref().filter(|(t, f, _)| t.elapsed() < MODELS_TTL && *f == fingerprint)
    {
        return Ok(Json(v.clone()));
    }

    // Every provider is asked at the same time: the slowest one sets the
    // wait, not the sum of all of them.
    let ids = ws.gateway.configured();
    let mut set = tokio::task::JoinSet::new();
    for (i, id) in ids.iter().copied().enumerate() {
        let (s, engine, cfg) = (s.clone(), engine.clone(), cfg.clone());
        set.spawn(async move {
            let ws = engine.workspace();
            let mut out = Vec::new();
            let mut notes = Vec::new();
            if id == ProviderId::Compat {
                let models = match ws.gateway.provider(id) {
                    Ok(router) => router.list_models().await.unwrap_or_default(),
                    Err(_) => vec![],
                };
                for e in &cfg.endpoints {
                    let mine: Vec<&String> = models.iter().filter(|m| m.starts_with(&format!("{}/", e.id))).collect();
                    if mine.is_empty() {
                        notes.push(format!("{}: no models listed (is it reachable?)", e.name));
                    }
                    out.extend(mine.into_iter().map(
                        |m| json!({ "provider": id, "model": m, "local": e.local, "efforts": [], "endpoint": e.name }),
                    ));
                }
            } else if id.is_local() {
                match ws.gateway.provider(id) {
                    Ok(provider) => {
                        match tokio::time::timeout(std::time::Duration::from_secs(4), provider.list_models()).await {
                            Ok(Ok(models)) => out.extend(
                                models
                                    .into_iter()
                                    .map(|m| json!({ "provider": id, "model": m, "local": true, "efforts": [] })),
                            ),
                            Ok(Err(e)) => notes.push(e.to_string()),
                            Err(_) => notes.push(format!("{}: timed out", id.as_str())),
                        }
                    }
                    Err(e) => notes.push(e.to_string()),
                }
            } else {
                for m in s.cloud_models(ws, id).await {
                    let efforts = pricing::effort_levels(id, &m);
                    out.push(json!({ "provider": id, "model": m, "local": false, "efforts": efforts }));
                }
            }
            (i, out, notes)
        });
    }
    let mut parts = Vec::new();
    while let Some(r) = set.join_next().await {
        if let Ok(p) = r {
            parts.push(p);
        }
    }
    parts.sort_by_key(|(i, ..)| *i); // same order as before, whoever answered first
    let (mut out, mut notes) = (Vec::new(), Vec::new());
    for (_, o, n) in parts {
        out.extend(o);
        notes.extend(n);
    }
    // External agents: offered when their launcher is on PATH.
    let mut agents = Vec::new();
    for a in &cfg.external_agents {
        let available = crate::agents::resolve_command(&a.command).is_some();
        agents.push(json!({ "provider": "acp", "model": a.id, "name": a.name, "available": available, "kind": a.kind(),
            "command": format!("{} {}", a.command, a.args.join(" ")) }));
    }
    let body = json!({ "models": out, "agents": agents, "notes": notes, "default": cfg.routing.actor });
    *s.models_cache.lock().unwrap() = Some((std::time::Instant::now(), fingerprint, body.clone()));
    Ok(Json(body))
}

/// Tests a cloud API key, then stores it and lists the models it unlocks.
async fn add_cloud_key(s: &Shared, provider: ProviderId, api_key: String) -> ApiResult {
    let engine = s.engine()?;
    let ws = engine.workspace();
    let key = api_key.trim().to_string();
    if key.is_empty() || provider.is_local() || matches!(provider, ProviderId::Acp | ProviderId::Compat) {
        return Err(Error::InvalidRequest("paste an API key for Claude, DeepSeek or Gemini".into()).into());
    }
    let probe: Box<dyn crate::providers::Provider> = match provider {
        ProviderId::Claude => Box::new(crate::providers::ClaudeProvider::new(key.clone())),
        ProviderId::DeepSeek => Box::new(OpenAiCompatProvider::deepseek(key.clone())),
        _ => Box::new(OpenAiCompatProvider::gemini(key.clone())),
    };
    tokio::time::timeout(std::time::Duration::from_secs(15), probe.list_models())
        .await
        .map_err(|_| Error::Provider("no answer within 15 s".into()))?
        .map_err(|e| Error::Provider(format!("the key was rejected: {e}")))?;
    ws.set_provider_key(provider, &key)?;
    s.cloud_models.lock().await.remove(&provider);
    s.models_changed();
    Ok(Json(json!({ "added": provider.as_str(), "models": s.cloud_models(ws, provider).await })))
}

/// One-step "add a model": tests the connection first and only saves what
/// works, then returns the models it made available.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum AddModelReq {
    /// Claude / DeepSeek / Gemini with an API key.
    Cloud {
        provider: ProviderId,
        api_key: String,
    },
    /// Any OpenAI-compatible endpoint.
    Endpoint {
        name: String,
        base_url: String,
        #[serde(default)]
        api_key: String,
        #[serde(default)]
        local: bool,
        #[serde(default)]
        price_in: Option<f64>,
        #[serde(default)]
        price_out: Option<f64>,
    },
    Ollama {
        url: String,
    },
    Llamacpp {
        url: String,
    },
    /// An external agent started by `command` (a full command line):
    /// over ACP, or as a headless CLI of the given dialect.
    Acp {
        name: String,
        command: String,
        #[serde(default)]
        protocol: crate::agents::Protocol,
        #[serde(default)]
        dialect: crate::agents::Dialect,
    },
}

fn slug(name: &str, taken: &[String]) -> String {
    let base: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let base = if base.is_empty() { "endpoint".to_string() } else { base };
    let mut id = base.clone();
    let mut n = 2;
    while taken.contains(&id) {
        id = format!("{base}-{n}");
        n += 1;
    }
    id
}

async fn add_model(State(s): State<Shared>, Json(req): Json<AddModelReq>) -> ApiResult {
    s.models_changed();
    let engine = s.engine()?;
    let ws = engine.workspace();
    let timeout = std::time::Duration::from_secs(15);
    let listed = |r: Result<Result<Vec<String>>, tokio::time::error::Elapsed>| -> Result<Vec<String>> {
        r.map_err(|_| Error::Provider("no answer within 15 s".into()))?
    };
    let (label, models) = match req {
        AddModelReq::Cloud { provider, api_key } => return add_cloud_key(&s, provider, api_key).await,
        AddModelReq::Endpoint { name, base_url, api_key, local, price_in, price_out } => {
            let base = base_url.trim().trim_end_matches('/').to_string();
            if !(base.starts_with("http://") || base.starts_with("https://")) {
                return Err(Error::InvalidRequest("the base URL must start with http:// or https://".into()).into());
            }
            let key = Some(api_key.trim().to_string()).filter(|k| !k.is_empty());
            let found = listed(
                tokio::time::timeout(timeout, OpenAiCompatProvider::endpoint(&base, key.clone(), local).list_models())
                    .await,
            )?;
            let taken: Vec<String> = ws.config().endpoints.iter().map(|e| e.id.clone()).collect();
            let id = slug(&name, &taken);
            if let Some(k) = &key {
                ws.secrets.set(&format!("endpoint:{id}"), k)?;
            }
            let ep = crate::config::Endpoint {
                id: id.clone(),
                name: if name.trim().is_empty() { id.clone() } else { name.trim().to_string() },
                base_url: base,
                local,
                price_in,
                price_out,
            };
            ws.update_config(|c| c.endpoints.push(ep))?;
            (id.clone(), found.into_iter().map(|m| format!("{id}/{m}")).collect())
        }
        AddModelReq::Ollama { url } => {
            let url = url.trim().trim_end_matches('/').to_string();
            let found =
                listed(tokio::time::timeout(timeout, OllamaProvider::new(url.clone(), 2048).list_models()).await)?;
            ws.update_config(|c| c.ollama_url = url)?;
            ("ollama".into(), found)
        }
        AddModelReq::Llamacpp { url } => {
            let url = url.trim().trim_end_matches('/').to_string();
            let found =
                listed(tokio::time::timeout(timeout, OpenAiCompatProvider::llamacpp(url.clone()).list_models()).await)?;
            ws.update_config(|c| c.llamacpp_url = Some(url))?;
            ("llamacpp".into(), found)
        }
        AddModelReq::Acp { name, command, protocol, dialect } => {
            let mut parts = command.split_whitespace();
            let program =
                parts.next().ok_or_else(|| Error::InvalidRequest("enter the command that starts the agent".into()))?;
            let taken: Vec<String> = ws.config().external_agents.iter().map(|a| a.id.clone()).collect();
            let args: Vec<&str> = parts.collect();
            let mut agent = crate::agents::ExternalAgent::new(&slug(&name, &taken), name.trim(), program, &args);
            if protocol == crate::agents::Protocol::Cli {
                agent = agent.cli(dialect);
            }
            crate::agents::probe(&agent, &ws.root).await?;
            let id = agent.id.clone();
            ws.update_config(|c| c.external_agents.push(agent))?;
            (id.clone(), vec![id])
        }
    };
    Ok(Json(json!({ "added": label, "models": models })))
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RemoveModelReq {
    Cloud { provider: ProviderId },
    Endpoint { id: String },
    Acp { id: String },
}

async fn remove_model(State(s): State<Shared>, Json(req): Json<RemoveModelReq>) -> ApiResult {
    s.models_changed();
    let engine = s.engine()?;
    let ws = engine.workspace();
    match req {
        RemoveModelReq::Cloud { provider } => ws.set_provider_key(provider, "")?,
        RemoveModelReq::Endpoint { id } => {
            ws.secrets.set(&format!("endpoint:{id}"), "")?;
            ws.update_config(|c| c.endpoints.retain(|e| e.id != id))?;
        }
        RemoveModelReq::Acp { id } => {
            ws.update_config(|c| c.external_agents.retain(|a| a.id != id))?;
        }
    }
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct PullReq {
    model: String,
}

/// Streams `ollama pull` progress (Ollama's own NDJSON lines) to the UI.
async fn ollama_pull(State(s): State<Shared>, Json(req): Json<PullReq>) -> Result<Response, ApiError> {
    let engine = s.engine()?;
    let url = engine.workspace().config().ollama_url;
    let model = req.model.trim().to_string();
    if model.is_empty() || model.contains(char::is_whitespace) {
        return Err(Error::InvalidRequest("enter a model name like qwen2.5-coder:7b".into()).into());
    }
    let resp = reqwest::Client::new()
        .post(format!("{}/api/pull", url.trim_end_matches('/')))
        .json(&json!({ "model": model, "stream": true }))
        .send()
        .await
        .map_err(|e| Error::Provider(format!("cannot reach Ollama at {url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(Error::Provider(format!("ollama pull: HTTP {}", resp.status())).into());
    }
    // The model list changes once the download is over (or aborted).
    struct Refresh(Shared);
    impl Drop for Refresh {
        fn drop(&mut self) {
            self.0.models_changed();
        }
    }
    let refresh = Refresh(s.clone());
    let stream = tokio_stream::StreamExt::map(resp.bytes_stream(), move |chunk| {
        let _keep = &refresh;
        chunk
    });
    let body = axum::body::Body::from_stream(stream);
    Ok(([(header::CONTENT_TYPE, "application/x-ndjson")], body).into_response())
}

#[derive(Deserialize)]
struct ProbeReq {
    id: String,
    #[serde(default)]
    refresh: bool,
}

/// What an external agent lets you choose (model, effort, …), as it
/// advertises it over ACP.
async fn agent_options(State(s): State<Shared>, Json(req): Json<ProbeReq>) -> ApiResult {
    if !req.refresh
        && let Some(v) = s.agent_options.lock().await.get(&req.id)
    {
        return Ok(Json(v.clone()));
    }
    let engine = s.engine()?;
    let ws = engine.workspace();
    let agent = ws
        .config()
        .external_agents
        .into_iter()
        .find(|a| a.id == req.id)
        .ok_or_else(|| Error::NotFound(format!("external agent {}", req.id)))?;
    // Discovery starts the agent: no lock is held meanwhile, so asking
    // two different agents at once does not queue one behind the other.
    let options = crate::agents::discover(&agent, &ws.root).await?;
    let v = json!({ "options": options });
    s.agent_options.lock().await.insert(req.id, v.clone());
    Ok(Json(v))
}

/// Starts an external agent and runs the ACP handshake only.
async fn probe_agent(State(s): State<Shared>, Json(req): Json<ProbeReq>) -> ApiResult {
    let engine = s.engine()?;
    let ws = engine.workspace();
    let agent = ws
        .config()
        .external_agents
        .into_iter()
        .find(|a| a.id == req.id)
        .ok_or_else(|| Error::NotFound(format!("external agent {}", req.id)))?;
    let info = crate::agents::probe(&agent, &ws.root).await?;
    Ok(Json(json!({ "ok": true, "info": info })))
}

#[derive(Deserialize)]
struct ProposeReq {
    goal: String,
    model: crate::config::ModelRef,
}

async fn propose_team(State(s): State<Shared>, Json(req): Json<ProposeReq>) -> ApiResult {
    let graph = s.engine()?.propose_flow(&req.goal, &req.model).await?;
    Ok(Json(json!({ "graph": graph })))
}

async fn step_diff(State(s): State<Shared>, UrlPath((id, seq)): UrlPath<(String, i64)>) -> ApiResult {
    let (stat, patch) = s.engine()?.step_diff(&id, seq).await?;
    Ok(Json(json!({ "stat": stat, "patch": crate::fsutil::clip(&patch, 200_000) })))
}

async fn suggest_lessons(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    Ok(Json(json!({ "lessons": s.engine()?.suggest_lessons(&id).await? })))
}

#[derive(Deserialize)]
struct LessonsReq {
    lessons: Vec<String>,
}

async fn save_lessons(State(s): State<Shared>, Json(req): Json<LessonsReq>) -> ApiResult {
    Ok(Json(json!({ "total": s.engine()?.save_lessons(&req.lessons)? })))
}

#[derive(Deserialize)]
struct PreflightReq {
    #[serde(default)]
    verify_command: Option<String>,
    #[serde(flatten)]
    spec: RunSpec,
}

async fn preflight(State(s): State<Shared>, Json(req): Json<PreflightReq>) -> ApiResult {
    Ok(Json(s.engine()?.preflight(&req.spec, req.verify_command.as_deref()).await?))
}

async fn kill_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    Ok(Json(json!({ "run": s.engine()?.kill(&id).await? })))
}

#[derive(Deserialize)]
struct EditFlowReq {
    graph: crate::storage::FlowGraph,
}

async fn edit_flow(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<EditFlowReq>) -> ApiResult {
    Ok(Json(json!({ "run": s.engine()?.edit_flow(&id, req.graph).await? })))
}

#[derive(Deserialize)]
struct CriteriaReq {
    goal: String,
    #[serde(default)]
    criteria: String,
    model: crate::config::ModelRef,
}

async fn detail_criteria(State(s): State<Shared>, Json(req): Json<CriteriaReq>) -> ApiResult {
    Ok(Json(json!({ "criteria": s.engine()?.detail_criteria(&req.goal, &req.criteria, &req.model).await? })))
}

#[derive(Deserialize)]
struct ReorchestrateReq {
    goal: String,
    /// Start from the cancelled run's latest state instead of HEAD.
    #[serde(default)]
    continue_current: bool,
}

async fn reorchestrate(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<ReorchestrateReq>) -> ApiResult {
    use crate::graph::flow::Reorchestration;
    Ok(Json(match s.engine()?.reorchestrate(&id, &req.goal, req.continue_current).await? {
        Reorchestration::Started(run) => json!({ "run": run }),
        Reorchestration::Draft { spec, base, verify } => {
            json!({ "draft": { "goal": req.goal.trim(), "spec": spec, "base_commit": base, "verify_command": verify } })
        }
    }))
}

// ---- settings ---------------------------------------------------------------------

async fn put_config(State(s): State<Shared>, Json(mut cfg): Json<Config>) -> ApiResult {
    s.models_changed();
    let engine = s.engine()?;
    let ws = engine.workspace();
    cfg.onboarded = ws.config().onboarded;
    validate_routing(ws, &cfg)?;
    let saved = ws.update_config(|c| *c = cfg)?;
    Ok(Json(json!({ "config": saved, "providers": providers_json(ws) })))
}

#[derive(Deserialize)]
struct ProviderKeyReq {
    provider: ProviderId,
    /// Empty removes the key.
    api_key: String,
}

async fn set_provider_key(State(s): State<Shared>, Json(req): Json<ProviderKeyReq>) -> ApiResult {
    s.models_changed();
    let engine = s.engine()?;
    let ws = engine.workspace();
    ws.set_provider_key(req.provider, &req.api_key)?;
    Ok(Json(json!({ "providers": providers_json(ws) })))
}

#[derive(Deserialize)]
struct ModelsQuery {
    /// For local providers: probe this URL instead of the saved one (the
    /// setup wizard checks an endpoint before it is saved).
    url: Option<String>,
}

/// Lists a provider's models live. For cloud providers this is also the
/// "test key" button.
async fn provider_models(
    State(s): State<Shared>,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<ModelsQuery>,
) -> ApiResult {
    let id = ProviderId::parse(&id)?;
    let url = q.url.map(|u| u.trim().to_string()).filter(|u| !u.is_empty());
    let models = match (id, url) {
        (ProviderId::Ollama, Some(u)) => OllamaProvider::new(u, 2048).list_models().await?,
        (ProviderId::LlamaCpp, Some(u)) => OpenAiCompatProvider::llamacpp(u).list_models().await?,
        _ => s.engine()?.workspace().gateway.provider(id)?.list_models().await?,
    };
    Ok(Json(json!({ "models": models })))
}

// ---- skills -----------------------------------------------------------------------

async fn list_skills(State(s): State<Shared>) -> ApiResult {
    let engine = s.engine()?;
    let ws = engine.workspace();
    Ok(Json(json!({ "skills": ws.skills.list(), "errors": skill_errors(ws), "dir": ws.skills.dir() })))
}

async fn reload_skills(State(s): State<Shared>) -> ApiResult {
    let engine = s.engine()?;
    let ws = engine.workspace();
    ws.skills.reload()?;
    Ok(Json(json!({ "skills": ws.skills.list(), "errors": skill_errors(ws) })))
}

#[derive(Deserialize)]
struct SkillToggleReq {
    name: String,
    on: bool,
}

async fn toggle_skill(State(s): State<Shared>, Json(req): Json<SkillToggleReq>) -> ApiResult {
    let engine = s.engine()?;
    engine.workspace().skills.set_enabled(&req.name, req.on)?;
    Ok(Json(json!({ "skills": engine.workspace().skills.list() })))
}

// ---- runs -------------------------------------------------------------------------

async fn list_runs(State(s): State<Shared>) -> ApiResult {
    let engine = s.engine()?;
    let runs = engine.list_runs().await?;
    Ok(Json(json!({ "runs": runs })))
}

#[derive(Deserialize)]
struct CreateRunReq {
    goal: String,
    #[serde(default)]
    verify_command: Option<String>,
    /// Start from this commit instead of HEAD (re-orchestration).
    #[serde(default)]
    base_commit: Option<String>,
    /// mode / model / graph; defaults to a classic run.
    #[serde(flatten)]
    spec: RunSpec,
}

async fn create_run(State(s): State<Shared>, Json(req): Json<CreateRunReq>) -> ApiResult {
    let engine = s.engine()?;
    let run = engine.create_run_at(&req.goal, req.verify_command.as_deref(), req.spec, req.base_commit.as_deref()).await?;
    Ok(Json(json!({ "run": run })))
}

#[derive(Deserialize)]
struct RunDetailQuery {
    /// Only steps after this one: a viewer that already has the earlier
    /// steps fetches just what is new instead of the whole history.
    after: Option<i64>,
}

async fn get_run(
    State(s): State<Shared>,
    UrlPath(id): UrlPath<String>,
    axum::extract::Query(q): axum::extract::Query<RunDetailQuery>,
) -> ApiResult {
    let engine = s.engine()?;
    let (run, steps) = match q.after {
        Some(seq) => engine.run_detail_after(&id, seq).await?,
        None => engine.run_detail(&id).await?,
    };
    // `step_count` lets an incremental viewer notice a rewind (fewer steps
    // than it holds) and refetch everything.
    let head = if run.spec.graph.is_some() { engine.workspace().store.head_step(&id).await? } else { None };
    let state = head.as_ref().map(|h| h.state.clone()).unwrap_or_default();
    let current = crate::graph::flow::current_graph(&run, &state);
    let vision = match &current {
        Some(g) => engine.vision_nodes(&run, g).await,
        None => vec![],
    };
    // Every step, read once (the incremental answer only has the new ones).
    let all = match (&current, q.after) {
        (None, _) => Vec::new(),
        (Some(_), None) => steps.clone(),
        (Some(_), Some(_)) => engine.workspace().store.steps(&id).await?,
    };
    let node_stats = match &current {
        Some(g) => Value::Object(engine.node_stats(&run, g, &all).await),
        None => Value::Null,
    };
    let lines = crate::git::GitRepo::new(&run.worktree)
        .numstat_from(&run.base_commit)
        .await
        .ok()
        .map(|(a, r)| json!({ "added": a, "removed": r }));
    // Where the flow stands: finished nodes and who acts next.
    let flow = match (&current, &head) {
        (Some(g), Some(h)) => {
            let next = crate::graph::flow::next_actor(g, &all, &h.state);
            // A Dispatcher at work: its branches' progress, and the agent
            // each running branch is at.
            let mut done = h.state.flow.done.clone();
            let mut working: Vec<String> = next.iter().map(|(n, _)| n.clone()).collect();
            let branches: Vec<Value> = h.state.flow.parallel.iter().flat_map(|p| &p.branches).map(|b| {
                let bdone = b.state.as_ref().map(|x| x.flow.done.clone()).unwrap_or_default();
                let now = b.nodes.iter().find(|n| !bdone.contains(n)).cloned();
                for n in b.nodes.iter().filter(|n| bdone.contains(n)) {
                    if !done.contains(n) {
                        done.push(n.clone());
                    }
                }
                if b.status == crate::storage::BranchStatus::Running && let Some(n) = &now {
                    working.push(n.clone());
                }
                json!({ "head": b.head, "nodes": b.nodes, "status": b.status, "now": now, "report": b.report })
            }).collect();
            json!({ "done": done, "goal_rounds": h.state.flow.goal_rounds,
                    "next": next.as_ref().map(|(n, _)| n), "reviewing": next.and_then(|(_, r)| r),
                    "working": working, "branches": branches })
        }
        (Some(g), None) => json!({ "done": [], "goal_rounds": 0, "next": g.lead().map(|l| l.id.clone()), "reviewing": null }),
        _ => Value::Null,
    };
    Ok(Json(json!({
        "run": run,
        "steps": steps,
        "active": engine.is_active(&id),
        "partial": q.after.is_some(),
        "approval": engine.pending_approval(&id),
        "flow": flow,
        "vision": vision,
        "node_stats": node_stats,
        "lines": lines,
        // The flow as it stands now, with the Lead's runtime changes.
        "graph": current.map(|mut g| {
            crate::storage::flowgraph::auto_layout(&mut g);
            g
        }),
    })))
}

async fn delete_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    s.engine()?.delete(&id).await?;
    Ok(Json(json!({ "deleted": id })))
}

async fn pause_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    Ok(Json(json!({ "run": s.engine()?.pause(&id).await? })))
}

async fn resume_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    Ok(Json(json!({ "run": s.engine()?.start(&id).await? })))
}

#[derive(Deserialize)]
struct InjectReq {
    text: String,
    /// Orchestra runs: the agent the message is for (default: the lead).
    #[serde(default)]
    agent: Option<String>,
}

async fn inject_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<InjectReq>) -> ApiResult {
    Ok(Json(json!({ "step": s.engine()?.inject(&id, &req.text, req.agent.as_deref()).await? })))
}

#[derive(Deserialize)]
struct RewindReq {
    /// Keep steps up to and including this one; -1 rewinds to the start.
    seq: i64,
}

async fn rewind_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<RewindReq>) -> ApiResult {
    Ok(Json(json!({ "run": s.engine()?.rewind(&id, req.seq).await? })))
}

async fn diff_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    let (stat, patch) = s.engine()?.diff(&id).await?;
    Ok(Json(json!({ "stat": stat, "patch": crate::fsutil::clip(&patch, 400_000) })))
}

async fn push_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    Ok(Json(json!({ "output": s.engine()?.push(&id).await? })))
}

async fn merge_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    let output = s.engine()?.merge(&id).await?;
    Ok(Json(json!({ "output": output })))
}

#[derive(Deserialize)]
struct DecideReq {
    id: String,
    allow: bool,
    #[serde(default)]
    always: bool,
    #[serde(default)]
    note: String,
}

async fn decide_approval(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<DecideReq>) -> ApiResult {
    s.engine()?.decide(&id, &req.id, req.allow, req.always, &req.note)?;
    Ok(Json(json!({ "ok": true })))
}

#[derive(Deserialize)]
struct AutonomyReq {
    autonomy: crate::storage::Autonomy,
}

#[derive(Deserialize)]
struct QuestionsReq {
    /// "user", or "lead" (also accepted as "commander", the tree's name for it).
    route: String,
}

async fn set_questions(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<QuestionsReq>) -> ApiResult {
    use crate::storage::flowgraph::QuestionRoute;
    let route = match req.route.as_str() {
        "user" => QuestionRoute::User,
        "lead" | "commander" => QuestionRoute::Lead,
        other => return Err(Error::InvalidRequest(format!("unknown question route {other:?}")).into()),
    };
    Ok(Json(json!({ "run": s.engine()?.set_questions(&id, route).await? })))
}

async fn set_autonomy(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<AutonomyReq>) -> ApiResult {
    Ok(Json(json!({ "run": s.engine()?.set_autonomy(&id, req.autonomy).await? })))
}

async fn browser_status() -> ApiResult {
    let exe = crate::graph::browser::find_browser();
    Ok(Json(json!({ "headless": exe.as_ref().map(|p| p.display().to_string()) })))
}

/// A run's worktree over http, for the dashboard's browser pane. Pages are
/// sandboxed (see `browser::serve_file`).
async fn serve_preview(s: &Shared, id: &str, path: &str) -> Response {
    let run = match s.engine() {
        Ok(e) => e.workspace().store.get_run(id).await,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    match run {
        Ok(run) => crate::graph::browser::serve_file(Path::new(&run.worktree), &format!("/{path}"), true).await,
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn preview_index(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> Response {
    serve_preview(&s, &id, "").await
}

async fn preview_index_t(State(s): State<Shared>, UrlPath((_token, id)): UrlPath<(String, String)>) -> Response {
    serve_preview(&s, &id, "").await
}

async fn preview_file(State(s): State<Shared>, UrlPath((id, path)): UrlPath<(String, String)>) -> Response {
    serve_preview(&s, &id, &path).await
}

async fn preview_file_t(
    State(s): State<Shared>,
    UrlPath((_token, id, path)): UrlPath<(String, String, String)>,
) -> Response {
    serve_preview(&s, &id, &path).await
}

// Relative URLs in a page must resolve under the run's prefix.
async fn preview_root(UrlPath(id): UrlPath<String>) -> Response {
    axum::response::Redirect::permanent(&format!("/preview/{id}/")).into_response()
}

async fn preview_root_t(UrlPath((token, id)): UrlPath<(String, String)>) -> Response {
    axum::response::Redirect::permanent(&format!("/pvt/{token}/{id}/")).into_response()
}

#[derive(Deserialize)]
struct AnswerReq {
    /// The question's step.
    seq: i64,
    text: String,
}

async fn answer_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<AnswerReq>) -> ApiResult {
    Ok(Json(json!({ "run": s.engine()?.answer(&id, req.seq, &req.text).await? })))
}

/// `GET /api/access`: what Settings → Remote access shows.
async fn get_access(State(s): State<Shared>) -> ApiResult {
    Ok(Json(describe_access(&s)?))
}

fn describe_access(s: &AppState) -> Result<Value> {
    let require = s.require_token.load(std::sync::atomic::Ordering::SeqCst);
    let base = s.remote_base.lock().unwrap().clone();
    let link = match &base {
        Some(b) => Some(format!("{b}/?token={}", s.link_token()?)),
        None => None,
    };
    Ok(json!({ "remote": base.is_some(), "require_token": require, "url": base, "link": link }))
}

#[derive(Deserialize)]
struct AccessReq {
    require_token: bool,
}

/// `PUT /api/access` (this machine only): turns the token check on or off,
/// saved across restarts.
async fn put_access(State(s): State<Shared>, Json(req): Json<AccessReq>) -> ApiResult {
    let body = json!({ "require_token": req.require_token }).to_string();
    crate::fsutil::write_atomic(&s.access_file, body.as_bytes(), true)?;
    s.require_token.store(req.require_token, std::sync::atomic::Ordering::SeqCst);
    Ok(Json(describe_access(&s)?))
}

async fn list_tokens(State(s): State<Shared>) -> ApiResult {
    let tokens: Vec<Value> = s
        .tokens
        .list()
        .into_iter()
        .map(|t| {
            json!({
                "id": t.id,
                "name": t.name,
                "label": t.name,
                "token": t.token,
                "created_at": t.created_at,
            })
        })
        .collect();
    Ok(Json(json!({ "tokens": tokens })))
}

#[derive(Deserialize, Default)]
struct CreateTokenReq {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    label: Option<String>,
}

async fn create_token(State(s): State<Shared>, req: Option<Json<CreateTokenReq>>) -> ApiResult {
    let req = req.map(|Json(r)| r).unwrap_or_default();
    let name_str = req.name.or(req.label);
    let name = name_str
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("Remote Access");
    let token = s.tokens.create(name)?;
    Ok(Json(json!({
        "id": token.id,
        "name": token.name,
        "label": token.name,
        "token": token.token,
        "created_at": token.created_at,
    })))
}

async fn delete_token(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    let removed = s.tokens.delete(&id)?;
    Ok(Json(json!({ "ok": removed, "deleted": removed })))
}

/// Live feed of every engine event. If a slow client falls behind, it gets
/// a `resync` event and should refetch.
async fn events(
    State(s): State<Shared>,
) -> ApiResult<Sse<impl Stream<Item = std::result::Result<SseEvent, Infallible>>>> {
    let engine = s.engine()?;
    let stream = tokio_stream::wrappers::BroadcastStream::new(engine.subscribe()).map(|msg| {
        let data = match msg {
            Ok(ev) => serde_json::to_string(&ev).unwrap_or_else(|_| r#"{"kind":"resync"}"#.into()),
            Err(_) => r#"{"kind":"resync"}"#.into(),
        };
        Ok(SseEvent::default().data(data))
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn spa_index() -> impl IntoResponse {
    ([(header::CACHE_CONTROL, "no-store")], Html(INDEX_HTML))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use tower::ServiceExt;

    #[test]
    fn slugs_are_unique_and_url_safe() {
        assert_eq!(slug("Open Router!", &[]), "open-router");
        assert_eq!(slug("Open Router", &["open-router".into()]), "open-router-2");
        assert_eq!(slug("  ", &[]), "endpoint");
    }

    #[test]
    fn host_check_accepts_only_loopback_names() {
        for h in ["localhost:7878", "127.0.0.1:7878", "[::1]:7878", "LOCALHOST", "127.0.0.1"] {
            assert!(is_local_host(h), "{h}");
        }
        for h in ["evil.com", "evil.com:7878", "127.0.0.1.evil.com", "localhost.evil.com:80", "[::2]:1"] {
            assert!(!is_local_host(h), "{h}");
        }
    }

    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        host: &str,
        origin: Option<&str>,
        body: Value,
    ) -> (u16, Value) {
        let mut req = Request::builder().method(method).uri(uri).header("host", host);
        if let Some(o) = origin {
            req = req.header("origin", o);
        }
        let req = req.header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status().as_u16();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    async fn wizard_binds_the_chosen_directory_and_rebinds_on_restart() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let state = AppState::new(Some(root.clone())).await.unwrap();
        assert!(!root.join(".orchopork").exists(), "nothing is written before the user picks a directory");
        let app = router(state.clone());
        let h = "127.0.0.1:7878";

        assert_eq!(call(&app, "POST", "/api/wizard/initialize", h, None, json!({})).await.0, 200);
        let (code, w) = call(&app, "POST", "/api/wizard/workspace", h, None, json!({ "path": root })).await;
        assert_eq!(code, 200, "{w}");
        assert_eq!(w["step"], "vcs");
        assert!(root.join(".orchopork").exists() && !root.join(".git").exists());

        let (code, _) =
            call(&app, "POST", "/api/wizard/vcs", h, None, json!({ "init_if_missing": true, "remote_auth": "none" }))
                .await;
        assert_eq!(code, 200);
        assert!(root.join(".git").exists());

        let finish = json!({
            "routing": { "actor": { "provider": "claude", "model": "claude-haiku-4-5" } },
            "ollama_url": "http://127.0.0.1:11434",
            "monthly_cap_usd": 5.0,
            "enabled_skills": ["anti-sycophancy-terse"],
        });
        let (code, e) = call(&app, "POST", "/api/wizard/finish", h, None, finish.clone()).await;
        assert_eq!(code, 400, "claude without a key must be refused: {e}");
        let mut with_key = finish;
        with_key["keys"] = json!({ "claude": "sk-test" });
        let (code, w) = call(&app, "POST", "/api/wizard/finish", h, None, with_key).await;
        assert_eq!((code, w["step"].as_str()), (200, Some("dashboard")));

        let (_, st) = call(&app, "GET", "/api/state", h, None, Value::Null).await;
        assert_eq!(st["workspace"]["config"]["monthly_cap_usd"], 5.0);
        assert!(!st.to_string().contains("sk-test"), "secrets never reach the browser");

        // A restart in the same directory skips the wizard.
        drop(app);
        drop(state);
        let state = AppState::new(Some(root.clone())).await.unwrap();
        assert_eq!(state.wizard.lock().unwrap().step, wizard::Step::Dashboard);
    }

    #[tokio::test]
    async fn foreign_hosts_and_origins_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let state = AppState::new(Some(dir.path().to_path_buf())).await.unwrap();
        let app = router(state.clone());
        assert_eq!(call(&app, "GET", "/api/state", "evil.example:7878", None, Value::Null).await.0, 403);
        let (code, _) =
            call(&app, "POST", "/api/wizard/initialize", "127.0.0.1:7878", Some("http://evil.example"), json!({}))
                .await;
        assert_eq!(code, 403);
        let (code, _) =
            call(&app, "POST", "/api/wizard/initialize", "127.0.0.1:7878", Some("http://localhost:7878"), json!({}))
                .await;
        assert_eq!(code, 200);

        // Previews of run files are never public.
        let r = app
            .clone()
            .oneshot(Request::builder().uri("/preview/x/index.html").header("host", "remote.example:7878").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::FORBIDDEN);
        let r = app
            .clone()
            .oneshot(Request::builder().uri("/pvt/op_wrong/x/index.html").header("host", "remote.example:7878").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::FORBIDDEN);

        // Foreign host WITH valid token is accepted
        let token = state.tokens.create("test-remote").unwrap();
        let req = Request::builder()
            .method("GET")
            .uri("/api/state")
            .header("host", "remote.example:7878")
            .header("authorization", format!("Bearer {}", token.token))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);

        // Token check off (the default): other machines get in by IP, not by
        // a host name, and only this machine may turn the check on.
        let lan = |method: &str, uri: &str, origin: Option<&str>| {
            let mut b = Request::builder().method(method).uri(uri).header("host", "192.168.1.16:7878");
            if let Some(o) = origin {
                b = b.header("origin", o);
            }
            let mut r = b.header("content-type", "application/json").body(Body::from(r#"{"require_token": true}"#)).unwrap();
            r.extensions_mut().insert(axum::extract::ConnectInfo("192.168.1.50:5000".parse::<std::net::SocketAddr>().unwrap()));
            r
        };
        assert_eq!(app.clone().oneshot(lan("GET", "/api/state", None)).await.unwrap().status().as_u16(), 200);
        assert_eq!(
            app.clone().oneshot(lan("PUT", "/api/access", Some("http://192.168.1.16:7878"))).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        assert_eq!(
            app.clone().oneshot(lan("POST", "/api/runs", Some("http://evil.example"))).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        state.require_token.store(true, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(app.clone().oneshot(lan("GET", "/api/state", None)).await.unwrap().status(), StatusCode::FORBIDDEN);
        state.require_token.store(false, std::sync::atomic::Ordering::SeqCst);

        // ...but not from a sandboxed preview page that read it from its URL.
        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/runs/x/kill?token={}", token.token))
            .header("host", "remote.example:7878")
            .header("origin", "null")
            .body(Body::empty())
            .unwrap();
        assert_eq!(app.clone().oneshot(req).await.unwrap().status(), StatusCode::FORBIDDEN);
    }
}
