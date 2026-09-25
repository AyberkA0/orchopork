//! Local HTTP API + embedded dashboard. Binds loopback only, and rejects
//! requests whose `Host`/`Origin` is not local (DNS-rebinding protection:
//! this API can run shell commands through the agent, so a web page must
//! never be able to reach it).

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
use axum::routing::{get, post};
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
use wizard::{Event, VcsStatus, Wizard};

/// Single-file embedded SPA; client JS routes off `wizard.step`.
const INDEX_HTML: &str = include_str!("../../ui/index.html");

pub struct AppState {
    engine: RwLock<Option<Engine>>,
    wizard: Mutex<Wizard>,
    /// Pre-filled in the workspace picker (the `--workspace` argument).
    default_root: PathBuf,
}

pub type Shared = Arc<AppState>;

impl AppState {
    /// If `default_root` is an already-onboarded workspace, bind it and go
    /// straight to the dashboard; otherwise start the wizard and touch
    /// nothing on disk until the user picks a directory.
    pub async fn new(default_root: PathBuf) -> Result<Shared> {
        let state = Arc::new(Self {
            engine: RwLock::new(None),
            wizard: Mutex::new(Wizard::default()),
            default_root: default_root.clone(),
        });
        if Workspace::is_initialized(&default_root) {
            let ws = Workspace::open(&default_root).await?;
            if ws.config().onboarded {
                let engine = Engine::new(ws).await?;
                *state.engine.write().unwrap() = Some(engine);
                state.wizard.lock().unwrap().apply(Event::Resume)?;
            }
        }
        Ok(state)
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
        .route("/api/runs/{id}/push", post(push_run))
        .route("/api/events", get(events))
        // Client-side-routed SPA: every other GET gets the same shell.
        .fallback(get(spa_index))
        .layer(middleware::from_fn(local_only))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

fn is_local_host(host: &str) -> bool {
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.rsplit_once(':').map_or(host, |(h, p)| if p.bytes().all(|b| b.is_ascii_digit()) { h } else { host })
    };
    matches!(name.to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1")
}

async fn local_only(req: Request, next: Next) -> Response {
    let headers = req.headers();
    let host_ok = headers.get(header::HOST).and_then(|h| h.to_str().ok()).is_some_and(is_local_host);
    let origin_ok = match headers.get(header::ORIGIN).and_then(|h| h.to_str().ok()) {
        None => true,
        Some(o) => o.split_once("://").is_some_and(|(_, rest)| is_local_host(rest.trim_end_matches('/'))),
    };
    let state_changing = !matches!(*req.method(), Method::GET | Method::HEAD);
    if !host_ok || (state_changing && !origin_ok) {
        return (StatusCode::FORBIDDEN, Json(json!({ "error": "orchopork only accepts local requests" })))
            .into_response();
    }
    next.run(req).await
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
    let Some(engine) = engine else {
        return Ok(Json(json!({ "wizard": wizard, "default_root": s.default_root, "workspace": null })));
    };
    let ws = engine.workspace();
    Ok(Json(json!({
        "wizard": wizard,
        "default_root": s.default_root,
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
    let path = tokio::fs::canonicalize(expand_home(&req.path))
        .await
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
    if r.actor.is_none() {
        return Err(Error::InvalidRequest("choose an actor model (the one that does most of the work)".into()));
    }
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
    validate_routing(ws, &next)?;
    next.validate()?;
    ws.skills.set_enabled_exact(&req.enabled_skills)?;
    ws.update_config(|c| *c = next)?;
    apply(&s, Event::SkillsConfirmed)
}

// ---- settings ---------------------------------------------------------------------

async fn put_config(State(s): State<Shared>, Json(mut cfg): Json<Config>) -> ApiResult {
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
}

async fn create_run(State(s): State<Shared>, Json(req): Json<CreateRunReq>) -> ApiResult {
    let engine = s.engine()?;
    let run = engine.create_run(&req.goal, req.verify_command.as_deref()).await?;
    Ok(Json(json!({ "run": run })))
}

async fn get_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    let engine = s.engine()?;
    let (run, steps) = engine.run_detail(&id).await?;
    Ok(Json(json!({ "run": run, "steps": steps, "active": engine.is_active(&id) })))
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
}

async fn inject_run(State(s): State<Shared>, UrlPath(id): UrlPath<String>, Json(req): Json<InjectReq>) -> ApiResult {
    Ok(Json(json!({ "step": s.engine()?.inject(&id, &req.text).await? })))
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
        let state = AppState::new(root.clone()).await.unwrap();
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
        let state = AppState::new(root.clone()).await.unwrap();
        assert_eq!(state.wizard.lock().unwrap().step, wizard::Step::Dashboard);
    }

    #[tokio::test]
    async fn foreign_hosts_and_origins_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let app = router(AppState::new(dir.path().to_path_buf()).await.unwrap());
        assert_eq!(call(&app, "GET", "/api/state", "evil.example:7878", None, Value::Null).await.0, 403);
        let (code, _) =
            call(&app, "POST", "/api/wizard/initialize", "127.0.0.1:7878", Some("http://evil.example"), json!({}))
                .await;
        assert_eq!(code, 403);
        let (code, _) =
            call(&app, "POST", "/api/wizard/initialize", "127.0.0.1:7878", Some("http://localhost:7878"), json!({}))
                .await;
        assert_eq!(code, 200);
    }
}
