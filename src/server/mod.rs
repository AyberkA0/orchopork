//! Axum server. Binds loopback only; the UI is single-user and local.

pub mod wizard;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::error::Error;
use crate::git::GitRepo;
use crate::providers::{ClaudeProvider, Gateway, OpenAiCompatProvider, ProviderId};
use crate::secrets::SecretStore;
use crate::skills::SkillRegistry;
use wizard::{Event, RemoteAuth, Step, VcsChoice, Wizard};

pub struct AppState {
    pub wizard: Mutex<Wizard>,
    pub skills: SkillRegistry,
    pub gateway: Gateway,
    pub secrets: SecretStore,
}

pub type Shared = Arc<AppState>;

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/api/wizard", get(get_wizard))
        .route("/api/wizard/initialize", post(initialize))
        .route("/api/wizard/workspace", post(set_workspace))
        .route("/api/wizard/vcs", post(configure_vcs))
        .route("/api/wizard/skills", post(confirm_skills))
        .route("/api/wizard/back", post(back))
        .route("/api/skills", get(list_skills))
        .route("/api/skills/reload", post(reload_skills))
        .route("/api/providers/budget", get(get_budget))
        .route("/api/providers/keys", post(set_provider_key))
        .route("/app", get(app_gate))
        // Static SPA (ServeDir / include_dir) is added as the fallback service.
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::new()) // no cross-origin access: same-origin UI only
        .with_state(state)
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
            Error::NotFound(_) => StatusCode::NOT_FOUND,
            Error::Budget(_) => StatusCode::PAYMENT_REQUIRED,
            Error::Provider(_) => StatusCode::BAD_GATEWAY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (code, Json(serde_json::json!({ "error": self.0.to_string() }))).into_response()
    }
}

type ApiResult = Result<Json<Wizard>, ApiError>;

fn apply(s: &Shared, ev: Event) -> ApiResult {
    let mut w = s.wizard.lock().unwrap();
    w.apply(ev)?;
    Ok(Json(w.clone()))
}

async fn get_wizard(State(s): State<Shared>) -> Json<Wizard> {
    Json(s.wizard.lock().unwrap().clone())
}

async fn initialize(State(s): State<Shared>) -> ApiResult {
    apply(&s, Event::Initialize)
}

async fn back(State(s): State<Shared>) -> ApiResult {
    apply(&s, Event::Back)
}

#[derive(Deserialize)]
struct WorkspaceReq {
    path: PathBuf,
}

async fn set_workspace(State(s): State<Shared>, Json(req): Json<WorkspaceReq>) -> ApiResult {
    let path = tokio::fs::canonicalize(&req.path).await.map_err(Error::from)?;
    let report = wizard::check_permissions(&path).await; // IO happens outside the lock
    apply(&s, Event::PermissionsChecked { report })
}

#[derive(Deserialize)]
struct VcsReq {
    /// Initialize a repo in the workspace if it is not one already.
    init_if_missing: bool,
    remote_auth: RemoteAuth,
}

async fn configure_vcs(State(s): State<Shared>, Json(req): Json<VcsReq>) -> ApiResult {
    let root = s
        .wizard
        .lock()
        .unwrap()
        .permissions
        .as_ref()
        .map(|p| p.workspace.clone())
        .ok_or_else(|| Error::Wizard("workspace not selected".into()))?;
    let repo = GitRepo::new(root);
    if req.init_if_missing {
        repo.init().await?;
    }
    let repo_ready = repo.is_repo().await;
    apply(&s, Event::VcsConfigured { choice: VcsChoice { repo_ready, remote_auth: req.remote_auth } })
}

#[derive(Deserialize)]
struct SkillsReq {
    enabled_skills: Vec<String>,
    /// Provider names only; keys go to the secret store, never through wizard state.
    providers: Vec<String>,
}

async fn confirm_skills(State(s): State<Shared>, Json(req): Json<SkillsReq>) -> ApiResult {
    s.skills.set_enabled_exact(&req.enabled_skills)?;
    apply(&s, Event::SkillsConfirmed { enabled_skills: req.enabled_skills, providers: req.providers })
}

async fn list_skills(State(s): State<Shared>) -> Json<serde_json::Value> {
    let errors: Vec<_> =
        s.skills.load_errors().into_iter().map(|(p, e)| serde_json::json!({ "path": p, "error": e })).collect();
    Json(serde_json::json!({ "skills": s.skills.list(), "errors": errors }))
}

async fn reload_skills(State(s): State<Shared>) -> Result<Json<serde_json::Value>, ApiError> {
    let errors = s.skills.reload()?;
    Ok(Json(serde_json::json!({ "errors": errors.len() })))
}

async fn get_budget(State(s): State<Shared>) -> Result<Json<serde_json::Value>, ApiError> {
    let spent = s.gateway.spent_this_month().await.map_err(ApiError)?;
    let remaining = s.gateway.budget_remaining().await.map_err(ApiError)?;
    Ok(Json(serde_json::json!({
        "spent_this_month_usd": spent,
        "remaining_usd": remaining,
        "configured_providers": s.gateway.configured(),
    })))
}

#[derive(Deserialize)]
struct ProviderKeyReq {
    provider: ProviderId,
    api_key: String,
}

/// Persists the key and immediately (re)registers the provider so it is
/// usable without a restart. Keys never pass through `Wizard` state.
async fn set_provider_key(
    State(s): State<Shared>,
    Json(req): Json<ProviderKeyReq>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if req.provider == ProviderId::Ollama {
        return Err(ApiError(Error::Provider("ollama is local and needs no API key".into())));
    }
    s.secrets.set(req.provider, req.api_key.clone()).map_err(ApiError)?;
    match req.provider {
        ProviderId::Claude => s.gateway.register(Box::new(ClaudeProvider::new(req.api_key))),
        ProviderId::DeepSeek => s.gateway.register(Box::new(OpenAiCompatProvider::deepseek(req.api_key))),
        ProviderId::Gemini => s.gateway.register(Box::new(OpenAiCompatProvider::gemini(req.api_key))),
        ProviderId::Ollama => unreachable!(),
    }
    Ok(Json(serde_json::json!({ "configured_providers": s.gateway.configured() })))
}

/// Dashboard is unlocked only after steps 1-4; otherwise bounce to the
/// current step's page.
async fn app_gate(State(s): State<Shared>) -> Response {
    let w = s.wizard.lock().unwrap();
    if w.can_access(Step::Dashboard) {
        (StatusCode::OK, "orchopork dashboard").into_response() // SPA index goes here
    } else {
        Redirect::to(w.step.path()).into_response()
    }
}
