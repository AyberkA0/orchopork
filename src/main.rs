use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use orchopork::git::STATE_DIR;
use orchopork::providers::{ClaudeProvider, Gateway, OllamaProvider, OpenAiCompatProvider, ProviderId};
use orchopork::secrets::SecretStore;
use orchopork::server::{self, AppState, wizard::Wizard};
use orchopork::skills::SkillRegistry;
use orchopork::storage::Store;

const DEFAULT_MONTHLY_CAP_USD: f64 = 40.0;
const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=debug".into()),
        )
        .init();

    let workspace: PathBuf = std::env::args().nth(1).map(Into::into).unwrap_or(std::env::current_dir()?);
    let state_dir = workspace.join(STATE_DIR);
    let skills = SkillRegistry::open(state_dir.join("skills"), state_dir.join("skills.enabled.yaml"))?;
    let secrets = SecretStore::open(state_dir.join("secrets.yaml"))?;

    let store = Store::open(&state_dir.join("state.db")).await?;
    let cap: f64 =
        std::env::var("ORCHOPORK_MONTHLY_CAP_USD").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MONTHLY_CAP_USD);
    let gateway = Gateway::new(store, cap);

    let ollama_url = std::env::var("ORCHOPORK_OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.into());
    gateway.register(Box::new(OllamaProvider::new(ollama_url)));
    if let Some(key) = secrets.get(ProviderId::Claude) {
        gateway.register(Box::new(ClaudeProvider::new(key)));
    }
    if let Some(key) = secrets.get(ProviderId::DeepSeek) {
        gateway.register(Box::new(OpenAiCompatProvider::deepseek(key)));
    }
    if let Some(key) = secrets.get(ProviderId::Gemini) {
        gateway.register(Box::new(OpenAiCompatProvider::gemini(key)));
    }

    let state = Arc::new(AppState { wizard: Mutex::new(Wizard::default()), skills, gateway, secrets });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878").await?;
    tracing::info!("orchopork on http://{}", listener.local_addr()?);
    axum::serve(listener, server::router(state)).await?;
    Ok(())
}
