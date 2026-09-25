use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use orchopork::git::STATE_DIR;
use orchopork::server::{self, AppState, wizard::Wizard};
use orchopork::skills::SkillRegistry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=debug".into()))
        .init();

    let workspace: PathBuf = std::env::args().nth(1).map(Into::into).unwrap_or(std::env::current_dir()?);
    let state_dir = workspace.join(STATE_DIR);
    let skills = SkillRegistry::open(state_dir.join("skills"), state_dir.join("skills.enabled.yaml"))?;

    let state = Arc::new(AppState { wizard: Mutex::new(Wizard::default()), skills });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878").await?;
    tracing::info!("orchopork on http://{}", listener.local_addr()?);
    axum::serve(listener, server::router(state)).await?;
    Ok(())
}
