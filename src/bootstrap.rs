//! The single place a workspace is turned into a running set of
//! components (secrets, skills, provider gateway, checkpointer).
//!
//! `main.rs` (server), the CLI, and any embedder using orchopork as a
//! library all call `open()` — there is no second, slightly-different
//! wiring path. A provider registered here, a skill loaded here, a secret
//! read here is registered/loaded/read identically no matter which of the
//! three ever ends up calling it.

use std::path::Path;
use std::sync::Arc;

use crate::error::Result;
use crate::git::{GitRepo, STATE_DIR};
use crate::graph::Executor;
use crate::providers::{ClaudeProvider, Gateway, OllamaProvider, OpenAiCompatProvider, ProviderId};
use crate::secrets::SecretStore;
use crate::skills::SkillRegistry;
use crate::storage::{Checkpointer, Store};

pub struct Config {
    pub monthly_cap_usd: f64,
    pub ollama_url: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            monthly_cap_usd: std::env::var("ORCHOPORK_MONTHLY_CAP_USD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(40.0),
            ollama_url: std::env::var("ORCHOPORK_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
        }
    }
}

/// Everything a workspace needs to run: secrets, skills, the provider
/// gateway (Ollama always registered; cloud providers registered when a
/// key is already on disk), the checkpointer, and an `Executor` built from
/// the same `Arc`s the standalone skill/provider endpoints and CLI
/// commands read and mutate.
pub struct Workspace {
    pub gateway: Arc<Gateway>,
    pub skills: Arc<SkillRegistry>,
    pub secrets: SecretStore,
    pub checkpointer: Checkpointer,
    pub executor: Executor,
}

/// Opens (creating on first use) `<root>/.orchopork/`: the skill directory,
/// the enabled-skills file, the secrets file, and the SQLite state DB. Also
/// idempotently `git init`s the workspace so checkpointing works from the
/// first step.
pub async fn open(root: &Path, cfg: &Config) -> Result<Workspace> {
    let state_dir = root.join(STATE_DIR);
    let skills = Arc::new(SkillRegistry::open(state_dir.join("skills"), state_dir.join("skills.enabled.yaml"))?);
    let secrets = SecretStore::open(state_dir.join("secrets.yaml"))?;

    let store = Store::open(&state_dir.join("state.db")).await?;
    let repo = GitRepo::new(root);
    repo.init().await?;
    let checkpointer = Checkpointer::new(store.clone(), repo);

    let gateway = Arc::new(Gateway::new(store, cfg.monthly_cap_usd));
    gateway.register(Box::new(OllamaProvider::new(cfg.ollama_url.clone())));
    for id in [ProviderId::Claude, ProviderId::DeepSeek, ProviderId::Gemini] {
        if let Some(key) = secrets.get(id) {
            register_cloud(&gateway, id, key);
        }
    }

    let executor = Executor::new(gateway.clone(), skills.clone(), checkpointer.clone());
    Ok(Workspace { gateway, skills, secrets, checkpointer, executor })
}

/// Shared by `open` (loading a persisted key) and the `/api/providers/keys`
/// handler (a key set live): the same match arm either way.
pub fn register_cloud(gateway: &Gateway, id: ProviderId, api_key: String) {
    match id {
        ProviderId::Claude => gateway.register(Box::new(ClaudeProvider::new(api_key))),
        ProviderId::DeepSeek => gateway.register(Box::new(OpenAiCompatProvider::deepseek(api_key))),
        ProviderId::Gemini => gateway.register(Box::new(OpenAiCompatProvider::gemini(api_key))),
        ProviderId::Ollama => {}
    }
}
