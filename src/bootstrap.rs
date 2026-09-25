//! The single place a workspace directory is turned into running
//! components (config, secrets, skills, store, provider gateway). The
//! server, the CLI and library embedders all go through `Workspace::open`.
//!
//! Opening a workspace creates `.orchopork/` but never runs `git init` on
//! its own: that only happens when the user asks for it (setup wizard or
//! `orchopork init --git-init`).

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::git::{GitRepo, STATE_DIR};
use crate::providers::{ClaudeProvider, Gateway, OllamaProvider, OpenAiCompatProvider, ProviderId};
use crate::secrets::SecretStore;
use crate::skills::SkillRegistry;
use crate::storage::Store;

pub struct Workspace {
    pub root: PathBuf,
    pub state_dir: PathBuf,
    config: RwLock<Config>,
    pub secrets: SecretStore,
    pub skills: Arc<SkillRegistry>,
    pub store: Store,
    pub gateway: Arc<Gateway>,
    pub repo: GitRepo,
}

impl Workspace {
    /// Whether `root` has been through setup (a config file exists).
    pub fn is_initialized(root: &Path) -> bool {
        root.join(STATE_DIR).join("config.yaml").is_file()
    }

    /// Opens (creating on first use) `<root>/.orchopork/`.
    pub async fn open(root: &Path) -> Result<Arc<Self>> {
        let root = tokio::fs::canonicalize(root)
            .await
            .map_err(|e| Error::InvalidRequest(format!("workspace {}: {e}", root.display())))?;
        if !root.is_dir() {
            return Err(Error::InvalidRequest(format!("workspace {} is not a directory", root.display())));
        }
        let state_dir = root.join(STATE_DIR);
        tokio::fs::create_dir_all(&state_dir).await?;

        let config_path = state_dir.join("config.yaml");
        let config = Config::load(&config_path)?;
        if !config_path.exists() {
            config.save(&config_path)?;
        }
        let secrets = SecretStore::open(state_dir.join("secrets.yaml"))?;
        let skills = Arc::new(SkillRegistry::open(state_dir.join("skills"), state_dir.join("skills.enabled.yaml"))?);
        let store = Store::open(&state_dir.join("state.db")).await?;
        let gateway = Arc::new(Gateway::new(store.clone(), config.monthly_cap_usd));
        let repo = GitRepo::new(&root);
        if repo.is_repo().await {
            repo.ensure_state_dir_excluded().await?;
        }

        let ws = Arc::new(Self { root, state_dir, config: RwLock::new(config), secrets, skills, store, gateway, repo });
        ws.register_providers();
        Ok(ws)
    }

    pub fn config(&self) -> Config {
        self.config.read().unwrap().clone()
    }

    /// Validates, persists and applies a new config (providers are
    /// re-registered so URL or cap changes take effect immediately).
    pub fn update_config(&self, f: impl FnOnce(&mut Config)) -> Result<Config> {
        let mut next = self.config();
        f(&mut next);
        next.save(&self.state_dir.join("config.yaml"))?;
        *self.config.write().unwrap() = next.clone();
        self.register_providers();
        Ok(next)
    }

    /// Stores (or with an empty key, removes) a provider API key and
    /// re-registers providers.
    pub fn set_provider_key(&self, provider: ProviderId, key: &str) -> Result<()> {
        if provider.is_local() {
            return Err(Error::InvalidRequest(format!("{} is local and needs no API key", provider.as_str())));
        }
        self.secrets.set(provider.as_str(), key)?;
        self.register_providers();
        Ok(())
    }

    /// Rebuilds the gateway's provider set from config + secrets. Ollama is
    /// always registered (reachability is checked when it is used), cloud
    /// providers only with a key.
    pub fn register_providers(&self) {
        let cfg = self.config();
        let gw = &self.gateway;
        gw.set_cap(cfg.monthly_cap_usd);
        gw.unregister_all();
        gw.register(Box::new(OllamaProvider::new(cfg.ollama_url.clone(), cfg.ollama_num_ctx)));
        if let Some(url) = cfg.llamacpp_url.as_deref().filter(|u| !u.trim().is_empty()) {
            gw.register(Box::new(OpenAiCompatProvider::llamacpp(url.trim())));
        }
        if let Some(k) = self.secrets.provider_key(ProviderId::Claude) {
            gw.register(Box::new(ClaudeProvider::new(k)));
        }
        if let Some(k) = self.secrets.provider_key(ProviderId::DeepSeek) {
            gw.register(Box::new(OpenAiCompatProvider::deepseek(k)));
        }
        if let Some(k) = self.secrets.provider_key(ProviderId::Gemini) {
            gw.register(Box::new(OpenAiCompatProvider::gemini(k)));
        }
    }
}
