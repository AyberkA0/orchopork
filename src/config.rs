//! Per-workspace settings, persisted as `.orchopork/config.yaml`.
//!
//! Every field has a default, so a hand-edited file only needs the keys it
//! changes. Secrets never live here (see `secrets.rs`): this file is safe to
//! show in the UI.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::providers::ProviderId;

/// A concrete model on a concrete provider, written `provider:model`
/// (e.g. `ollama:qwen2.5-coder:14b`, `claude:claude-sonnet-5`), optionally
/// tuned with options: `claude:claude-opus-5-5?effort=high`,
/// `acp:claude-code?model=opus&effort=xhigh`.
///
/// Options: `effort` for API models (Claude `output_config.effort`, Gemini
/// `reasoning_effort`); for ACP agents, any session config option the
/// agent advertises (typically `model` and `effort`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider: ProviderId,
    pub model: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, String>,
}

impl ModelRef {
    pub fn parse(s: &str) -> Result<Self> {
        let (p, rest) =
            s.split_once(':').ok_or_else(|| Error::InvalidRequest(format!("expected provider:model, got {s:?}")))?;
        let provider = ProviderId::parse(p)?;
        let (m, query) = rest.split_once('?').unwrap_or((rest, ""));
        if m.trim().is_empty() {
            return Err(Error::InvalidRequest(format!("empty model name in {s:?}")));
        }
        let mut options = BTreeMap::new();
        for pair in query.split('&').filter(|p| !p.is_empty()) {
            let (k, v) =
                pair.split_once('=').ok_or_else(|| Error::InvalidRequest(format!("expected key=value in {pair:?}")))?;
            options.insert(k.trim().to_string(), v.trim().to_string());
        }
        Ok(Self { provider, model: m.trim().to_string(), options })
    }

    pub fn new(provider: ProviderId, model: impl Into<String>) -> Self {
        Self { provider, model: model.into(), options: BTreeMap::new() }
    }

    pub fn option(&self, key: &str) -> Option<&str> {
        self.options.get(key).map(String::as_str).filter(|v| !v.is_empty() && *v != "default")
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.provider.as_str(), self.model)?;
        for (i, (k, v)) in self.options.iter().enumerate() {
            write!(f, "{}{k}={v}", if i == 0 { '?' } else { '&' })?;
        }
        Ok(())
    }
}

/// Which model plays which role. Only `actor` is required: without a
/// planner the actor plans, without a critic finished work is accepted once
/// verification passes, and without an escalation model a struggling actor
/// just keeps its own model.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Routing {
    pub planner: Option<ModelRef>,
    pub actor: Option<ModelRef>,
    pub critic: Option<ModelRef>,
    pub escalation: Option<ModelRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Limits {
    /// Steps a run may take per start/resume before it pauses for a human
    /// look. The main guard against a local model looping for hours.
    pub steps_per_session: u32,
    /// Consecutive failed actor turns (unparseable reply, tool error,
    /// validator rejection) before the run pauses.
    pub max_failures: u32,
    /// Consecutive failures after which the next actor turn goes to the
    /// escalation model (if one is configured).
    pub escalate_after: u32,
    /// Review rounds before the run pauses instead of looping with the critic.
    pub max_review_rounds: u32,
    /// Whether the agent may call `run_command`. Verification commands and
    /// skill tools still run: they are written by you, not by the model.
    pub allow_commands: bool,
    pub command_timeout_secs: u64,
    /// Per-observation cap when tool output is fed back to the model.
    pub tool_output_chars: usize,
    /// Transcript budget (in chars) for local and cloud models respectively.
    pub local_context_chars: usize,
    pub cloud_context_chars: usize,
    /// Output cap for every LLM call; also bounds the worst-case cost the
    /// budget guard reserves before a cloud call.
    pub max_output_tokens: u32,
    /// Wall-clock limit for one external (ACP) agent turn.
    pub external_timeout_secs: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            steps_per_session: 40,
            max_failures: 6,
            escalate_after: 2,
            max_review_rounds: 3,
            allow_commands: true,
            command_timeout_secs: 300,
            tool_output_chars: 8_000,
            local_context_chars: 40_000,
            cloud_context_chars: 200_000,
            max_output_tokens: 16_000,
            external_timeout_secs: 3600,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteAuth {
    /// Pushes use whatever git already has (credential helper, ssh-agent).
    #[default]
    None,
    /// Pushes send the stored GitHub token as an HTTP header.
    Pat,
    /// Pushes rely on the user's SSH setup; recorded for the UI only.
    Ssh,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Set once the setup wizard has been completed for this workspace.
    pub onboarded: bool,
    pub monthly_cap_usd: f64,
    pub ollama_url: String,
    /// Ollama's context window per request. Ollama's own default is small
    /// and it silently truncates longer prompts, which wrecks agent turns.
    pub ollama_num_ctx: u32,
    /// Optional llama.cpp `llama-server` (OpenAI-compatible) base URL, e.g.
    /// `http://127.0.0.1:8080/v1`.
    pub llamacpp_url: Option<String>,
    pub remote_auth: RemoteAuth,
    pub routing: Routing,
    pub limits: Limits,
    /// Agents driven over ACP (see `acp.rs`), usable anywhere a model is.
    pub external_agents: Vec<crate::acp::ExternalAgent>,
    /// OpenAI-compatible endpoints you added (keys live in secrets as
    /// `endpoint:<id>`).
    pub endpoints: Vec<Endpoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Endpoint {
    pub id: String,
    pub name: String,
    pub base_url: String,
    /// Runs on this machine: free, never budget-gated.
    #[serde(default)]
    pub local: bool,
    /// USD per 1M (input, output) tokens; unknown hosted endpoints are
    /// priced conservatively high.
    #[serde(default)]
    pub price_in: Option<f64>,
    #[serde(default)]
    pub price_out: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            onboarded: false,
            monthly_cap_usd: std::env::var("ORCHOPORK_MONTHLY_CAP_USD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(40.0),
            ollama_url: std::env::var("ORCHOPORK_OLLAMA_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            ollama_num_ctx: 16_384,
            llamacpp_url: None,
            remote_auth: RemoteAuth::None,
            routing: Routing::default(),
            limits: Limits::default(),
            external_agents: crate::acp::default_agents(),
            endpoints: Vec::new(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) if text.trim().is_empty() => Ok(Self::default()),
            Ok(text) => Ok(serde_yaml::from_str(&text)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        crate::fsutil::write_atomic(path, serde_yaml::to_string(self)?.as_bytes(), false)
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.monthly_cap_usd.is_finite() && self.monthly_cap_usd >= 0.0) {
            return Err(Error::InvalidRequest("monthly_cap_usd must be a non-negative number".into()));
        }
        let l = &self.limits;
        if l.steps_per_session == 0 || l.max_failures == 0 || l.max_output_tokens < 256 {
            return Err(Error::InvalidRequest(
                "limits: steps_per_session and max_failures must be > 0, max_output_tokens >= 256".into(),
            ));
        }
        Ok(())
    }

    /// Transcript budget for a model, by where it runs.
    pub fn context_chars(&self, m: &ModelRef) -> usize {
        if m.provider.is_local() { self.limits.local_context_chars } else { self.limits.cloud_context_chars }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ref_splits_on_the_first_colon_only() {
        let m = ModelRef::parse("ollama:qwen2.5-coder:14b").unwrap();
        assert_eq!((m.provider, m.model.as_str()), (ProviderId::Ollama, "qwen2.5-coder:14b"));
        assert_eq!(m.to_string(), "ollama:qwen2.5-coder:14b");
        assert!(ModelRef::parse("nope").is_err());
        assert!(ModelRef::parse("mystery:x").is_err());
        let t = ModelRef::parse("acp:claude-code?model=opus&effort=high").unwrap();
        assert_eq!(
            (t.model.as_str(), t.option("model"), t.option("effort")),
            ("claude-code", Some("opus"), Some("high"))
        );
        assert_eq!(t.to_string(), "acp:claude-code?effort=high&model=opus");
    }

    #[test]
    fn partial_file_fills_in_defaults_and_round_trips() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("config.yaml");
        std::fs::write(&p, "limits:\n  steps_per_session: 5\nrouting:\n  actor: {provider: ollama, model: m}\n")
            .unwrap();
        let c = Config::load(&p).unwrap();
        assert_eq!(c.limits.steps_per_session, 5);
        assert_eq!(c.limits.max_failures, Limits::default().max_failures);
        c.save(&p).unwrap();
        assert_eq!(Config::load(&p).unwrap(), c);
        assert!(!Config::load(&d.path().join("missing.yaml")).unwrap().onboarded);
    }
}
