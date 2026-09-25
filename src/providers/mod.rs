//! LLM provider gateway: local models (Ollama, llama.cpp) and cloud APIs
//! (Claude, DeepSeek, Gemini) behind one trait, metered by a hard monthly
//! budget.
//!
//! The gateway, not the caller, owns the spend ledger: every successful
//! call is recorded the moment it returns, whether or not the agent later
//! keeps the reply. Before a cloud call it *reserves* the call's worst-case
//! cost (prompt estimate + `max_tokens` of output) under a lock, so
//! concurrent runs cannot jointly overshoot the cap.

pub mod pricing;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::ModelRef;
use crate::error::{Error, Result};
use crate::storage::{Store, month_start_unix, now_unix};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProviderId {
    #[serde(rename = "ollama")]
    Ollama,
    #[serde(rename = "llamacpp")]
    LlamaCpp,
    #[serde(rename = "claude")]
    Claude,
    #[serde(rename = "deepseek")]
    DeepSeek,
    #[serde(rename = "gemini")]
    Gemini,
    /// An external agent (Claude Code, Gemini CLI, Codex…) driven over the
    /// Agent Client Protocol; `model` is the agent id from config. Never
    /// called through the gateway.
    #[serde(rename = "acp")]
    Acp,
}

impl ProviderId {
    pub const ALL: [ProviderId; 5] =
        [ProviderId::Ollama, ProviderId::LlamaCpp, ProviderId::Claude, ProviderId::DeepSeek, ProviderId::Gemini];

    pub fn as_str(self) -> &'static str {
        match self {
            ProviderId::Ollama => "ollama",
            ProviderId::LlamaCpp => "llamacpp",
            ProviderId::Claude => "claude",
            ProviderId::DeepSeek => "deepseek",
            ProviderId::Gemini => "gemini",
            ProviderId::Acp => "acp",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        if s.trim() == "acp" {
            return Ok(ProviderId::Acp);
        }
        Self::ALL.into_iter().find(|p| p.as_str() == s.trim()).ok_or_else(|| {
            Error::InvalidRequest(format!("unknown provider {s:?} (ollama|llamacpp|claude|deepseek|gemini)"))
        })
    }

    /// Local providers are free and never gated by the budget.
    pub fn is_local(self) -> bool {
        matches!(self, ProviderId::Ollama | ProviderId::LlamaCpp)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub system: String,
    /// Must start with a user message and end with one (current models
    /// reject assistant prefill).
    pub messages: Vec<Message>,
    pub max_tokens: u32,
}

/// What a provider returns.
#[derive(Debug, Clone, Default)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// The reply hit `max_tokens` and is cut off.
    pub truncated: bool,
}

/// What the gateway returns: the completion plus its metered cost.
#[derive(Debug, Clone, Serialize)]
pub struct CallOutcome {
    pub model: ModelRef,
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cost_usd: f64,
    pub truncated: bool,
}

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> ProviderId;
    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion>;
    /// Models the endpoint offers. For cloud providers this doubles as an
    /// API key check.
    async fn list_models(&self) -> Result<Vec<String>>;
}

pub struct Gateway {
    providers: RwLock<HashMap<ProviderId, Arc<dyn Provider>>>,
    store: Store,
    cap_usd: RwLock<f64>,
    /// Worst-case cost of cloud calls currently in flight.
    reserved_usd: tokio::sync::Mutex<f64>,
}

impl Gateway {
    pub fn new(store: Store, monthly_cap_usd: f64) -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            store,
            cap_usd: RwLock::new(monthly_cap_usd),
            reserved_usd: tokio::sync::Mutex::new(0.0),
        }
    }

    pub fn set_cap(&self, usd: f64) {
        *self.cap_usd.write().unwrap() = usd;
    }

    pub fn cap(&self) -> f64 {
        *self.cap_usd.read().unwrap()
    }

    /// Register or replace a provider. In-flight calls keep the instance
    /// they already hold.
    pub fn register(&self, provider: Box<dyn Provider>) {
        let provider: Arc<dyn Provider> = Arc::from(provider);
        self.providers.write().unwrap().insert(provider.id(), provider);
    }

    pub fn unregister_all(&self) {
        self.providers.write().unwrap().clear();
    }

    pub fn configured(&self) -> Vec<ProviderId> {
        let mut v: Vec<_> = self.providers.read().unwrap().keys().copied().collect();
        v.sort();
        v
    }

    pub fn provider(&self, id: ProviderId) -> Result<Arc<dyn Provider>> {
        self.providers.read().unwrap().get(&id).cloned().ok_or_else(|| {
            Error::Provider(format!(
                "{} is not configured{}",
                id.as_str(),
                if id.is_local() { " (set its URL in settings)" } else { " (add an API key in settings)" }
            ))
        })
    }

    pub async fn spent_this_month(&self) -> Result<f64> {
        self.store.spend_since(month_start_unix(now_unix())).await
    }

    pub async fn budget_remaining(&self) -> Result<f64> {
        let reserved = *self.reserved_usd.lock().await;
        Ok((self.cap() - self.spent_this_month().await? - reserved).max(0.0))
    }

    /// Runs one completion and records its cost against `run_id`.
    ///
    /// Cloud calls are refused *before* any network I/O when this month's
    /// spend, plus everything in flight, plus this call's worst case would
    /// exceed the cap. The worst case uses a conservative token estimate
    /// (bytes / 2.5) and the full `max_tokens`, so the cap holds even if
    /// the model writes its longest possible answer.
    pub async fn complete(&self, target: &ModelRef, req: &CompletionRequest, run_id: &str) -> Result<CallOutcome> {
        let provider = self.provider(target.provider)?;
        let reservation = if target.provider.is_local() {
            0.0
        } else {
            let est = estimate_cost(target, req);
            let mut reserved = self.reserved_usd.lock().await;
            let spent = self.spent_this_month().await?;
            let cap = self.cap();
            if spent + *reserved + est > cap {
                return Err(Error::Budget(format!(
                    "monthly cap ${cap:.2} would be exceeded (spent ${spent:.2}, in flight ${:.2}, this call up to ${est:.2})",
                    *reserved
                )));
            }
            *reserved += est;
            est
        };

        let result = provider.complete(&target.model, req).await;
        let outcome = match result {
            Ok(c) => {
                let cost = pricing::cost_usd(target.provider, &target.model, c.prompt_tokens, c.completion_tokens);
                let recorded = self
                    .store
                    .record_spend(
                        run_id,
                        target.provider.as_str(),
                        &target.model,
                        c.prompt_tokens,
                        c.completion_tokens,
                        cost,
                    )
                    .await;
                recorded.map(|()| CallOutcome {
                    model: target.clone(),
                    text: c.text,
                    prompt_tokens: c.prompt_tokens,
                    completion_tokens: c.completion_tokens,
                    cost_usd: cost,
                    truncated: c.truncated,
                })
            }
            Err(e) => Err(e),
        };
        // Released only after the real cost is in the ledger, so there is
        // no window in which the call counts as neither.
        if reservation > 0.0 {
            let mut reserved = self.reserved_usd.lock().await;
            *reserved = (*reserved - reservation).max(0.0);
        }
        outcome
    }
}

fn estimate_cost(target: &ModelRef, req: &CompletionRequest) -> f64 {
    let bytes = req.system.len() + req.messages.iter().map(|m| m.content.len() + 16).sum::<usize>();
    let prompt_tokens = (bytes as u64) * 2 / 5 + 64;
    pricing::cost_usd(target.provider, &target.model, prompt_tokens, u64::from(req.max_tokens))
}

// ---- shared HTTP plumbing ------------------------------------------------

fn client(timeout_secs: u64) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client")
}

/// Sends `rb` and decodes JSON, retrying transient failures (connection
/// errors, 408/409/429/5xx/529) with backoff that honours `retry-after`.
/// Non-retryable errors keep the provider's own error message.
async fn send_json<T: DeserializeOwned>(label: &str, rb: reqwest::RequestBuilder, retries: u32) -> Result<T> {
    let mut attempt = 0;
    loop {
        let req = rb.try_clone().ok_or_else(|| Error::Provider(format!("{label}: request cannot be retried")))?;
        match req.send().await {
            Err(e) => {
                if attempt < retries && (e.is_connect() || e.is_timeout()) {
                    backoff(attempt, None).await;
                    attempt += 1;
                    continue;
                }
                let what = if e.is_connect() {
                    format!("cannot connect to {}", e.url().map(|u| u.as_str()).unwrap_or("endpoint"))
                } else if e.is_timeout() {
                    "request timed out".to_string()
                } else {
                    e.to_string()
                };
                return Err(Error::Provider(format!("{label}: {what}")));
            }
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return resp
                        .json::<T>()
                        .await
                        .map_err(|e| Error::Provider(format!("{label}: unexpected response: {e}")));
                }
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok());
                let body = resp.text().await.unwrap_or_default();
                if attempt < retries && matches!(status.as_u16(), 408 | 409 | 429 | 500 | 502 | 503 | 504 | 529) {
                    backoff(attempt, retry_after).await;
                    attempt += 1;
                    continue;
                }
                return Err(Error::Provider(format!("{label}: HTTP {}: {}", status.as_u16(), error_message(&body))));
            }
        }
    }
}

async fn backoff(attempt: u32, retry_after: Option<u64>) {
    let secs = retry_after.unwrap_or(2u64.saturating_pow(attempt + 1)).min(30);
    tokio::time::sleep(Duration::from_secs(secs)).await;
}

/// Pulls the human-readable message out of the common error shapes
/// (`{"error":{"message":..}}`, `{"error":".."}`), else a clipped body.
fn error_message(body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        let e = &v["error"];
        if let Some(m) = e["message"].as_str().or_else(|| e.as_str()).or_else(|| v["message"].as_str()) {
            return m.to_string();
        }
    }
    crate::fsutil::clip(body.trim(), 300)
}

fn wire_messages<'a>(system: Option<&'a str>, messages: &'a [Message]) -> Vec<serde_json::Value> {
    system
        .filter(|s| !s.is_empty())
        .map(|s| json!({ "role": "system", "content": s }))
        .into_iter()
        .chain(messages.iter().map(|m| json!({ "role": m.role.as_str(), "content": m.content })))
        .collect()
}

// ---- Ollama (local, free) ------------------------------------------------

pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    num_ctx: u32,
}

impl OllamaProvider {
    pub fn new(base_url: impl Into<String>, num_ctx: u32) -> Self {
        // Local models on modest hardware can take many minutes per turn.
        Self { client: client(1800), base_url: base_url.into().trim_end_matches('/').to_string(), num_ctx }
    }

    fn hint(&self, e: Error) -> Error {
        match e {
            Error::Provider(m) if m.contains("cannot connect") => {
                Error::Provider(format!("{m} (is `ollama serve` running at {}?)", self.base_url))
            }
            other => other,
        }
    }
}

#[derive(Deserialize)]
struct OllamaChatResp {
    message: OllamaMsg,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
    #[serde(default)]
    done_reason: Option<String>,
}

#[derive(Deserialize)]
struct OllamaMsg {
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaTag>,
}

#[derive(Deserialize)]
struct OllamaTag {
    name: String,
}

#[async_trait::async_trait]
impl Provider for OllamaProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Ollama
    }

    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion> {
        let body = json!({
            "model": model,
            "messages": wire_messages(Some(&req.system), &req.messages),
            "stream": false,
            "options": { "num_ctx": self.num_ctx, "num_predict": req.max_tokens },
        });
        let rb = self.client.post(format!("{}/api/chat", self.base_url)).json(&body);
        let r: OllamaChatResp = send_json("ollama", rb, 0).await.map_err(|e| self.hint(e))?;
        Ok(Completion {
            text: r.message.content,
            prompt_tokens: r.prompt_eval_count,
            completion_tokens: r.eval_count,
            truncated: r.done_reason.as_deref() == Some("length"),
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let rb = self.client.get(format!("{}/api/tags", self.base_url)).timeout(Duration::from_secs(5));
        let tags: OllamaTags = send_json("ollama", rb, 0).await.map_err(|e| self.hint(e))?;
        let mut v: Vec<String> = tags.models.into_iter().map(|m| m.name).collect();
        v.sort();
        Ok(v)
    }
}

// ---- Claude (Anthropic Messages API) ---------------------------------------

pub struct ClaudeProvider {
    client: reqwest::Client,
    api_key: String,
}

impl ClaudeProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { client: client(600), api_key: api_key.into() }
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        self.client.get(url).header("x-api-key", &self.api_key).header("anthropic-version", "2023-06-01")
    }
}

#[derive(Deserialize)]
struct ClaudeResp {
    #[serde(default)]
    content: Vec<ClaudeBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    usage: ClaudeUsage,
}

#[derive(Deserialize)]
struct ClaudeBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct ClaudeUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[async_trait::async_trait]
impl Provider for ClaudeProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Claude
    }

    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion> {
        let body = json!({
            "model": model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": wire_messages(None, &req.messages),
        });
        let rb = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body);
        let r: ClaudeResp = send_json("claude", rb, 3).await?;
        if r.stop_reason.as_deref() == Some("refusal") {
            return Err(Error::Provider("claude declined this request (stop_reason: refusal)".into()));
        }
        // Only text blocks: thinking blocks are the model's private reasoning.
        let text = r.content.into_iter().filter(|b| b.kind == "text").map(|b| b.text).collect::<Vec<_>>().join("");
        Ok(Completion {
            text,
            prompt_tokens: r.usage.input_tokens,
            completion_tokens: r.usage.output_tokens,
            truncated: r.stop_reason.as_deref() == Some("max_tokens"),
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let list: ModelList = send_json("claude", self.get("https://api.anthropic.com/v1/models?limit=100"), 1).await?;
        Ok(list.data.into_iter().map(|m| m.id).collect())
    }
}

// ---- OpenAI-compatible: DeepSeek, Gemini, llama.cpp server -----------------

pub struct OpenAiCompatProvider {
    id: ProviderId,
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    /// Provider-side ceiling on `max_tokens` (DeepSeek rejects > 8192).
    max_tokens_cap: u32,
}

impl OpenAiCompatProvider {
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::new(ProviderId::DeepSeek, "https://api.deepseek.com/v1", Some(api_key.into()), 8192, 600)
    }

    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::new(
            ProviderId::Gemini,
            "https://generativelanguage.googleapis.com/v1beta/openai",
            Some(api_key.into()),
            u32::MAX,
            600,
        )
    }

    /// A local `llama-server`; `base_url` is its OpenAI root, e.g.
    /// `http://127.0.0.1:8080/v1`.
    pub fn llamacpp(base_url: impl Into<String>) -> Self {
        Self::new(ProviderId::LlamaCpp, &base_url.into(), None, u32::MAX, 1800)
    }

    fn new(id: ProviderId, base_url: &str, api_key: Option<String>, max_tokens_cap: u32, timeout: u64) -> Self {
        Self {
            id,
            client: client(timeout),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            max_tokens_cap,
        }
    }

    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(k) => rb.bearer_auth(k),
            None => rb,
        }
    }
}

#[derive(Deserialize)]
struct OaiResp {
    #[serde(default)]
    choices: Vec<OaiChoice>,
    #[serde(default)]
    usage: Option<OaiUsage>,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OaiMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OaiMessage {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct OaiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[async_trait::async_trait]
impl Provider for OpenAiCompatProvider {
    fn id(&self) -> ProviderId {
        self.id
    }

    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion> {
        let body = json!({
            "model": model,
            "messages": wire_messages(Some(&req.system), &req.messages),
            "max_tokens": req.max_tokens.min(self.max_tokens_cap),
        });
        let rb = self.auth(self.client.post(format!("{}/chat/completions", self.base_url)).json(&body));
        let retries = if self.id.is_local() { 0 } else { 3 };
        let r: OaiResp = send_json(self.id.as_str(), rb, retries).await?;
        let choice = r
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Provider(format!("{}: response had no choices", self.id.as_str())))?;
        let usage = r.usage.unwrap_or(OaiUsage { prompt_tokens: 0, completion_tokens: 0 });
        Ok(Completion {
            text: choice.message.content.unwrap_or_default(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            truncated: choice.finish_reason.as_deref() == Some("length"),
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let rb = self.auth(self.client.get(format!("{}/models", self.base_url))).timeout(Duration::from_secs(10));
        let list: ModelList = send_json(self.id.as_str(), rb, 0).await?;
        let mut v: Vec<String> =
            list.data.into_iter().map(|m| m.id.trim_start_matches("models/").to_string()).collect();
        v.sort();
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Stub {
        id: ProviderId,
        delay_ms: u64,
    }

    #[async_trait::async_trait]
    impl Provider for Stub {
        fn id(&self) -> ProviderId {
            self.id
        }
        async fn complete(&self, _model: &str, _req: &CompletionRequest) -> Result<Completion> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(Completion { text: "ok".into(), prompt_tokens: 1000, completion_tokens: 1000, truncated: false })
        }
        async fn list_models(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }

    fn req(max_tokens: u32) -> CompletionRequest {
        CompletionRequest { system: "s".into(), messages: vec![Message::user("hi")], max_tokens }
    }

    async fn store() -> (tempfile::TempDir, Store) {
        let d = tempfile::tempdir().unwrap();
        let s = Store::open(&d.path().join("s.db")).await.unwrap();
        (d, s)
    }

    fn haiku() -> ModelRef {
        ModelRef { provider: ProviderId::Claude, model: "claude-haiku-4-5".into() }
    }

    #[tokio::test]
    async fn unconfigured_provider_errors() {
        let (_d, store) = store().await;
        let gw = Gateway::new(store, 40.0);
        assert!(matches!(gw.complete(&haiku(), &req(10), "r").await, Err(Error::Provider(_))));
    }

    #[tokio::test]
    async fn every_call_is_recorded_and_local_bypasses_the_cap() {
        let (_d, store) = store().await;
        store.record_spend("r", "claude", "m", 0, 0, 999.0).await.unwrap();
        let gw = Gateway::new(store.clone(), 40.0);
        gw.register(Box::new(Stub { id: ProviderId::Ollama, delay_ms: 0 }));
        gw.register(Box::new(Stub { id: ProviderId::Claude, delay_ms: 0 }));
        let local = ModelRef { provider: ProviderId::Ollama, model: "m".into() };
        assert_eq!(gw.complete(&local, &req(10), "r").await.unwrap().cost_usd, 0.0);
        assert!(matches!(gw.complete(&haiku(), &req(10), "r").await, Err(Error::Budget(_))));
    }

    #[tokio::test]
    async fn concurrent_calls_cannot_jointly_overshoot_the_cap() {
        let (_d, store) = store().await;
        // Each call reserves ~$0.50 (100k output tokens at $5/M), cap is $1.
        let gw = Arc::new(Gateway::new(store.clone(), 1.0));
        gw.register(Box::new(Stub { id: ProviderId::Claude, delay_ms: 200 }));
        let calls: Vec<_> = (0..3)
            .map(|_| {
                let gw = gw.clone();
                tokio::spawn(async move { gw.complete(&haiku(), &req(99_000), "r").await })
            })
            .collect();
        let mut ok = 0;
        let mut budget = 0;
        for c in calls {
            match c.await.unwrap() {
                Ok(_) => ok += 1,
                Err(Error::Budget(_)) => budget += 1,
                Err(e) => panic!("{e}"),
            }
        }
        assert_eq!((ok, budget), (2, 1));
        let spent = store.spend_since(0).await.unwrap();
        assert!((spent - 2.0 * pricing::cost_usd(ProviderId::Claude, "claude-haiku-4-5", 1000, 1000)).abs() < 1e-9);
        assert!((gw.budget_remaining().await.unwrap() - (1.0 - spent)).abs() < 1e-9, "reservations released");
    }

    #[test]
    fn provider_error_messages_are_extracted() {
        assert_eq!(error_message(r#"{"type":"error","error":{"type":"x","message":"bad model"}}"#), "bad model");
        assert_eq!(error_message(r#"{"error":"model 'x' not found"}"#), "model 'x' not found");
        assert_eq!(error_message("plain"), "plain");
    }
}
