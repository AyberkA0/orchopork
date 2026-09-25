//! LLM provider gateway: local Ollama + cloud frontier APIs behind one
//! trait, metered by a monthly-budget circuit breaker.
//!
//! Cost is computed from the token counts each provider reports and pushed
//! into `storage::spend_ledger` by the *caller* (the checkpointer already
//! owns that transaction); this module only decides whether a call is
//! allowed to proceed and what it would cost.

pub mod pricing;

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::storage::{Store, month_start_unix, now_unix};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderId {
    Ollama,
    Claude,
    DeepSeek,
    Gemini,
}

impl ProviderId {
    pub fn is_local(self) -> bool {
        matches!(self, ProviderId::Ollama)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cost_usd: f64,
}

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> ProviderId;
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse>;
}

/// Routes completions to the right backend and enforces the monthly spend
/// cap before any cloud call leaves the process. Local (Ollama) calls are
/// never gated: they cost nothing and must keep working if the cloud
/// budget is exhausted or the network is down.
pub struct Gateway {
    providers: RwLock<HashMap<ProviderId, std::sync::Arc<dyn Provider>>>,
    store: Store,
    monthly_cap_usd: f64,
}

impl Gateway {
    pub fn new(store: Store, monthly_cap_usd: f64) -> Self {
        Self { providers: RwLock::new(HashMap::new()), store, monthly_cap_usd }
    }

    /// Register/replace a provider (e.g. once its API key is set in step 4,
    /// or hot-swapped later from the dashboard). Existing in-flight calls
    /// keep using the instance they already borrowed.
    pub fn register(&self, provider: Box<dyn Provider>) {
        let provider: std::sync::Arc<dyn Provider> = std::sync::Arc::from(provider);
        self.providers.write().unwrap().insert(provider.id(), provider);
    }

    pub fn configured(&self) -> Vec<ProviderId> {
        self.providers.read().unwrap().keys().copied().collect()
    }

    pub async fn spent_this_month(&self) -> Result<f64> {
        self.store.spend_since(month_start_unix(now_unix())).await
    }

    pub async fn budget_remaining(&self) -> Result<f64> {
        Ok((self.monthly_cap_usd - self.spent_this_month().await?).max(0.0))
    }

    /// Runs the completion. For cloud providers, refuses *before* the
    /// network call if the current month's ledger has already met or
    /// exceeded the cap: the breaker trips on spend already recorded, not
    /// on the cost of the call about to be made, so it cannot be bypassed
    /// by racing large requests in under the wire.
    pub async fn complete(&self, id: ProviderId, req: &CompletionRequest) -> Result<CompletionResponse> {
        if !id.is_local() {
            let spent = self.spent_this_month().await?;
            if spent >= self.monthly_cap_usd {
                return Err(Error::Budget(format!(
                    "monthly cap ${:.2} reached (spent ${spent:.2}); route to a local model or raise the cap",
                    self.monthly_cap_usd
                )));
            }
        }
        // Clone the Arc out and drop the lock before the network call, so a
        // key rotation never blocks on (or is blocked by) an in-flight request.
        let provider = self
            .providers
            .read()
            .unwrap()
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::Provider(format!("provider {id:?} not configured")))?;
        provider.complete(req).await
    }
}

// ---- Ollama (local, free) --------------------------------------------

pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder().timeout(Duration::from_secs(300)).build().expect("reqwest client"),
            base_url: base_url.into(),
        }
    }
}

#[derive(Serialize)]
struct OllamaChatReq<'a> {
    model: &'a str,
    messages: Vec<OllamaMsg<'a>>,
    stream: bool,
}

#[derive(Serialize)]
struct OllamaMsg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OllamaChatResp {
    message: OllamaMsgOwned,
    #[serde(default)]
    prompt_eval_count: u64,
    #[serde(default)]
    eval_count: u64,
}

#[derive(Deserialize)]
struct OllamaMsgOwned {
    content: String,
}

#[async_trait::async_trait]
impl Provider for OllamaProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Ollama
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse> {
        let mut messages = vec![OllamaMsg { role: "system", content: &req.system }];
        messages.extend(req.messages.iter().map(|m| OllamaMsg { role: role_str(m.role), content: &m.content }));
        let body = OllamaChatReq { model: &req.model, messages, stream: false };
        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("ollama: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::Provider(format!("ollama: HTTP {}", resp.status())));
        }
        let parsed: OllamaChatResp =
            resp.json().await.map_err(|e| Error::Provider(format!("ollama: bad response: {e}")))?;
        Ok(CompletionResponse {
            text: parsed.message.content,
            prompt_tokens: parsed.prompt_eval_count,
            completion_tokens: parsed.eval_count,
            cost_usd: 0.0,
        })
    }
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

// ---- Claude (Anthropic Messages API) ----------------------------------

pub struct ClaudeProvider {
    client: reqwest::Client,
    api_key: String,
}

impl ClaudeProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder().timeout(Duration::from_secs(120)).build().expect("reqwest client"),
            api_key: api_key.into(),
        }
    }
}

#[derive(Serialize)]
struct ClaudeReq<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<OllamaMsg<'a>>,
}

#[derive(Deserialize)]
struct ClaudeResp {
    content: Vec<ClaudeBlock>,
    usage: ClaudeUsage,
}

#[derive(Deserialize)]
struct ClaudeBlock {
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct ClaudeUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[async_trait::async_trait]
impl Provider for ClaudeProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Claude
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse> {
        let messages: Vec<_> =
            req.messages.iter().map(|m| OllamaMsg { role: role_str(m.role), content: &m.content }).collect();
        let body = ClaudeReq { model: &req.model, max_tokens: 4096, system: &req.system, messages };
        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("claude: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::Provider(format!("claude: HTTP {}", resp.status())));
        }
        let parsed: ClaudeResp =
            resp.json().await.map_err(|e| Error::Provider(format!("claude: bad response: {e}")))?;
        let text = parsed.content.into_iter().map(|b| b.text).collect();
        let cost =
            pricing::cost_usd(ProviderId::Claude, &req.model, parsed.usage.input_tokens, parsed.usage.output_tokens);
        Ok(CompletionResponse {
            text,
            prompt_tokens: parsed.usage.input_tokens,
            completion_tokens: parsed.usage.output_tokens,
            cost_usd: cost,
        })
    }
}

// ---- DeepSeek / Gemini share an OpenAI-compatible chat/completions shape --

pub struct OpenAiCompatProvider {
    id: ProviderId,
    client: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiCompatProvider {
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::new(ProviderId::DeepSeek, "https://api.deepseek.com/v1", api_key)
    }

    pub fn gemini(api_key: impl Into<String>) -> Self {
        Self::new(ProviderId::Gemini, "https://generativelanguage.googleapis.com/v1beta/openai", api_key)
    }

    fn new(id: ProviderId, base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            id,
            client: reqwest::Client::builder().timeout(Duration::from_secs(120)).build().expect("reqwest client"),
            base_url: base_url.into(),
            api_key: api_key.into(),
        }
    }
}

#[derive(Serialize)]
struct OaiReq<'a> {
    model: &'a str,
    messages: Vec<OllamaMsg<'a>>,
}

#[derive(Deserialize)]
struct OaiResp {
    choices: Vec<OaiChoice>,
    usage: OaiUsage,
}

#[derive(Deserialize)]
struct OaiChoice {
    message: OllamaMsgOwned,
}

#[derive(Deserialize)]
struct OaiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

#[async_trait::async_trait]
impl Provider for OpenAiCompatProvider {
    fn id(&self) -> ProviderId {
        self.id
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse> {
        let mut messages = vec![OllamaMsg { role: "system", content: &req.system }];
        messages.extend(req.messages.iter().map(|m| OllamaMsg { role: role_str(m.role), content: &m.content }));
        let body = OaiReq { model: &req.model, messages };
        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("{:?}: {e}", self.id)))?;
        if !resp.status().is_success() {
            return Err(Error::Provider(format!("{:?}: HTTP {}", self.id, resp.status())));
        }
        let parsed: OaiResp =
            resp.json().await.map_err(|e| Error::Provider(format!("{:?}: bad response: {e}", self.id)))?;
        let text = parsed.choices.into_iter().next().map(|c| c.message.content).unwrap_or_default();
        let cost = pricing::cost_usd(self.id, &req.model, parsed.usage.prompt_tokens, parsed.usage.completion_tokens);
        Ok(CompletionResponse {
            text,
            prompt_tokens: parsed.usage.prompt_tokens,
            completion_tokens: parsed.usage.completion_tokens,
            cost_usd: cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Stub {
        id: ProviderId,
        cost_usd: f64,
    }

    #[async_trait::async_trait]
    impl Provider for Stub {
        fn id(&self) -> ProviderId {
            self.id
        }
        async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse> {
            Ok(CompletionResponse {
                text: "ok".into(),
                prompt_tokens: 10,
                completion_tokens: 10,
                cost_usd: self.cost_usd,
            })
        }
    }

    fn req() -> CompletionRequest {
        CompletionRequest { model: "m".into(), system: "s".into(), messages: vec![] }
    }

    #[tokio::test]
    async fn unconfigured_provider_errors() {
        let store = Store::open(&std::env::temp_dir().join(format!("orchopork-test-{}.db", uuid::Uuid::new_v4())))
            .await
            .unwrap();
        let gw = Gateway::new(store, 40.0);
        assert!(gw.complete(ProviderId::Claude, &req()).await.is_err());
    }

    #[tokio::test]
    async fn local_provider_bypasses_the_budget_cap() {
        let store = Store::open(&std::env::temp_dir().join(format!("orchopork-test-{}.db", uuid::Uuid::new_v4())))
            .await
            .unwrap();
        sqlx::query("INSERT INTO spend_ledger (thread_id, node, cost_usd, created_at) VALUES ('t', 'n', 999.0, ?)")
            .bind(now_unix())
            .execute(&store.pool)
            .await
            .unwrap();
        let gw = Gateway::new(store, 40.0);
        gw.register(Box::new(Stub { id: ProviderId::Ollama, cost_usd: 0.0 }));
        assert!(gw.complete(ProviderId::Ollama, &req()).await.is_ok());
    }

    #[tokio::test]
    async fn cloud_provider_trips_once_spend_meets_the_cap() {
        let store = Store::open(&std::env::temp_dir().join(format!("orchopork-test-{}.db", uuid::Uuid::new_v4())))
            .await
            .unwrap();
        let gw = Gateway::new(store.clone(), 1.0);
        gw.register(Box::new(Stub { id: ProviderId::Claude, cost_usd: 1.0 }));
        assert!(gw.complete(ProviderId::Claude, &req()).await.is_ok());

        sqlx::query("INSERT INTO spend_ledger (thread_id, node, cost_usd, created_at) VALUES ('t', 'n', 1.0, ?)")
            .bind(now_unix())
            .execute(&store.pool)
            .await
            .unwrap();
        let err = gw.complete(ProviderId::Claude, &req()).await.unwrap_err();
        assert!(matches!(err, Error::Budget(_)));
    }
}
