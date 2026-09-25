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
    /// Any OpenAI-compatible endpoint you add (OpenRouter, Groq, Mistral,
    /// LM Studio, vLLM…); `model` is `<endpoint id>/<model>`.
    #[serde(rename = "compat")]
    Compat,
}

impl ProviderId {
    pub const ALL: [ProviderId; 6] = [
        ProviderId::Ollama,
        ProviderId::LlamaCpp,
        ProviderId::Claude,
        ProviderId::DeepSeek,
        ProviderId::Gemini,
        ProviderId::Compat,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ProviderId::Ollama => "ollama",
            ProviderId::LlamaCpp => "llamacpp",
            ProviderId::Claude => "claude",
            ProviderId::DeepSeek => "deepseek",
            ProviderId::Gemini => "gemini",
            ProviderId::Acp => "acp",
            ProviderId::Compat => "compat",
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
    /// Prompt-caching hint: the first `cache_split` bytes of `content` are a
    /// prefix other requests share too (e.g. every agent of an orchestra
    /// sees the same mission and file overview), so providers with explicit
    /// caching mark it as its own cache entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_split: Option<usize>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), cache_split: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), cache_split: None }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompletionRequest {
    pub system: String,
    /// Must start with a user message and end with one (current models
    /// reject assistant prefill). Everything here is expected to be
    /// byte-stable from one turn to the next, so it can be cached.
    pub messages: Vec<Message>,
    /// Context that changes every turn (who has reported, which files
    /// changed). Sent after the last cache breakpoint, so it never
    /// invalidates the cached history in front of it.
    pub tail: Option<String>,
    /// Cache for an hour instead of five minutes: for callers whose next
    /// turn is likely more than five minutes away (commanders waiting on
    /// their subordinates).
    pub cache_long: bool,
    pub max_tokens: u32,
    /// Reasoning effort (`low`…`max`); set by the gateway from the target's
    /// `effort` option. Providers without such a knob ignore it.
    pub effort: Option<String>,
}

/// What a provider returns.
#[derive(Debug, Clone, Default)]
pub struct Completion {
    pub text: String,
    /// Input tokens billed at the full rate (not read from or written to
    /// a cache).
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Input tokens served from the provider's prompt cache.
    pub cache_read_tokens: u64,
    /// Input tokens written to a 5-minute / 1-hour cache entry.
    pub cache_write_tokens: u64,
    pub cache_write_long_tokens: u64,
    /// The reply hit `max_tokens` and is cut off.
    pub truncated: bool,
}

/// What the gateway returns: the completion plus its metered cost.
#[derive(Debug, Clone, Serialize)]
pub struct CallOutcome {
    pub model: ModelRef,
    pub text: String,
    /// All input tokens, cached or not.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
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
    /// Provider-specific (input, output) USD per 1M tokens, overriding the
    /// static table; `Some((0.0, 0.0))` marks a free (local) model.
    fn rates(&self, _model: &str) -> Option<(f64, f64)> {
        None
    }
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
        let rates = provider.rates(&target.model).unwrap_or_else(|| pricing::rates(target.provider, &target.model));
        let free = target.provider.is_local() || rates == (0.0, 0.0);
        let reservation = if free {
            0.0
        } else {
            let est = estimate_cost(rates, req);
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

        let mut tuned = req.clone();
        tuned.effort = target.option("effort").map(str::to_string);
        let result = provider.complete(&target.model, &tuned).await;
        let outcome = match result {
            Ok(c) => {
                let cost = pricing::cost_with_cache(rates, target.provider, &target.model, &c);
                let prompt_tokens =
                    c.prompt_tokens + c.cache_read_tokens + c.cache_write_tokens + c.cache_write_long_tokens;
                let recorded = self
                    .store
                    .record_spend(
                        run_id,
                        target.provider.as_str(),
                        &target.model,
                        prompt_tokens,
                        c.completion_tokens,
                        cost,
                    )
                    .await;
                recorded.map(|()| CallOutcome {
                    model: target.clone(),
                    text: c.text,
                    prompt_tokens,
                    completion_tokens: c.completion_tokens,
                    cache_read_tokens: c.cache_read_tokens,
                    cache_write_tokens: c.cache_write_tokens + c.cache_write_long_tokens,
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

fn estimate_cost(rates: (f64, f64), req: &CompletionRequest) -> f64 {
    // Priced as if nothing were cached (and a long cache write costs 2x),
    // so the reservation stays a true worst case.
    let bytes = req.system.len()
        + req.messages.iter().map(|m| m.content.len() + 16).sum::<usize>()
        + req.tail.as_ref().map_or(0, String::len);
    let bytes = if req.cache_long { bytes * 2 } else { bytes * 5 / 4 };
    let prompt_tokens = (bytes as u64) * 2 / 5 + 64;
    pricing::cost_from(rates, prompt_tokens, u64::from(req.max_tokens))
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

/// OpenAI-style messages. The volatile `tail` is appended to the last
/// message, after the stable history, so providers with automatic prefix
/// caching (OpenAI, DeepSeek, …) still reuse everything before it.
fn wire_messages(system: Option<&str>, messages: &[Message], tail: Option<&str>) -> Vec<serde_json::Value> {
    let last = messages.len().saturating_sub(1);
    system
        .filter(|s| !s.is_empty())
        .map(|s| json!({ "role": "system", "content": s }))
        .into_iter()
        .chain(messages.iter().enumerate().map(|(i, m)| {
            let content = match tail.filter(|t| i == last && !t.trim().is_empty()) {
                Some(t) => format!("{}\n\n{t}", m.content),
                None => m.content.clone(),
            };
            json!({ "role": m.role.as_str(), "content": content })
        }))
        .collect()
}

/// Claude messages with explicit cache breakpoints: after the system prompt,
/// after each message's shared prefix (`cache_split`), and at the end of the
/// stable history. The volatile `tail` goes after the last breakpoint as its
/// own block, so a turn only pays full price for what is new since the last
/// one. At most 4 breakpoints are allowed; this uses at most 3 (only the
/// first message carries a `cache_split`).
fn claude_body(model: &str, req: &CompletionRequest) -> serde_json::Value {
    let cc = if req.cache_long { json!({ "type": "ephemeral", "ttl": "1h" }) } else { json!({ "type": "ephemeral" }) };
    let text = |t: &str, cached: bool| {
        let mut b = json!({ "type": "text", "text": t });
        if cached {
            b["cache_control"] = cc.clone();
        }
        b
    };
    let mut splits = 0;
    let last = req.messages.len().saturating_sub(1);
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let mut blocks = Vec::new();
            let split = m.cache_split.filter(|&n| n > 0 && n < m.content.len() && m.content.is_char_boundary(n));
            match split {
                Some(n) if splits == 0 && !m.content[..n].trim().is_empty() && !m.content[n..].trim().is_empty() => {
                    splits += 1;
                    blocks.push(text(&m.content[..n], true));
                    blocks.push(text(&m.content[n..], false));
                }
                _ => blocks.push(text(&m.content, false)),
            }
            if i == last {
                if let Some(b) = blocks.last_mut() {
                    b["cache_control"] = cc.clone();
                }
                if let Some(t) = req.tail.as_deref().filter(|t| !t.trim().is_empty()) {
                    blocks.push(text(t, false));
                }
            }
            json!({ "role": m.role.as_str(), "content": blocks })
        })
        .collect();
    let mut body = json!({
        "model": model,
        "max_tokens": req.max_tokens,
        "messages": messages,
    });
    if !req.system.trim().is_empty() {
        body["system"] = json!([text(&req.system, true)]);
    }
    body
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
            "messages": wire_messages(Some(&req.system), &req.messages, req.tail.as_deref()),
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
            ..Default::default()
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
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_creation: Option<ClaudeCacheCreation>,
}

#[derive(Deserialize)]
struct ClaudeCacheCreation {
    #[serde(default)]
    ephemeral_5m_input_tokens: u64,
    #[serde(default)]
    ephemeral_1h_input_tokens: u64,
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
        let mut body = claude_body(model, req);
        // Haiku 4.5 rejects `effort`; every newer model accepts it.
        if let Some(e) = req.effort.as_deref().filter(|_| !model.starts_with("claude-haiku")) {
            body["output_config"] = json!({ "effort": e });
        }
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
        let u = &r.usage;
        // The TTL breakdown is authoritative when present; without it all
        // writes are 5-minute ones unless this request asked for 1 hour.
        let (w5, w1h) = match &u.cache_creation {
            Some(c) if c.ephemeral_5m_input_tokens + c.ephemeral_1h_input_tokens > 0 => {
                (c.ephemeral_5m_input_tokens, c.ephemeral_1h_input_tokens)
            }
            _ if req.cache_long => (0, u.cache_creation_input_tokens),
            _ => (u.cache_creation_input_tokens, 0),
        };
        Ok(Completion {
            text,
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            cache_read_tokens: u.cache_read_input_tokens,
            cache_write_tokens: w5,
            cache_write_long_tokens: w1h,
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

    /// A user-added OpenAI-compatible endpoint (served through `CompatRouter`).
    pub fn endpoint(base_url: &str, api_key: Option<String>, local: bool) -> Self {
        Self::new(ProviderId::Compat, base_url, api_key, u32::MAX, if local { 1800 } else { 600 })
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
    /// OpenAI-style automatic prefix caching.
    #[serde(default)]
    prompt_tokens_details: Option<OaiPromptDetails>,
    /// DeepSeek's name for the same thing.
    #[serde(default)]
    prompt_cache_hit_tokens: u64,
}

#[derive(Deserialize)]
struct OaiPromptDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[async_trait::async_trait]
impl Provider for OpenAiCompatProvider {
    fn id(&self) -> ProviderId {
        self.id
    }

    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion> {
        let mut body = json!({
            "model": model,
            "messages": wire_messages(Some(&req.system), &req.messages, req.tail.as_deref()),
            "max_tokens": req.max_tokens.min(self.max_tokens_cap),
        });
        // Gemini's OpenAI endpoint takes low/medium/high.
        if let (ProviderId::Gemini, Some(e)) = (self.id, req.effort.as_deref()) {
            let e = match e {
                "xhigh" | "max" => "high",
                "minimal" => "low",
                other => other,
            };
            body["reasoning_effort"] = json!(e);
        }
        let rb = self.auth(self.client.post(format!("{}/chat/completions", self.base_url)).json(&body));
        let retries =
            if self.id.is_local() || self.base_url.contains("://127.0.0.1") || self.base_url.contains("://localhost") {
                0
            } else {
                3
            };
        let r: OaiResp = send_json(self.id.as_str(), rb, retries).await?;
        let choice = r
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Provider(format!("{}: response had no choices", self.id.as_str())))?;
        let (prompt, completion, cached) = match r.usage {
            Some(u) => {
                let cached = u.prompt_tokens_details.map_or(0, |d| d.cached_tokens).max(u.prompt_cache_hit_tokens);
                (u.prompt_tokens, u.completion_tokens, cached.min(u.prompt_tokens))
            }
            None => (0, 0, 0),
        };
        Ok(Completion {
            text: choice.message.content.unwrap_or_default(),
            prompt_tokens: prompt - cached,
            completion_tokens: completion,
            cache_read_tokens: cached,
            truncated: choice.finish_reason.as_deref() == Some("length"),
            ..Default::default()
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

// ---- user-added OpenAI-compatible endpoints --------------------------------

/// One configured endpoint, as the router needs it.
pub struct CompatEndpoint {
    pub id: String,
    pub client: OpenAiCompatProvider,
    pub local: bool,
    pub price: Option<(f64, f64)>,
}

/// Dispatches `compat` models (`<endpoint>/<model>`) to their endpoint.
pub struct CompatRouter {
    endpoints: Vec<CompatEndpoint>,
}

impl CompatRouter {
    pub fn new(endpoints: Vec<CompatEndpoint>) -> Self {
        Self { endpoints }
    }

    fn route<'a>(&'a self, model: &'a str) -> Result<(&'a CompatEndpoint, &'a str)> {
        let (ep, name) = model
            .split_once('/')
            .ok_or_else(|| Error::InvalidRequest(format!("compat model {model:?} must be <endpoint>/<model>")))?;
        let e = self
            .endpoints
            .iter()
            .find(|e| e.id == ep)
            .ok_or_else(|| Error::Provider(format!("endpoint {ep:?} is not configured")))?;
        Ok((e, name))
    }
}

#[async_trait::async_trait]
impl Provider for CompatRouter {
    fn id(&self) -> ProviderId {
        ProviderId::Compat
    }

    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion> {
        let (e, name) = self.route(model)?;
        e.client.complete(name, req).await.map_err(|err| match err {
            Error::Provider(m) => Error::Provider(m.replacen("compat", &e.id, 1)),
            other => other,
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let mut all = Vec::new();
        for e in &self.endpoints {
            if let Ok(Ok(ms)) = tokio::time::timeout(Duration::from_secs(4), e.client.list_models()).await {
                all.extend(ms.into_iter().map(|m| format!("{}/{m}", e.id)));
            }
        }
        Ok(all)
    }

    fn rates(&self, model: &str) -> Option<(f64, f64)> {
        let (e, _) = self.route(model).ok()?;
        if e.local { Some((0.0, 0.0)) } else { e.price }
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
            Ok(Completion { text: "ok".into(), prompt_tokens: 1000, completion_tokens: 1000, ..Default::default() })
        }
        async fn list_models(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }

    #[test]
    fn claude_requests_cache_the_stable_prefix_and_keep_the_tail_after_it() {
        let req = CompletionRequest {
            system: "system prompt".into(),
            messages: vec![
                Message { cache_split: Some("SHARED ".len()), ..Message::user("SHARED own part") },
                Message::assistant("a call"),
                Message::user("its result"),
            ],
            tail: Some("status now".into()),
            max_tokens: 100,
            ..Default::default()
        };
        let body = claude_body("claude-sonnet-5", &req);
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        let first = &body["messages"][0]["content"];
        assert_eq!(first[0]["text"], "SHARED ");
        assert!(first[0]["cache_control"].is_object(), "the shared prefix is its own cache entry");
        assert_eq!(first[1]["text"], "own part");
        let last = body["messages"][2]["content"].as_array().unwrap();
        assert_eq!(last.len(), 2);
        assert!(last[0]["cache_control"].is_object(), "breakpoint at the end of the stable history");
        assert_eq!(last[1]["text"], "status now");
        assert!(last[1].get("cache_control").is_none(), "the volatile tail is never cached");
        let marks = body.to_string().matches("cache_control").count();
        assert!(marks <= 4, "at most 4 breakpoints, got {marks}");

        let long = claude_body("claude-sonnet-5", &CompletionRequest { cache_long: true, ..req });
        assert_eq!(long["system"][0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn openai_style_requests_append_the_tail_to_the_last_message() {
        let m = wire_messages(Some("s"), &[Message::user("history")], Some("tail"));
        assert_eq!(m[1]["content"], "history\n\ntail");
    }

    fn req(max_tokens: u32) -> CompletionRequest {
        CompletionRequest { system: "s".into(), messages: vec![Message::user("hi")], max_tokens, ..Default::default() }
    }

    async fn store() -> (tempfile::TempDir, Store) {
        let d = tempfile::tempdir().unwrap();
        let s = Store::open(&d.path().join("s.db")).await.unwrap();
        (d, s)
    }

    fn haiku() -> ModelRef {
        ModelRef::new(ProviderId::Claude, "claude-haiku-4-5")
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
        let local = ModelRef::new(ProviderId::Ollama, "m");
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

    #[tokio::test]
    async fn compat_models_route_by_endpoint_and_local_ones_are_free() {
        let (_d, store) = store().await;
        let router = CompatRouter::new(vec![
            CompatEndpoint {
                id: "lm".into(),
                client: OpenAiCompatProvider::endpoint("http://127.0.0.1:9", None, true),
                local: true,
                price: None,
            },
            CompatEndpoint {
                id: "or".into(),
                client: OpenAiCompatProvider::endpoint("http://127.0.0.1:9", None, false),
                local: false,
                price: Some((1.0, 2.0)),
            },
        ]);
        assert_eq!(router.rates("lm/x"), Some((0.0, 0.0)));
        assert_eq!(router.rates("or/x"), Some((1.0, 2.0)));
        assert!(matches!(router.complete("nope/x", &req(10)).await, Err(Error::Provider(_))));
        assert!(matches!(router.complete("no-slash", &req(10)).await, Err(Error::InvalidRequest(_))));
        // A local endpoint is never budget-gated, even with the cap at zero.
        let gw = Gateway::new(store, 0.0);
        gw.register(Box::new(router));
        let err = gw.complete(&ModelRef::new(ProviderId::Compat, "lm/x"), &req(10), "r").await.unwrap_err();
        assert!(matches!(err, Error::Provider(_)), "fails on the network, not the budget: {err}");
        let err = gw.complete(&ModelRef::new(ProviderId::Compat, "or/x"), &req(10), "r").await.unwrap_err();
        assert!(matches!(err, Error::Budget(_)));
    }

    #[test]
    fn provider_error_messages_are_extracted() {
        assert_eq!(error_message(r#"{"type":"error","error":{"type":"x","message":"bad model"}}"#), "bad model");
        assert_eq!(error_message(r#"{"error":"model 'x' not found"}"#), "model 'x' not found");
        assert_eq!(error_message("plain"), "plain");
    }
}
