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
    /// Images shown to the model with this message (vision models only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Image>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), cache_split: None, images: vec![] }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), cache_split: None, images: vec![] }
    }
    pub fn with_images(mut self, images: Vec<Image>) -> Self {
        self.images = images;
        self
    }
}

/// An image for a vision model: its media type and base64 data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub media_type: String,
    pub data: String,
}

/// Largest image sent to a model.
pub const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

impl Image {
    pub fn from_file(path: &std::path::Path) -> std::result::Result<Image, String> {
        let media_type = match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "webp" => "image/webp",
            "gif" => "image/gif",
            other => return Err(format!("{} is not an image (.{other}); use png, jpeg, webp or gif", path.display())),
        };
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(format!("{} is {} MB; images must be under 5 MB", path.display(), bytes.len() / 1_048_576));
        }
        Ok(Image { media_type: media_type.into(), data: crate::fsutil::base64(&bytes) })
    }
}

/// Whether a model accepts images, judged by its name (providers that can
/// ask the server do so instead; see `Provider::supports_vision`).
pub fn vision_by_name(model: &str) -> bool {
    let m = model.to_lowercase();
    [
        "vision", "-vl", "vl-", "vl:", "llava", "bakllava", "moondream", "minicpm-v", "pixtral", "gpt-4o", "gpt-4.1",
        "gpt-5", "gemini", "claude", "gemma3", "llama4", "qwen2.5vl", "qwen2.5-vl", "qwen3-vl", "granite3.2-vision",
        "mistral-small3", "internvl", "cogvlm",
    ]
    .iter()
    .any(|k| m.contains(k))
}

/// A tool offered through the provider's native tool-calling API. The
/// model's call comes back as `Completion::tool_call` and is rendered into
/// the reply text as a `{"tool": .., "args": ..}` object, so the rest of
/// the engine only ever sees the text protocol.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
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
    /// Native tool definitions; empty for plain text calls. Endpoints that
    /// reject tools are retried without them (see `Gateway::complete`).
    pub tools: Vec<ToolDef>,
    /// Receives the reply while it is being written (providers that stream).
    pub sink: Option<Sink>,
}

/// Called with each piece of a reply as it arrives: (text, is_thinking).
#[derive(Clone)]
pub struct Sink(pub Arc<SinkFn>);

pub type SinkFn = dyn Fn(&str, bool) + Send + Sync;

impl std::fmt::Debug for Sink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Sink")
    }
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
    /// First native tool call (name, arguments), if the model made one.
    pub tool_call: Option<(String, serde_json::Value)>,
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
    /// Whether `model` accepts images.
    async fn supports_vision(&self, model: &str) -> bool {
        vision_by_name(model)
    }
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
    /// Serializes the budget check of cloud calls, so two calls cannot both
    /// pass it against the same remaining budget.
    admission: tokio::sync::Mutex<()>,
    /// Worst-case cost of cloud calls currently in flight. Released by a
    /// guard, so a call dropped mid-flight (a killed run) frees it too.
    reserved_usd: Arc<std::sync::Mutex<f64>>,
    /// `provider:model`s whose endpoint rejected native tools; they get the
    /// plain text protocol from then on.
    no_tools: std::sync::Mutex<std::collections::HashSet<String>>,
    /// Vision capability by `provider:model`, once known.
    vision: std::sync::Mutex<HashMap<String, bool>>,
}

impl Gateway {
    pub fn new(store: Store, monthly_cap_usd: f64) -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
            store,
            cap_usd: RwLock::new(monthly_cap_usd),
            admission: tokio::sync::Mutex::new(()),
            reserved_usd: Arc::new(std::sync::Mutex::new(0.0)),
            no_tools: Default::default(),
            vision: Default::default(),
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

    /// Replaces the whole provider set at once, so a call starting during a
    /// settings change never finds the set empty.
    pub fn replace_all(&self, providers: Vec<Box<dyn Provider>>) {
        let map = providers
            .into_iter()
            .map(|p| {
                let p: Arc<dyn Provider> = Arc::from(p);
                (p.id(), p)
            })
            .collect();
        *self.providers.write().unwrap() = map;
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

    /// Whether `target` takes native tool calls (false once its endpoint
    /// refused them).
    pub fn native_tools(&self, target: &ModelRef) -> bool {
        !self.no_tools.lock().unwrap().contains(&target.to_string())
    }

    /// Whether `target` accepts images. `?vision=true|false` on the model
    /// reference overrides what the provider reports.
    pub async fn supports_vision(&self, target: &ModelRef) -> bool {
        match target.option("vision").map(str::to_lowercase).as_deref() {
            Some("true" | "1" | "yes") => return true,
            Some("false" | "0" | "no") => return false,
            _ => {}
        }
        if target.provider == ProviderId::Acp {
            return false;
        }
        let key = target.to_string();
        if let Some(v) = self.vision.lock().unwrap().get(&key) {
            return *v;
        }
        let Ok(p) = self.provider(target.provider) else { return vision_by_name(&target.model) };
        let v = p.supports_vision(&target.model).await;
        self.vision.lock().unwrap().insert(key, v);
        v
    }

    pub async fn spent_this_month(&self) -> Result<f64> {
        self.store.spend_since(month_start_unix(now_unix())).await
    }

    pub async fn budget_remaining(&self) -> Result<f64> {
        let reserved = *self.reserved_usd.lock().unwrap();
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
        // Released only after the real cost is in the ledger (or when the
        // call is dropped), so there is no window in which the call counts
        // as neither.
        let _reservation = if free {
            None
        } else {
            let est = estimate_cost(rates, req);
            let _admit = self.admission.lock().await;
            let spent = self.spent_this_month().await?;
            let cap = self.cap();
            let mut reserved = self.reserved_usd.lock().unwrap();
            if spent + *reserved + est > cap {
                return Err(Error::Budget(format!(
                    "monthly cap ${cap:.2} would be exceeded (spent ${spent:.2}, in flight ${:.2}, this call up to ${est:.2})",
                    *reserved
                )));
            }
            *reserved += est;
            Some(Reservation { pool: self.reserved_usd.clone(), usd: est })
        };

        let mut tuned = req.clone();
        tuned.effort = target.option("effort").map(str::to_string);
        let key = target.to_string();
        if self.no_tools.lock().unwrap().contains(&key) {
            tuned.tools.clear();
        }
        let mut result = provider.complete(&target.model, &tuned).await;
        if let Err(Error::Provider(msg)) = &result
            && !tuned.tools.is_empty()
            && rejects_tools(msg)
        {
            tracing::info!("{key} does not accept native tools ({msg}); using the text protocol");
            self.no_tools.lock().unwrap().insert(key);
            tuned.tools.clear();
            result = provider.complete(&target.model, &tuned).await;
        }
        // `_reservation` is released when this returns: after the real cost
        // is in the ledger.
        match result {
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
                let text = match &c.tool_call {
                    Some((name, args)) => {
                        let call = json!({ "tool": name, "args": args });
                        if c.text.trim().is_empty() { call.to_string() } else { format!("{}\n\n{call}", c.text.trim_end()) }
                    }
                    None => c.text,
                };
                recorded.map(|()| CallOutcome {
                    model: target.clone(),
                    text,
                    prompt_tokens,
                    completion_tokens: c.completion_tokens,
                    cache_read_tokens: c.cache_read_tokens,
                    cache_write_tokens: c.cache_write_tokens + c.cache_write_long_tokens,
                    cost_usd: cost,
                    truncated: c.truncated,
                })
            }
            Err(e) => Err(e),
        }
    }
}

/// A cloud call's share of `Gateway::reserved_usd`, given back on drop.
struct Reservation {
    pool: Arc<std::sync::Mutex<f64>>,
    usd: f64,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut r = self.pool.lock().unwrap_or_else(|p| p.into_inner());
        *r = (*r - self.usd).max(0.0);
    }
}

/// Whether a provider error means "this endpoint/model cannot do native
/// tool calls" (Ollama models without a tool template, llama-server
/// without --jinja, proxies that drop the field, ...).
fn rejects_tools(msg: &str) -> bool {
    let m = msg.to_lowercase();
    // One malformed call from a model that does take tools.
    if m.contains("error parsing tool call") {
        return false;
    }
    (m.contains("tool") || m.contains("function"))
        && (m.contains("not support")
            || m.contains("unsupported")
            || m.contains("jinja")
            || m.contains("invalid")
            || m.contains("unknown")
            || m.contains("unrecognized")
            || m.contains("not allowed")
            || m.contains("http 400"))
}

/// OpenAI/Ollama-style `tools` array.
fn function_tools(tools: &[ToolDef]) -> serde_json::Value {
    json!(
        tools
            .iter()
            .map(|t| json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } }))
            .collect::<Vec<_>>()
    )
}

/// Arguments arrive as an object (Ollama, Claude) or a JSON string (OpenAI).
fn call_args(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::String(s) => serde_json::from_str(&s).unwrap_or_else(|_| json!({ "input": s })),
        serde_json::Value::Null => json!({}),
        other => other,
    }
}

#[derive(Deserialize)]
struct FnCall {
    #[serde(default)]
    function: FnCallBody,
}

#[derive(Deserialize, Default)]
struct FnCallBody {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

fn first_call(calls: Vec<FnCall>) -> Option<(String, serde_json::Value)> {
    calls.into_iter().find(|c| !c.function.name.is_empty()).map(|c| (c.function.name, call_args(c.function.arguments)))
}

fn estimate_cost(rates: (f64, f64), req: &CompletionRequest) -> f64 {
    // Priced as if nothing were cached (and a long cache write costs 2x),
    // so the reservation stays a true worst case.
    let bytes = req.system.len()
        + req.messages.iter().map(|m| m.content.len() + 16 + m.images.len() * 6_000).sum::<usize>()
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

/// Sends `rb` and hands every line of the streamed response to `on_line`.
/// Errors read like `send_json`'s (so e.g. tool rejections are recognised).
async fn send_lines(label: &str, rb: reqwest::RequestBuilder, mut on_line: impl FnMut(&str)) -> Result<()> {
    use tokio_stream::StreamExt;
    let resp = rb.send().await.map_err(|e| {
        let what = if e.is_connect() {
            format!("cannot connect to {}", e.url().map(|u| u.as_str()).unwrap_or("endpoint"))
        } else if e.is_timeout() {
            "request timed out".to_string()
        } else {
            e.to_string()
        };
        Error::Provider(format!("{label}: {what}"))
    })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Provider(format!("{label}: HTTP {}: {}", status.as_u16(), error_message(&body))));
    }
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Error::Provider(format!("{label}: the stream broke: {e}")))?;
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            if !line.trim().is_empty() {
                on_line(line.trim());
            }
        }
    }
    let rest = String::from_utf8_lossy(&buf);
    if !rest.trim().is_empty() {
        on_line(rest.trim());
    }
    Ok(())
}

/// Folds Ollama's streamed NDJSON chunks into one reply.
#[derive(Default)]
struct OllamaFold {
    content: String,
    thinking: String,
    tool_calls: Vec<FnCall>,
    prompt_eval_count: u64,
    eval_count: u64,
    done_reason: Option<String>,
    error: Option<String>,
}

impl OllamaFold {
    fn line(&mut self, line: &str, sink: Option<&Sink>) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { return };
        if let Some(e) = v["error"].as_str() {
            self.error = Some(e.to_string());
            return;
        }
        let m = &v["message"];
        if let Some(t) = m["thinking"].as_str().filter(|t| !t.is_empty()) {
            self.thinking.push_str(t);
            if let Some(s) = sink {
                (s.0)(t, true);
            }
        }
        if let Some(t) = m["content"].as_str().filter(|t| !t.is_empty()) {
            self.content.push_str(t);
            if let Some(s) = sink {
                (s.0)(t, false);
            }
        }
        if let Some(calls) = m.get("tool_calls").cloned()
            && let Ok(mut c) = serde_json::from_value::<Vec<FnCall>>(calls)
        {
            self.tool_calls.append(&mut c);
        }
        if v["done"] == true {
            self.prompt_eval_count = v["prompt_eval_count"].as_u64().unwrap_or(0);
            self.eval_count = v["eval_count"].as_u64().unwrap_or(0);
            self.done_reason = v["done_reason"].as_str().map(str::to_string);
        }
    }

    fn finish(self) -> Result<OllamaChatResp> {
        if let Some(e) = self.error {
            return Err(Error::Provider(format!("ollama: {e}")));
        }
        Ok(OllamaChatResp {
            message: OllamaMsg {
                tool_calls: self.tool_calls,
                content: self.content,
                thinking: (!self.thinking.is_empty()).then_some(self.thinking),
            },
            prompt_eval_count: self.prompt_eval_count,
            eval_count: self.eval_count,
            done_reason: self.done_reason,
        })
    }
}

/// Folds an OpenAI-style SSE stream (`data: {...}` chunks) into one reply,
/// joining tool-call fragments by index.
#[derive(Default)]
struct OaiFold {
    content: String,
    reasoning: String,
    calls: Vec<(String, String)>,
    finish_reason: Option<String>,
    usage: Option<serde_json::Value>,
    error: Option<String>,
}

impl OaiFold {
    fn line(&mut self, line: &str, sink: Option<&Sink>) {
        let Some(data) = line.strip_prefix("data:") else { return };
        let data = data.trim();
        if data == "[DONE]" {
            return;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else { return };
        if v.get("error").is_some() {
            self.error = Some(error_message(data));
            return;
        }
        if v["usage"].is_object() {
            self.usage = Some(v["usage"].clone());
        }
        let c = &v["choices"][0];
        if let Some(f) = c["finish_reason"].as_str() {
            self.finish_reason = Some(f.to_string());
        }
        let d = &c["delta"];
        if let Some(t) = d["reasoning_content"].as_str().or(d["reasoning"].as_str()).filter(|t| !t.is_empty()) {
            self.reasoning.push_str(t);
            if let Some(s) = sink {
                (s.0)(t, true);
            }
        }
        if let Some(t) = d["content"].as_str().filter(|t| !t.is_empty()) {
            self.content.push_str(t);
            if let Some(s) = sink {
                (s.0)(t, false);
            }
        }
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            let i = tc["index"].as_u64().unwrap_or(self.calls.len().saturating_sub(1) as u64) as usize;
            while self.calls.len() <= i {
                self.calls.push((String::new(), String::new()));
            }
            if let Some(n) = tc["function"]["name"].as_str() {
                self.calls[i].0.push_str(n);
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                self.calls[i].1.push_str(a);
            }
        }
    }

    fn finish(self, label: &str) -> Result<OaiResp> {
        if let Some(e) = self.error {
            return Err(Error::Provider(format!("{label}: {e}")));
        }
        let tool_calls = self
            .calls
            .into_iter()
            .map(|(name, args)| FnCall { function: FnCallBody { name, arguments: serde_json::Value::String(args) } })
            .collect();
        Ok(OaiResp {
            choices: vec![OaiChoice {
                message: OaiMessage {
                    tool_calls,
                    content: Some(self.content),
                    reasoning_content: (!self.reasoning.is_empty()).then_some(self.reasoning),
                },
                finish_reason: self.finish_reason,
            }],
            usage: self.usage.and_then(|u| serde_json::from_value(u).ok()),
        })
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
    wire_messages_as(system, messages, tail, false)
}

/// `wire_messages`; images go as OpenAI `image_url` parts, or with
/// `ollama` as Ollama's per-message `images` list.
fn wire_messages_as(system: Option<&str>, messages: &[Message], tail: Option<&str>, ollama: bool) -> Vec<serde_json::Value> {
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
            if m.images.is_empty() {
                json!({ "role": m.role.as_str(), "content": content })
            } else if ollama {
                json!({ "role": m.role.as_str(), "content": content, "images": m.images.iter().map(|im| &im.data).collect::<Vec<_>>() })
            } else {
                let mut parts = vec![json!({ "type": "text", "text": content })];
                for im in &m.images {
                    parts.push(json!({ "type": "image_url", "image_url": { "url": format!("data:{};base64,{}", im.media_type, im.data) } }));
                }
                json!({ "role": m.role.as_str(), "content": parts })
            }
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
            for im in &m.images {
                blocks.push(json!({ "type": "image", "source": { "type": "base64", "media_type": im.media_type, "data": im.data } }));
            }
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
    if !req.tools.is_empty() {
        body["tools"] = json!(
            req.tools
                .iter()
                .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.parameters }))
                .collect::<Vec<_>>()
        );
        // Every turn of the protocol is exactly one tool call.
        body["tool_choice"] = json!({ "type": "any", "disable_parallel_tool_use": true });
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

/// The model output Ollama quotes when it fails to parse a tool call
/// (`error parsing tool call: raw='...', err=...`).
fn unparsed_tool_call(msg: &str) -> Option<String> {
    let rest = &msg[msg.find("error parsing tool call")?..];
    let start = rest.find("raw='")? + "raw='".len();
    let end = rest.rfind("', err=")?;
    let raw = rest.get(start..end)?.trim();
    (!raw.is_empty()).then(|| raw.to_string())
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
    tool_calls: Vec<FnCall>,
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: Option<String>,
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
            "messages": wire_messages_as(Some(&req.system), &req.messages, req.tail.as_deref(), true),
            "stream": false,
            "options": { "num_ctx": self.num_ctx, "num_predict": req.max_tokens },
        });
        let mut body = body;
        if !req.tools.is_empty() {
            body["tools"] = function_tools(&req.tools);
        }
        let result: Result<OllamaChatResp> = match &req.sink {
            Some(sink) => {
                body["stream"] = json!(true);
                let rb = self.client.post(format!("{}/api/chat", self.base_url)).json(&body);
                let mut fold = OllamaFold::default();
                match send_lines("ollama", rb, |l| fold.line(l, Some(sink))).await {
                    Ok(()) => fold.finish(),
                    Err(e) => Err(self.hint(e)),
                }
            }
            None => {
                let rb = self.client.post(format!("{}/api/chat", self.base_url)).json(&body);
                send_json("ollama", rb, 0).await.map_err(|e| self.hint(e))
            }
        };
        let mut r = match result {
            Ok(r) => r,
            // Ollama could not parse the model's tool call and dropped the
            // reply; the call it quotes goes to the lenient text parser.
            Err(Error::Provider(msg)) if unparsed_tool_call(&msg).is_some() => {
                return Ok(Completion { text: unparsed_tool_call(&msg).unwrap_or_default(), ..Default::default() });
            }
            Err(e) => return Err(e),
        };
        let tool_call = first_call(std::mem::take(&mut r.message.tool_calls));
        let text = if r.message.content.trim().is_empty() {
            r.message.thinking.unwrap_or_default()
        } else if let Some(ref t) = r.message.thinking {
            if !r.message.content.contains("\"tool\"")
                && !r.message.content.contains("'tool'")
                && (t.contains("\"tool\"") || t.contains("'tool'"))
            {
                t.clone()
            } else {
                r.message.content
            }
        } else {
            r.message.content
        };
        Ok(Completion {
            text,
            prompt_tokens: r.prompt_eval_count,
            completion_tokens: r.eval_count,
            truncated: r.done_reason.as_deref() == Some("length"),
            tool_call,
            ..Default::default()
        })
    }

    async fn supports_vision(&self, model: &str) -> bool {
        let rb = self.client.post(format!("{}/api/show", self.base_url)).json(&json!({ "model": model })).timeout(Duration::from_secs(10));
        match send_json::<serde_json::Value>("ollama", rb, 0).await {
            Ok(v) => match v["capabilities"].as_array() {
                Some(caps) => caps.iter().any(|c| c == "vision"),
                None => vision_by_name(model),
            },
            Err(_) => vision_by_name(model),
        }
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
    #[serde(default)]
    name: String,
    #[serde(default)]
    input: serde_json::Value,
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

    async fn supports_vision(&self, _model: &str) -> bool {
        true
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
        // Only text and tool_use blocks: thinking blocks are the model's
        // private reasoning.
        let tool_call =
            r.content.iter().find(|b| b.kind == "tool_use").map(|b| (b.name.clone(), call_args(b.input.clone())));
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
            tool_call,
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let list: ModelList = send_json("claude", self.get("https://api.anthropic.com/v1/models?limit=100"), 1).await?;
        Ok(list.data.into_iter().map(|m| m.id).collect())
    }
}

// ---- OpenAI-compatible: DeepSeek, Gemini, llama.cpp server -----------------

#[derive(Clone)]
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
    tool_calls: Vec<FnCall>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
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

    async fn supports_vision(&self, model: &str) -> bool {
        match self.id {
            ProviderId::Gemini => true,
            ProviderId::DeepSeek => false,
            _ => vision_by_name(model),
        }
    }

    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion> {
        let mut body = json!({
            "model": model,
            "messages": wire_messages(Some(&req.system), &req.messages, req.tail.as_deref()),
            "max_tokens": req.max_tokens.min(self.max_tokens_cap),
        });
        if !req.tools.is_empty() {
            body["tools"] = function_tools(&req.tools);
        }
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
        let r: OaiResp = match &req.sink {
            Some(sink) => {
                let mut body = body.clone();
                body["stream"] = json!(true);
                if !self.id.is_local() {
                    body["stream_options"] = json!({ "include_usage": true });
                }
                let rb = self.auth(self.client.post(format!("{}/chat/completions", self.base_url)).json(&body));
                let mut fold = OaiFold::default();
                send_lines(self.id.as_str(), rb, |l| fold.line(l, Some(sink))).await?;
                fold.finish(self.id.as_str())?
            }
            None => send_json(self.id.as_str(), rb, retries).await?,
        };
        let mut choice = r
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Provider(format!("{}: response had no choices", self.id.as_str())))?;
        let tool_call = first_call(std::mem::take(&mut choice.message.tool_calls));
        let (prompt, completion, cached) = match r.usage {
            Some(u) => {
                let cached = u.prompt_tokens_details.map_or(0, |d| d.cached_tokens).max(u.prompt_cache_hit_tokens);
                (u.prompt_tokens, u.completion_tokens, cached.min(u.prompt_tokens))
            }
            None => (0, 0, 0),
        };
        let text = match choice.message.content {
            Some(ref c) if !c.trim().is_empty() => {
                if let Some(ref r) = choice.message.reasoning_content {
                    if !c.contains("\"tool\"")
                        && !c.contains("'tool'")
                        && (r.contains("\"tool\"") || r.contains("'tool'"))
                    {
                        r.clone()
                    } else {
                        c.clone()
                    }
                } else {
                    c.clone()
                }
            }
            _ => choice.message.reasoning_content.unwrap_or_default(),
        };
        Ok(Completion {
            text,
            prompt_tokens: prompt - cached,
            completion_tokens: completion,
            cache_read_tokens: cached,
            truncated: choice.finish_reason.as_deref() == Some("length"),
            tool_call,
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

    async fn supports_vision(&self, model: &str) -> bool {
        match self.route(model) {
            Ok((e, name)) => e.client.supports_vision(name).await,
            Err(_) => vision_by_name(model),
        }
    }

    async fn complete(&self, model: &str, req: &CompletionRequest) -> Result<Completion> {
        let (e, name) = self.route(model)?;
        e.client.complete(name, req).await.map_err(|err| match err {
            Error::Provider(m) => Error::Provider(m.replacen("compat", &e.id, 1)),
            other => other,
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        // All endpoints at once, so one unreachable endpoint costs its
        // timeout once instead of once per endpoint before it.
        let mut set = tokio::task::JoinSet::new();
        for (i, e) in self.endpoints.iter().enumerate() {
            let (id, client) = (e.id.clone(), e.client.clone());
            set.spawn(async move {
                let ms = tokio::time::timeout(Duration::from_secs(4), client.list_models()).await;
                (i, id, ms.ok().and_then(|r| r.ok()).unwrap_or_default())
            });
        }
        let mut parts = Vec::new();
        while let Some(Ok(p)) = set.join_next().await {
            parts.push(p);
        }
        parts.sort_by_key(|(i, ..)| *i);
        Ok(parts.into_iter().flat_map(|(_, id, ms)| ms.into_iter().map(move |m| format!("{id}/{m}"))).collect())
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
    fn ollama_tool_call_parse_errors_keep_the_raw_call() {
        let msg = r#"ollama: error parsing tool call: raw='{"path":"a.js","content":"import x from 'three';\n"},"tool":"write_file"}', err=invalid character ',' after top-level value"#;
        assert_eq!(
            unparsed_tool_call(msg).as_deref(),
            Some(r#"{"path":"a.js","content":"import x from 'three';\n"},"tool":"write_file"}"#)
        );
        assert!(!rejects_tools(msg), "one bad call does not mean the model lacks tools");
        assert!(rejects_tools("ollama: HTTP 400: registry.ollama.ai/library/x does not support tools"));
        assert_eq!(unparsed_tool_call("ollama: model 'x' not found"), None);
    }

    #[test]
    fn openai_style_requests_append_the_tail_to_the_last_message() {
        // Streams fold into the same reply as a non-streamed call.
        let got = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let g2 = got.clone();
        let sink = Sink(std::sync::Arc::new(move |t: &str, _| g2.lock().unwrap().push_str(t)));
        let mut o = OllamaFold::default();
        for l in [
            r#"{"message":{"role":"assistant","content":"Hel"},"done":false}"#,
            r#"{"message":{"role":"assistant","content":"lo","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a"}}}]},"done":false}"#,
            r#"{"message":{"content":""},"done":true,"prompt_eval_count":12,"eval_count":3,"done_reason":"stop"}"#,
        ] {
            o.line(l, Some(&sink));
        }
        let r = o.finish().unwrap();
        assert_eq!((r.message.content.as_str(), r.prompt_eval_count, r.eval_count), ("Hello", 12, 3));
        assert_eq!(first_call(r.message.tool_calls).unwrap().0, "read_file");
        assert_eq!(got.lock().unwrap().as_str(), "Hello");
        let mut f = OaiFold::default();
        for l in [
            r#"data: {"choices":[{"delta":{"content":"Hi "}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"write_file","arguments":"{\"pa"}}]}}]}"#,
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"b\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":5}}"#,
            "data: [DONE]",
        ] {
            f.line(l, None);
        }
        let mut r = f.finish("x").unwrap();
        let c = r.choices.remove(0);
        assert_eq!(c.message.content.as_deref(), Some("Hi "));
        let (name, args) = first_call(c.message.tool_calls).unwrap();
        assert_eq!((name.as_str(), args["path"].as_str()), ("write_file", Some("b")));
        assert_eq!(r.usage.unwrap().prompt_tokens, 20);
        let mut e = OllamaFold::default();
        e.line(r#"{"error":"model 'x' not found"}"#, None);
        assert!(e.finish().err().unwrap().to_string().contains("not found"));

        let img = Image { media_type: "image/png".into(), data: "QUJD".into() };
        let with = [Message::user("look").with_images(vec![img.clone()])];
        let oai = wire_messages(None, &with, None);
        assert_eq!(oai[0]["content"][1]["image_url"]["url"], "data:image/png;base64,QUJD");
        let oll = wire_messages_as(None, &with, None, true);
        assert_eq!((oll[0]["content"].as_str(), oll[0]["images"][0].as_str()), (Some("look"), Some("QUJD")));
        let cl = claude_body("m", &CompletionRequest { messages: with.to_vec(), ..Default::default() });
        assert_eq!(cl["messages"][0]["content"][0]["type"], "image");
        assert_eq!(cl["messages"][0]["content"][0]["source"]["data"], "QUJD");
        assert!(vision_by_name("qwen2.5vl:7b") && vision_by_name("llava:13b") && !vision_by_name("qwen2.5-coder:7b"));
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

    /// A killed run drops its call mid-flight: the reservation must come back.
    #[tokio::test]
    async fn a_dropped_call_releases_its_reservation() {
        let (_d, store) = store().await;
        let gw = Arc::new(Gateway::new(store, 1.0));
        gw.register(Box::new(Stub { id: ProviderId::Claude, delay_ms: 5_000 }));
        let call = {
            let gw = gw.clone();
            tokio::spawn(async move { gw.complete(&haiku(), &req(99_000), "r").await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(gw.budget_remaining().await.unwrap() < 0.6, "the call's worst case is reserved while it runs");
        call.abort();
        let _ = call.await;
        assert!((gw.budget_remaining().await.unwrap() - 1.0).abs() < 1e-9, "released after the abort");
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
