//! Static USD-per-1M-token pricing. Local providers are always free.
//!
//! Models match by longest prefix, so dated or suffixed ids
//! (`deepseek-chat-v3`, `gemini-2.5-flash-lite`) resolve to their family.
//! Unknown models fall back to a deliberately high per-provider rate: the
//! budget guard should over-estimate a new model, never under-estimate it.

use super::ProviderId;

const RATES: &[(ProviderId, &str, f64, f64)] = &[
    (ProviderId::Claude, "claude-haiku-4-5", 1.0, 5.0),
    (ProviderId::Claude, "claude-sonnet-5", 2.0, 10.0),
    (ProviderId::Claude, "claude-sonnet-4", 3.0, 15.0),
    (ProviderId::Claude, "claude-opus-5-5", 4.0, 20.0),
    (ProviderId::Claude, "claude-opus-5", 5.0, 25.0),
    (ProviderId::Claude, "claude-opus-4", 5.0, 25.0),
    (ProviderId::Claude, "claude-fable-5-1", 10.0, 50.0),
    (ProviderId::Claude, "claude-fable-5", 10.0, 50.0),
    (ProviderId::DeepSeek, "deepseek-chat", 0.28, 0.42),
    (ProviderId::DeepSeek, "deepseek-reasoner", 0.55, 2.19),
    (ProviderId::Gemini, "gemini-2.5-flash", 0.30, 2.50),
    (ProviderId::Gemini, "gemini-2.5-pro", 1.25, 10.0),
];

fn fallback(provider: ProviderId) -> (f64, f64) {
    match provider {
        ProviderId::Claude => (10.0, 50.0),
        ProviderId::DeepSeek => (0.6, 2.5),
        ProviderId::Gemini => (2.5, 15.0),
        ProviderId::Ollama | ProviderId::LlamaCpp | ProviderId::Acp => (0.0, 0.0),
        // Unknown hosted endpoint: priced high unless you set its price.
        ProviderId::Compat => (5.0, 15.0),
    }
}

/// (input, output) USD per 1M tokens.
pub fn rates(provider: ProviderId, model: &str) -> (f64, f64) {
    if provider.is_local() {
        return (0.0, 0.0);
    }
    RATES
        .iter()
        .filter(|(p, m, ..)| *p == provider && model.starts_with(m))
        .max_by_key(|(_, m, ..)| m.len())
        .map(|(_, _, i, o)| (*i, *o))
        .unwrap_or_else(|| fallback(provider))
}

pub fn cost_usd(provider: ProviderId, model: &str, prompt_tokens: u64, completion_tokens: u64) -> f64 {
    cost_from(rates(provider, model), prompt_tokens, completion_tokens)
}

pub fn cost_from((inp, out): (f64, f64), prompt_tokens: u64, completion_tokens: u64) -> f64 {
    (prompt_tokens as f64 / 1_000_000.0) * inp + (completion_tokens as f64 / 1_000_000.0) * out
}

/// Price of a cache read as a fraction of the base input price. Claude
/// writes cost 1.25x (5-minute entries) or 2x (1-hour entries). Providers
/// whose cache discount is not known here are billed as if uncached, so the
/// budget never under-counts.
fn cache_read_factor(provider: ProviderId, model: &str) -> f64 {
    match provider {
        ProviderId::Claude if model.starts_with("claude-fable-5-1") => 0.025,
        ProviderId::Claude if model.starts_with("claude-opus-5-5") => 0.05,
        ProviderId::Claude => 0.1,
        ProviderId::DeepSeek => 0.1,
        _ => 1.0,
    }
}

/// Cost of a completion, pricing cached input at the provider's rates.
pub fn cost_with_cache(rates: (f64, f64), provider: ProviderId, model: &str, c: &super::Completion) -> f64 {
    let (inp, _) = rates;
    let per = |t: u64, factor: f64| (t as f64 / 1_000_000.0) * inp * factor;
    cost_from(rates, c.prompt_tokens, c.completion_tokens)
        + per(c.cache_read_tokens, cache_read_factor(provider, model))
        + per(c.cache_write_tokens, 1.25)
        + per(c.cache_write_long_tokens, 2.0)
}

/// Reasoning-effort levels a model accepts (empty: no effort knob).
pub fn effort_levels(provider: ProviderId, model: &str) -> &'static [&'static str] {
    match provider {
        ProviderId::Claude if model.starts_with("claude-haiku") => &[],
        ProviderId::Claude => &["low", "medium", "high", "xhigh", "max"],
        ProviderId::Gemini => &["low", "medium", "high"],
        _ => &[],
    }
}

/// Models offered in the UI for a cloud provider (anything else can still
/// be typed in by hand).
pub fn suggested_models(provider: ProviderId) -> Vec<&'static str> {
    RATES.iter().filter(|(p, ..)| *p == provider).map(|(_, m, ..)| *m).filter(|m| !m.ends_with("-4")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_is_free_regardless_of_tokens() {
        assert_eq!(cost_usd(ProviderId::Ollama, "llama3", 1_000_000, 1_000_000), 0.0);
        assert_eq!(cost_usd(ProviderId::LlamaCpp, "x", 1_000_000, 1_000_000), 0.0);
    }

    #[test]
    fn longest_prefix_wins() {
        assert_eq!(rates(ProviderId::Claude, "claude-opus-5-5"), (4.0, 20.0));
        assert_eq!(rates(ProviderId::Claude, "claude-opus-5"), (5.0, 25.0));
        assert_eq!(rates(ProviderId::Gemini, "gemini-2.5-flash-lite"), (0.30, 2.50));
        let c = cost_usd(ProviderId::Claude, "claude-haiku-4-5", 1_000_000, 1_000_000);
        assert!((c - 6.0).abs() < 1e-9);
    }

    #[test]
    fn cached_input_is_priced_by_kind() {
        let c = super::super::Completion {
            prompt_tokens: 1_000_000,
            completion_tokens: 0,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
            cache_write_long_tokens: 1_000_000,
            ..Default::default()
        };
        let r = rates(ProviderId::Claude, "claude-sonnet-5");
        // 2.0 uncached + 0.2 read + 2.5 write(5m) + 4.0 write(1h)
        assert!((cost_with_cache(r, ProviderId::Claude, "claude-sonnet-5", &c) - 8.7).abs() < 1e-9);
        // Unknown cache discount: billed like normal input.
        let r = rates(ProviderId::Compat, "x");
        let plain = super::super::Completion { prompt_tokens: 0, cache_read_tokens: 1_000_000, ..Default::default() };
        assert!((cost_with_cache(r, ProviderId::Compat, "x", &plain) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn unknown_model_is_priced_high_not_free() {
        assert_eq!(rates(ProviderId::Claude, "some-future-model"), fallback(ProviderId::Claude));
    }
}
