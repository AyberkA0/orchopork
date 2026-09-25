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
    fn unknown_model_is_priced_high_not_free() {
        assert_eq!(rates(ProviderId::Claude, "some-future-model"), fallback(ProviderId::Claude));
    }
}
