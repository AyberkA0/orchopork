//! Static $/1M-token pricing. Ollama is always free (local compute only).
//!
//! Prices are USD per 1,000,000 tokens, (prompt, completion). Unknown
//! model/provider pairs fall back to a conservative default rather than
//! erroring, so the circuit breaker never mis-fires open on a new model id.

use super::ProviderId;

const DEFAULT_RATE: (f64, f64) = (5.0, 15.0);

const RATES: &[(ProviderId, &str, f64, f64)] = &[
    (ProviderId::Claude, "claude-haiku-4-5", 1.0, 5.0),
    (ProviderId::Claude, "claude-sonnet-5", 3.0, 15.0),
    (ProviderId::Claude, "claude-opus-5-5", 15.0, 75.0),
    (ProviderId::DeepSeek, "deepseek-chat", 0.28, 0.42),
    (ProviderId::DeepSeek, "deepseek-reasoner", 0.55, 2.19),
    (ProviderId::Gemini, "gemini-2.5-flash", 0.30, 2.50),
    (ProviderId::Gemini, "gemini-2.5-pro", 1.25, 10.0),
];

pub fn cost_usd(provider: ProviderId, model: &str, prompt_tokens: u64, completion_tokens: u64) -> f64 {
    if provider == ProviderId::Ollama {
        return 0.0;
    }
    let (inp, out) = RATES
        .iter()
        .find(|(p, m, ..)| *p == provider && *m == model)
        .map(|(_, _, i, o)| (*i, *o))
        .unwrap_or(DEFAULT_RATE);
    (prompt_tokens as f64 / 1_000_000.0) * inp + (completion_tokens as f64 / 1_000_000.0) * out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_is_free_regardless_of_tokens() {
        assert_eq!(cost_usd(ProviderId::Ollama, "llama3", 1_000_000, 1_000_000), 0.0);
    }

    #[test]
    fn known_model_uses_its_own_rate() {
        let c = cost_usd(ProviderId::Claude, "claude-haiku-4-5", 1_000_000, 1_000_000);
        assert!((c - 6.0).abs() < 1e-9);
    }

    #[test]
    fn unknown_model_falls_back_to_default_rate() {
        let c = cost_usd(ProviderId::Claude, "some-future-model", 1_000_000, 0);
        assert!((c - DEFAULT_RATE.0).abs() < 1e-9);
    }
}
