//! Known agents: how to launch each one over ACP or as a headless CLI,
//! and how to install and sign in to the tool behind it.
//!
//! Detection scans for each preset's `cli` binary, so anything already
//! installed on this machine can be added with one click.

use std::time::Duration;

use super::{Dialect, ExternalAgent, spawn};
use crate::error::{Error, Result};

/// How to get a preset agent working: the CLI it wraps, the fixed
/// command that installs it, and how to sign in once.
#[derive(Debug, Clone, Copy)]
pub struct Setup {
    pub cli: &'static str,
    /// Empty when there is no one-line installer orchopork may run.
    pub install: &'static [&'static str],
    pub login: &'static str,
    /// Env var that works instead of an interactive login.
    pub key_env: &'static str,
}

const CLAUDE: Setup = Setup {
    cli: "claude",
    install: &["npm", "install", "-g", "@anthropic-ai/claude-code"],
    login: "claude",
    key_env: "ANTHROPIC_API_KEY",
};
const GEMINI: Setup = Setup {
    cli: "gemini",
    install: &["npm", "install", "-g", "@google/gemini-cli"],
    login: "gemini",
    key_env: "GEMINI_API_KEY",
};
const CODEX: Setup = Setup {
    cli: "codex",
    install: &["npm", "install", "-g", "@openai/codex"],
    login: "codex login",
    key_env: "OPENAI_API_KEY",
};
const QWEN: Setup = Setup {
    cli: "qwen",
    install: &["npm", "install", "-g", "@qwen-code/qwen-code"],
    login: "qwen",
    key_env: "OPENAI_API_KEY",
};
const COPILOT: Setup = Setup {
    cli: "copilot",
    install: &["npm", "install", "-g", "@github/copilot"],
    login: "copilot",
    key_env: "GH_TOKEN",
};
const OPENCODE: Setup = Setup {
    cli: "opencode",
    install: &["npm", "install", "-g", "opencode-ai"],
    login: "opencode auth login",
    key_env: "",
};
const CURSOR: Setup =
    Setup { cli: "cursor-agent", install: &[], login: "cursor-agent login", key_env: "CURSOR_API_KEY" };
const GOOSE: Setup = Setup { cli: "goose", install: &[], login: "goose configure", key_env: "" };
const AIDER: Setup = Setup { cli: "aider", install: &[], login: "aider", key_env: "" };

/// One launch recipe for a known agent.
#[derive(Debug, Clone)]
pub struct Preset {
    pub agent: ExternalAgent,
    pub setup: Setup,
    /// One line on what this way of connecting gives you.
    pub blurb: &'static str,
}

/// Every agent orchopork knows how to launch. Several tools appear twice:
/// over ACP (orchopork answers permissions and file access, confined to
/// the worktree) and as a headless CLI (the tool runs as in your terminal,
/// with your settings, MCP servers and plugins, and reports tokens).
pub fn catalog() -> Vec<Preset> {
    let p = |agent: ExternalAgent, setup: Setup, blurb: &'static str| Preset { agent, setup, blurb };
    let acp = ExternalAgent::new;
    let cli = |id, name, command, args: &[&str], d| ExternalAgent::new(id, name, command, args).cli(d);
    vec![
        p(
            cli("claude-code-cli", "Claude Code (CLI)", "claude", &[], Dialect::Claude),
            CLAUDE,
            "Your installed `claude`, headless: your login, settings, MCP servers, plugins and CLAUDE.md",
        ),
        p(
            acp("claude-code", "Claude Code", "npx", &["-y", "@agentclientprotocol/claude-agent-acp"]),
            CLAUDE,
            "Over ACP: file access confined to the worktree",
        ),
        p(
            cli("codex-cli", "Codex (CLI)", "codex", &[], Dialect::Codex),
            CODEX,
            "Your installed `codex exec`, in its own workspace sandbox",
        ),
        p(
            acp("codex", "Codex", "npx", &["-y", "@agentclientprotocol/codex-acp"]),
            CODEX,
            "Over ACP through the Codex adapter",
        ),
        p(
            cli("gemini-cli-headless", "Gemini CLI (headless)", "gemini", &[], Dialect::Gemini),
            GEMINI,
            "Your installed `gemini`, non-interactive",
        ),
        p(acp("gemini-cli", "Gemini CLI", "gemini", &["--acp"]), GEMINI, "Over ACP, built into Gemini CLI"),
        p(
            cli("qwen-code-cli", "Qwen Code (CLI)", "qwen", &[], Dialect::Qwen),
            QWEN,
            "Your installed `qwen`, non-interactive",
        ),
        p(acp("qwen-code", "Qwen Code", "qwen", &["--acp"]), QWEN, "Over ACP, built into Qwen Code"),
        p(
            cli("cursor-cli", "Cursor Agent (CLI)", "cursor-agent", &["{prompt}"], Dialect::Cursor),
            CURSOR,
            "Your installed `cursor-agent -p`",
        ),
        p(acp("cursor", "Cursor Agent", "cursor-agent", &["acp"]), CURSOR, "Over ACP, built into Cursor's CLI"),
        p(
            cli("copilot-cli", "GitHub Copilot (CLI)", "copilot", &["{prompt}"], Dialect::Copilot),
            COPILOT,
            "Your installed `copilot -p`",
        ),
        p(
            acp("copilot", "GitHub Copilot", "copilot", &["--acp", "--stdio"]),
            COPILOT,
            "Over ACP, built into Copilot CLI",
        ),
        p(acp("opencode", "OpenCode", "opencode", &["acp"]), OPENCODE, "Over ACP, built into OpenCode"),
        p(acp("goose", "Goose", "goose", &["acp"]), GOOSE, "Over ACP, built into Goose"),
        p(
            cli("aider", "Aider", "aider", &["--message-file", "{prompt_file}"], Dialect::Aider),
            AIDER,
            "Your installed `aider`, one message per turn, no auto-commits",
        ),
    ]
}

/// Agents a fresh workspace starts with.
pub fn default_agents() -> Vec<ExternalAgent> {
    let wanted = ["claude-code-cli", "claude-code", "gemini-cli", "codex"];
    let all = catalog();
    wanted.iter().filter_map(|id| all.iter().find(|p| p.agent.id == *id).map(|p| p.agent.clone())).collect()
}

/// Setup for a configured agent: by preset id, else by the tool its
/// command launches (so a renamed or custom `claude` agent still gets
/// Claude Code's install and login hints).
pub fn setup_for(agent: &ExternalAgent) -> Option<Setup> {
    let all = catalog();
    all.iter()
        .find(|p| p.agent.id == agent.id)
        .or_else(|| all.iter().find(|p| p.setup.cli == agent.command))
        .map(|p| p.setup)
}

/// Runs a preset's fixed install command (never user-supplied text).
pub async fn install(id: &str) -> Result<String> {
    let setup = catalog()
        .into_iter()
        .find(|p| p.agent.id == id)
        .map(|p| p.setup)
        .filter(|s| !s.install.is_empty())
        .ok_or_else(|| Error::InvalidRequest(format!("no installer for {id}")))?;
    let npm = ExternalAgent::new(id, id, setup.install[0], &[]);
    let args: Vec<String> = setup.install[1..].iter().map(|s| s.to_string()).collect();
    let child = spawn(&npm, &std::env::temp_dir(), &args, false)
        .map_err(|_| Error::Provider("npm was not found: install Node.js from nodejs.org first".into()))?;
    let out = tokio::time::timeout(Duration::from_secs(600), child.wait_with_output())
        .await
        .map_err(|_| Error::Provider("install took longer than 10 minutes".into()))??;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() {
        let tail: String =
            text.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
        return Err(Error::Provider(format!("install failed:\n{tail}")));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique_and_defaults_exist() {
        let all = catalog();
        let mut ids: Vec<&str> = all.iter().map(|p| p.agent.id.as_str()).collect();
        ids.sort();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "duplicate preset id");
        assert_eq!(default_agents().len(), 4);
        let custom = ExternalAgent::new("mine", "Mine", "claude", &[]);
        assert_eq!(setup_for(&custom).map(|s| s.cli), Some("claude"));
    }
}
