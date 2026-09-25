//! Binary entry point: an HTTP server (`serve`, the default) plus CLI
//! subcommands (`run`, `skills`, `providers`) that drive the exact same
//! `bootstrap::open` wiring the server and any library embedder use. No
//! command here can do something the HTTP API or an in-process `Executor`
//! cannot, and vice versa — all three sit on top of `orchopork::bootstrap`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use orchopork::graph::{NodeKind, StepOutcome};
use orchopork::providers::ProviderId;
use orchopork::storage::{RewindTarget, Snapshot};
use orchopork::{bootstrap, server};

#[derive(Parser)]
#[command(name = "orchopork", version, about = "Lean hybrid local/cloud LLM agent orchestration engine")]
struct Cli {
    /// Workspace root; `.orchopork/` lives under it. Valid anywhere on the
    /// command line, before or after the subcommand.
    #[arg(long, global = true, default_value = ".")]
    workspace: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the local HTTP server and dashboard (the default).
    Serve,
    /// Drive the execution graph directly: step, inject, history, rewind.
    Run {
        #[command(subcommand)]
        action: RunAction,
    },
    /// Inspect or reload the skill registry.
    Skills {
        #[command(subcommand)]
        action: SkillsAction,
    },
    /// Inspect the monthly budget or configure a provider's API key.
    Providers {
        #[command(subcommand)]
        action: ProvidersAction,
    },
}

#[derive(Subcommand)]
enum RunAction {
    /// Run one node (planner|coder|critic|test_runner|tool_executor).
    Step {
        #[arg(long)]
        thread: String,
        #[arg(long)]
        node: String,
        #[arg(long)]
        provider: String,
        #[arg(long)]
        model: String,
    },
    /// Append a user message with no LLM call and no cost.
    Inject {
        #[arg(long)]
        thread: String,
        #[arg(long)]
        text: String,
    },
    /// Print a thread's checkpoint history.
    History {
        #[arg(long)]
        thread: String,
    },
    /// Roll back to a snapshot id, or N steps back from the current head.
    Rewind {
        #[arg(long)]
        thread: String,
        #[arg(long)]
        snapshot: Option<String>,
        #[arg(long)]
        steps: Option<u32>,
    },
}

#[derive(Subcommand)]
enum SkillsAction {
    List,
    Reload,
}

#[derive(Subcommand)]
enum ProvidersAction {
    Budget,
    /// Store a provider's API key (0600 under .orchopork/) and register it immediately.
    SetKey {
        #[arg(long, value_parser = parse_provider)]
        provider: ProviderId,
        #[arg(long)]
        key: String,
    },
}

fn parse_node(s: &str) -> Result<NodeKind, String> {
    Ok(match s {
        "planner" => NodeKind::Planner,
        "coder" => NodeKind::Coder,
        "critic" => NodeKind::Critic,
        "test_runner" => NodeKind::TestRunner,
        "tool_executor" => NodeKind::ToolExecutor,
        _ => return Err(format!("unknown node {s:?} (planner|coder|critic|test_runner|tool_executor)")),
    })
}

fn parse_provider(s: &str) -> Result<ProviderId, String> {
    Ok(match s {
        "ollama" => ProviderId::Ollama,
        "claude" => ProviderId::Claude,
        "deepseek" => ProviderId::DeepSeek,
        "gemini" => ProviderId::Gemini,
        _ => return Err(format!("unknown provider {s:?} (ollama|claude|deepseek|gemini)")),
    })
}

fn print_json<T: serde::Serialize>(v: &T) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=debug".into()),
        )
        .init();

    let cli = Cli::parse();
    let workspace = cli.workspace;
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => serve(workspace).await,
        Command::Run { action } => run_action(&workspace, action).await,
        Command::Skills { action } => skills_action(&workspace, action).await,
        Command::Providers { action } => providers_action(&workspace, action).await,
    }
}

async fn serve(workspace: PathBuf) -> anyhow::Result<()> {
    let ws = bootstrap::open(&workspace, &bootstrap::Config::default()).await?;
    let state = std::sync::Arc::new(server::AppState::from_workspace(ws));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:7878").await?;
    tracing::info!("orchopork on http://{}", listener.local_addr()?);
    axum::serve(listener, server::router(state)).await?;
    Ok(())
}

async fn run_action(workspace: &std::path::Path, action: RunAction) -> anyhow::Result<()> {
    let ws = bootstrap::open(workspace, &bootstrap::Config::default()).await?;
    match action {
        RunAction::Step { thread, node, provider, model } => {
            let node = parse_node(&node).map_err(anyhow::Error::msg)?;
            let provider = parse_provider(&provider).map_err(anyhow::Error::msg)?;
            let outcome: StepOutcome = ws.executor.step(&thread, node, provider, &model).await?;
            print_json(&outcome)?;
        }
        RunAction::Inject { thread, text } => {
            let snap: Snapshot = ws.executor.inject(&thread, &text).await?;
            print_json(&snap)?;
        }
        RunAction::History { thread } => {
            let history: Vec<Snapshot> = ws.executor.history(&thread).await?;
            print_json(&history)?;
        }
        RunAction::Rewind { thread, snapshot, steps } => {
            let target = match (snapshot, steps) {
                (Some(id), None) => RewindTarget::Snapshot(id),
                (None, Some(n)) => RewindTarget::Steps(n),
                _ => anyhow::bail!("pass exactly one of --snapshot or --steps"),
            };
            let snap: Snapshot = ws.executor.rewind(&thread, target).await?;
            print_json(&snap)?;
        }
    }
    Ok(())
}

async fn skills_action(workspace: &std::path::Path, action: SkillsAction) -> anyhow::Result<()> {
    let ws = bootstrap::open(workspace, &bootstrap::Config::default()).await?;
    match action {
        SkillsAction::List => print_json(&ws.skills.list())?,
        SkillsAction::Reload => {
            let errors = ws.skills.reload()?;
            print_json(&errors.into_iter().map(|(p, e)| format!("{}: {e}", p.display())).collect::<Vec<_>>())?;
        }
    }
    Ok(())
}

async fn providers_action(workspace: &std::path::Path, action: ProvidersAction) -> anyhow::Result<()> {
    let ws = bootstrap::open(workspace, &bootstrap::Config::default()).await?;
    match action {
        ProvidersAction::Budget => {
            let spent = ws.gateway.spent_this_month().await?;
            let remaining = ws.gateway.budget_remaining().await?;
            print_json(&serde_json::json!({
                "spent_this_month_usd": spent,
                "remaining_usd": remaining,
                "configured_providers": ws.gateway.configured(),
            }))?;
        }
        ProvidersAction::SetKey { provider, key } => {
            ws.secrets.set(provider, key.clone())?;
            bootstrap::register_cloud(&ws.gateway, provider, key);
            print_json(&serde_json::json!({ "configured_providers": ws.gateway.configured() }))?;
        }
    }
    Ok(())
}
