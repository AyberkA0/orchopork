//! Binary entry point: the local server + dashboard (`serve`, the default)
//! and a CLI over the same `Workspace`/`Engine` the server uses.

use std::future::IntoFuture;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser, Subcommand};
use orchopork::bootstrap::Workspace;
use orchopork::config::ModelRef;
use orchopork::graph::{Engine, Event};
use orchopork::providers::ProviderId;
use orchopork::server;
use orchopork::storage::{Run, RunStatus, Step, StepKind};

#[derive(Parser)]
#[command(name = "orchopork", version, about = "Lean hybrid local/cloud LLM agent orchestration engine")]
struct Cli {
    /// Workspace root (a git repository); `.orchopork/` lives under it.
    #[arg(long, global = true, default_value = ".")]
    workspace: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Start the local dashboard (the default).
    Serve {
        #[arg(long, default_value_t = 7878)]
        port: u16,
    },
    /// Set up a workspace without the web wizard.
    Init {
        /// Run `git init` if the workspace is not a repository yet.
        #[arg(long)]
        git_init: bool,
        /// Model doing the work, as provider:model (e.g. ollama:qwen2.5-coder:14b).
        #[arg(long, value_parser = parse_model)]
        actor: ModelRef,
        #[arg(long, value_parser = parse_model)]
        planner: Option<ModelRef>,
        #[arg(long, value_parser = parse_model)]
        critic: Option<ModelRef>,
        #[arg(long, value_parser = parse_model)]
        escalation: Option<ModelRef>,
        #[arg(long)]
        ollama_url: Option<String>,
        #[arg(long)]
        llamacpp_url: Option<String>,
        /// Monthly cloud budget in USD.
        #[arg(long)]
        cap: Option<f64>,
    },
    /// Start a run and follow it until it finishes or pauses (Ctrl-C pauses).
    Run {
        goal: String,
        /// Command that must succeed before the run may finish (e.g. "cargo test").
        #[arg(long)]
        verify: Option<String>,
    },
    /// List runs.
    Runs,
    /// Show a run's steps.
    Show { run: String },
    /// Resume a paused run and follow it.
    Resume { run: String },
    /// Send the agent a message (picked up on its next turn).
    Inject { run: String, text: String },
    /// Reset a run to just after step SEQ (-1 = back to the start).
    Rewind {
        run: String,
        #[arg(allow_hyphen_values = true)]
        seq: i64,
    },
    /// Show everything a run changed.
    Diff { run: String },
    /// Push a run's branch to origin.
    Push { run: String },
    /// Delete a run with its worktree and branch.
    Delete { run: String },
    /// Manage skills.
    Skills {
        #[command(subcommand)]
        action: SkillsAction,
    },
    /// Manage providers.
    Providers {
        #[command(subcommand)]
        action: ProvidersAction,
    },
    /// Show this month's spend against the cap.
    Budget,
}

#[derive(Subcommand)]
enum SkillsAction {
    List,
    Reload,
    Enable { name: String },
    Disable { name: String },
}

#[derive(Subcommand)]
enum ProvidersAction {
    /// Which providers are configured.
    List,
    /// Store an API key, read from stdin (so it stays out of shell history).
    SetKey {
        #[arg(value_parser = parse_provider)]
        provider: ProviderId,
    },
    /// List a provider's models (also checks the key / endpoint).
    Models {
        #[arg(value_parser = parse_provider)]
        provider: ProviderId,
    },
}

fn parse_model(s: &str) -> Result<ModelRef, String> {
    ModelRef::parse(s).map_err(|e| e.to_string())
}

fn parse_provider(s: &str) -> Result<ProviderId, String> {
    ProviderId::parse(s).map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let serving = matches!(cli.command, None | Some(Command::Serve { .. }));
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| if serving { "info,tower_http=warn".into() } else { "warn".into() }),
        )
        .with_writer(std::io::stderr)
        .init();

    let root = cli.workspace;
    match cli.command.unwrap_or(Command::Serve { port: 7878 }) {
        Command::Serve { port } => serve(root, port).await,
        Command::Init { git_init, actor, planner, critic, escalation, ollama_url, llamacpp_url, cap } => {
            let ws = Workspace::open(&root).await?;
            if git_init {
                ws.repo.init().await?;
            } else if !ws.repo.is_repo().await {
                anyhow::bail!("{} is not a git repository; pass --git-init to create one", ws.root.display());
            }
            let cfg = ws.update_config(|c| {
                c.onboarded = true;
                c.routing.actor = Some(actor);
                c.routing.planner = planner;
                c.routing.critic = critic;
                c.routing.escalation = escalation;
                if let Some(u) = ollama_url {
                    c.ollama_url = u;
                }
                if llamacpp_url.is_some() {
                    c.llamacpp_url = llamacpp_url;
                }
                if let Some(cap) = cap {
                    c.monthly_cap_usd = cap;
                }
            })?;
            println!("initialized {}\n{}", ws.root.display(), serde_yaml::to_string(&cfg)?);
            let missing: Vec<_> =
                [&cfg.routing.planner, &cfg.routing.actor, &cfg.routing.critic, &cfg.routing.escalation]
                    .into_iter()
                    .flatten()
                    .filter(|m| !m.provider.is_local() && ws.secrets.provider_key(m.provider).is_none())
                    .map(|m| m.provider.as_str())
                    .collect();
            if !missing.is_empty() {
                println!(
                    "add API keys with: orchopork providers set-key <provider>   (missing: {})",
                    missing.join(", ")
                );
            }
            Ok(())
        }
        Command::Run { goal, verify } => {
            let engine = engine(&root).await?;
            let mut rx = engine.subscribe();
            let run = engine.create_run(&goal, verify.as_deref(), Default::default()).await?;
            println!("run {} on branch {} ({})", run.id, run.branch, run.worktree);
            follow(&engine, &run.id, &mut rx).await
        }
        Command::Resume { run } => {
            let engine = engine(&root).await?;
            let mut rx = engine.subscribe();
            engine.start(&run).await?;
            follow(&engine, &run, &mut rx).await
        }
        Command::Runs => {
            let ws = Workspace::open(&root).await?;
            let runs = ws.store.list_runs().await?;
            if runs.is_empty() {
                println!("no runs yet; start one with: orchopork run \"<goal>\"");
            }
            for r in runs {
                println!(
                    "{}  {:<8} {:>4} steps  ${:<7.4}  {}",
                    r.id,
                    r.status.as_str(),
                    r.step_count,
                    r.cost_usd,
                    first_line(&r.goal, 70)
                );
            }
            Ok(())
        }
        Command::Show { run } => {
            let ws = Workspace::open(&root).await?;
            let r = ws.store.get_run(&run).await?;
            print_run_header(&r);
            for s in ws.store.steps(&run).await? {
                print_step(&s, true);
            }
            Ok(())
        }
        Command::Inject { run, text } => {
            let step = engine(&root).await?.inject(&run, &text, None).await?;
            println!("queued as step {}; resume the run if it is not executing", step.seq);
            Ok(())
        }
        Command::Rewind { run, seq } => {
            let r = engine(&root).await?.rewind(&run, seq).await?;
            println!("run {} rewound; {} steps kept", r.id, r.step_count);
            Ok(())
        }
        Command::Diff { run } => {
            let ws = Workspace::open(&root).await?;
            let r = ws.store.get_run(&run).await?;
            let (stat, patch) = orchopork::git::GitRepo::new(&r.worktree).diff_from(&r.base_commit).await?;
            println!("{stat}\n\n{patch}");
            Ok(())
        }
        Command::Push { run } => {
            println!("{}", engine(&root).await?.push(&run).await?);
            Ok(())
        }
        Command::Delete { run } => {
            engine(&root).await?.delete(&run).await?;
            println!("deleted run {run}");
            Ok(())
        }
        Command::Skills { action } => {
            let ws = Workspace::open(&root).await?;
            match action {
                SkillsAction::List => {}
                SkillsAction::Reload => {
                    for (p, e) in ws.skills.reload()? {
                        eprintln!("{}: {e}", p.display());
                    }
                }
                SkillsAction::Enable { name } => ws.skills.set_enabled(&name, true)?,
                SkillsAction::Disable { name } => ws.skills.set_enabled(&name, false)?,
            }
            for s in ws.skills.list() {
                println!(
                    "[{}] {:<28} {:<16} {}",
                    if s.enabled { "x" } else { " " },
                    s.skill.name,
                    format!("{:?}", s.skill.kind),
                    s.skill.description
                );
            }
            Ok(())
        }
        Command::Providers { action } => {
            let ws = Workspace::open(&root).await?;
            match action {
                ProvidersAction::List => {
                    let configured = ws.gateway.configured();
                    for p in ProviderId::ALL {
                        let state = if configured.contains(&p) { "configured" } else { "-" };
                        println!("{:<9} {:<6} {state}", p.as_str(), if p.is_local() { "local" } else { "cloud" });
                    }
                }
                ProvidersAction::SetKey { provider } => {
                    if std::io::stdin().is_terminal() {
                        eprint!("paste the {} API key and press Enter: ", provider.as_str());
                    }
                    let mut key = String::new();
                    if std::io::stdin().is_terminal() {
                        std::io::stdin().read_line(&mut key)?;
                    } else {
                        std::io::stdin().read_to_string(&mut key)?;
                    }
                    ws.set_provider_key(provider, key.trim())?;
                    println!("{} key {}", provider.as_str(), if key.trim().is_empty() { "removed" } else { "stored" });
                }
                ProvidersAction::Models { provider } => {
                    for m in ws.gateway.provider(provider)?.list_models().await? {
                        println!("{m}");
                    }
                }
            }
            Ok(())
        }
        Command::Budget => {
            let ws = Workspace::open(&root).await?;
            println!(
                "spent ${:.4} of ${:.2} this month (${:.4} left)",
                ws.gateway.spent_this_month().await?,
                ws.gateway.cap(),
                ws.gateway.budget_remaining().await?
            );
            Ok(())
        }
    }
}

async fn serve(root: PathBuf, port: u16) -> anyhow::Result<()> {
    let root = orchopork::fsutil::strip_verbatim(std::path::absolute(&root)?);
    let state = server::AppState::new(root).await?;
    let app = server::router(state.clone()).into_make_service_with_connect_info::<std::net::SocketAddr>();
    let mut remote = state.remote.subscribe();
    let mut first = true;
    // Rebinds whenever "access from other devices" is switched. Dropping the
    // old `serve` closes only its listener; open connections keep running.
    loop {
        let wanted = *remote.borrow_and_update();
        let (listener, error) = if wanted {
            match bind(std::net::Ipv4Addr::UNSPECIFIED, port).await {
                Ok(l) => (l, None),
                Err(e) => {
                    tracing::error!("could not listen on all interfaces, port {port}: {e}");
                    (bind(std::net::Ipv4Addr::LOCALHOST, port).await?, Some(e.to_string()))
                }
            }
        } else {
            (bind(std::net::Ipv4Addr::LOCALHOST, port).await?, None)
        };
        state.remote.set_listening(listener.local_addr()?, error);
        if first {
            println!("orchopork is running at http://localhost:{port}");
            first = false;
        }
        if listener.local_addr()?.ip().is_unspecified() {
            println!("access from other devices is on: http://<this machine's IP>:{port} (access token required)");
        } else if !wanted {
            tracing::info!("listening on this machine only");
        }
        let serve = axum::serve(listener, app.clone()).with_graceful_shutdown(shutdown_signal());
        tokio::select! {
            r = serve.into_future() => return Ok(r?),
            changed = remote.changed() => {
                if changed.is_err() {
                    std::future::pending::<()>().await;
                }
            }
        }
    }
}

/// Retries briefly: the previous listener on this port was just dropped.
async fn bind(ip: std::net::Ipv4Addr, port: u16) -> std::io::Result<tokio::net::TcpListener> {
    let mut tries = 0;
    loop {
        match tokio::net::TcpListener::bind((ip, port)).await {
            Err(e) if tries < 10 && e.kind() == std::io::ErrorKind::AddrInUse => {
                tries += 1;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            r => return r,
        }
    }
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn engine(root: &Path) -> anyhow::Result<Engine> {
    let ws = Workspace::open(root).await?;
    if !ws.config().onboarded {
        anyhow::bail!("workspace is not set up; run `orchopork init --actor <provider:model>` or use the dashboard");
    }
    Ok(Engine::new(ws).await?)
}

/// Prints events for one run until its loop stops. The first Ctrl-C asks
/// for a pause after the current step; a second one exits immediately.
async fn follow(engine: &Engine, run_id: &str, rx: &mut tokio::sync::broadcast::Receiver<Event>) -> anyhow::Result<()> {
    let mut paused = false;
    let mut tick = tokio::time::interval(Duration::from_millis(500));
    loop {
        tokio::select! {
            ev = rx.recv() => match ev {
                Ok(Event::Step { step }) if step.run_id == run_id => print_step(&step, false),
                Ok(Event::Log { run_id: r, message }) if r == run_id => println!("   · {message}"),
                Ok(Event::Run { run }) if run.id == run_id && run.status != RunStatus::Running && !engine.is_active(run_id) => {
                    print_final(&run);
                    return Ok(());
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
                _ => {}
            },
            _ = tokio::signal::ctrl_c() => {
                if paused {
                    anyhow::bail!("interrupted; the run will show as interrupted and can be resumed");
                }
                paused = true;
                eprintln!("\npausing after the current step (Ctrl-C again to quit now)...");
                engine.pause(run_id).await?;
            }
            _ = tick.tick() => {
                if !engine.is_active(run_id) {
                    let (run, _) = engine.run_detail(run_id).await?;
                    print_final(&run);
                    return Ok(());
                }
            }
        }
    }
}

fn print_final(run: &Run) {
    match (&run.status, &run.error) {
        (RunStatus::Done, _) => println!(
            "\n✔ done in {} steps, ${:.4}. Review with `orchopork diff {}`; the work is on branch {}.",
            run.step_count, run.cost_usd, run.id, run.branch
        ),
        (_, Some(e)) => println!("\n⏸ paused: {e}\n  resume with `orchopork resume {}`", run.id),
        _ => println!("\n⏸ paused. Resume with `orchopork resume {}`", run.id),
    }
}

fn print_run_header(r: &Run) {
    println!(
        "run {}  [{}]  ${:.4}\ngoal: {}\nbranch: {}\nworktree: {}",
        r.id,
        r.status.as_str(),
        r.cost_usd,
        r.goal,
        r.branch,
        r.worktree
    );
    if let Some(e) = &r.error {
        println!("note: {e}");
    }
    println!();
}

fn print_step(s: &Step, full: bool) {
    let model = s.meta["model"].as_str().map(|m| format!(" {m}")).unwrap_or_default();
    let cost = if s.cost_usd > 0.0 { format!(" ${:.4}", s.cost_usd) } else { String::new() };
    let head = format!("#{:<3} {:<6}{model}{cost}", s.seq, s.kind.as_str());
    match s.kind {
        StepKind::Act => {
            let tool = s.meta["tool"].as_str().unwrap_or("?");
            let ok = if s.meta["ok"].as_bool().unwrap_or(false) { "ok" } else { "FAILED" };
            println!("{head}  {tool} → {ok}");
            let note = s.output.split('{').next().unwrap_or("").trim();
            if !note.is_empty() {
                println!("      {}", first_line(note, 110));
            }
            if (full || !s.meta["ok"].as_bool().unwrap_or(false))
                && let Some(o) = &s.observation
            {
                println!("      {}", indent(&orchopork::fsutil::clip(o, if full { 4000 } else { 400 })));
            }
        }
        StepKind::Plan | StepKind::Review | StepKind::Inject => {
            let verdict = s.meta["verdict"].as_str().map(|v| format!(" ({v})")).unwrap_or_default();
            println!(
                "{head}{verdict}\n      {}",
                indent(&orchopork::fsutil::clip(&s.output, if full { 8000 } else { 1500 }))
            );
        }
        StepKind::Verify => {
            println!("{head}\n      {}", indent(&s.output));
            if full && let Some(o) = &s.observation {
                println!("      {}", indent(&orchopork::fsutil::clip(o, 4000)));
            }
        }
    }
}

fn indent(s: &str) -> String {
    s.trim_end().replace('\n', "\n      ")
}

fn first_line(s: &str, max: usize) -> String {
    let line = s.lines().next().unwrap_or("");
    if line.chars().count() > max {
        format!("{}…", line.chars().take(max).collect::<String>())
    } else {
        line.to_string()
    }
}
