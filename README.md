# orchopork

A lean agent orchestration engine written in Rust. It works through coding tasks on its own inside a git
repository. Cheap **local** models (Ollama, llama.cpp) do most of the work, and **cloud** models (Claude, DeepSeek,
Gemini) are used only where they pay off. A hard monthly budget caps cloud spending.

- **Isolated runs.** Each goal runs on its own branch (`orchopork/<id>`) in its own `git worktree` under
  `.orchopork/worktrees/`. Your checkout is never touched, and you can run several goals side by side.
- **Checkpointed.** Every step is a git commit plus a SQLite row that holds the full loop state. That makes pause,
  crash recovery, resume and rewind all exact.
- **Hybrid routing.** The actor model works turn by turn. After repeated failures a turn is escalated to a
  stronger model. When the budget is exhausted, cloud roles fall back to the local actor.
- **Hard budget.** Before any cloud call, its worst-case cost (prompt estimate + `max_tokens`) is reserved against
  the monthly cap under a lock. Concurrent runs cannot overshoot it, and every paid call is recorded, including
  calls whose reply was rejected.
- **Declarative skills.** Tone, tools and output checks are YAML/Markdown files in `.orchopork/skills/`, not
  hardcoded strings.

## Quick start

```sh
cargo build --release
ollama pull qwen2.5-coder:14b          # or any local model you like
cd /path/to/your/repo
/path/to/orchopork                     # opens http://localhost:7878
```

The interface is a plain chat box. When you send a message, orchopork asks
**"Activate an agent orchestra?"**:

- **No, single agent**: pick a model (local or cloud; a link leads to model setup if none is ready). One agent
  with tools works on the request like a classic coding assistant. Replying continues the same conversation.
- **Yes, orchestra**: pick a lead model. The lead drafts a team as a chain of command, and you review it on a canvas
  before anything runs:
  - each agent is a node with an auto-assigned decimal rank (private → corporal ≤10 → captain ≤100 → major ≤1000 →
    general);
  - **drag** an agent onto another to put it under that commander; any agent can command any number of agents;
  - **drag from the reserve** (coder, tester, reviewer, researcher, docs writer, refactorer) onto an agent to add a
    subordinate;
  - **double-click** an agent to edit its name, role, task definition, model and commander;
  - `Ctrl+Z` undo, `Delete` remove (its subordinates move up), `A` add a subordinate, `Enter` edit; pan by dragging
    the background, zoom with the wheel.

  **Approve & deploy** starts it. The same tree then shows live status (who is working, who reported, turns and
  cost per agent), and clicking an agent opens its feed, where you can message that agent directly.

### Juggling many tasks and changing your mind

- **Sidebar** groups conversations by what needs you: *Pinned*, *Running*, *Needs you* (finished or stopped since
  you last looked, shown bold with a dot), then *Today / This week / Older* by last activity. Search it, or pin a
  long-lived task with ☆. The tab title shows how many tasks are waiting, e.g. `(2) ▶ myrepo · orchopork`.
- **`Ctrl/⌘ K`** jumps to any conversation, recent project or action.
- **Background updates**: when a task you are not looking at finishes or stops, a toast links to it; turn on 🔔 in
  the sidebar for desktop notifications while the tab is in the background.
- **Drafts are kept**: unsent text in any composer (home, each chat, each orchestra agent) survives navigation,
  reloads and project switches. A team the lead drafted is saved as you edit it; **Later** keeps it on the home
  screen under *Pick up where you left off*.
- **Rewind from the UI**: hover a step and choose **↺ Rewind here**, or hover one of your messages and choose
  **✎ Edit & resend** to take it (and everything after it) back and send a corrected version. **↺ Start over**
  keeps the goal and resets the branch; **⧉ Retry as new** starts a separate task with the same goal (e.g. another
  model, or a team instead of one agent) and keeps the original. Rewound states stay under `refs/orchopork/rewound/`.

### Token economy: a strong lead with cheaper workers

A typical setup is one Opus lead with two Sonnet workers. In the team editor, **Workers:** sets the model of every
agent below the lead that has no model of its own (with an Opus or Fable lead, Sonnet is suggested when it is
available). The engine is built so such a team spends fewer tokens than the lead model working alone:

- **Prompt caching (Claude).** Every turn's context is split into a stable part and a volatile tail. The stable
  part is the mission and the repository as it was when the run started (identical for every agent, cached once
  for all of them), the agent's role, and its conversation so far; it is sent with cache breakpoints and re-read at
  a fraction of the input price. What changes every turn (who has reported, which files changed) goes after the
  last breakpoint. Commanders, who wait for their subordinates between turns, use the 1-hour cache.
- **A window that moves in steps.** When the conversation outgrows the context budget, old turns are dropped a
  third of the budget at a time rather than one per turn, so the cached prefix survives many turns in a row.
- **Reports carry diffs**, and teammates see each other's reports: fewer turns spent re-reading files.
- The run header, each agent's node and its panel show input tokens, the share read from cache, and output tokens.
  Cached tokens are billed at their real rates (reads 0.1x, or 0.05x on Opus 5.5; writes 1.25x, 2x for 1 hour);
  the budget guard still reserves the uncached worst case.

### Agents on this computer (ACP)

Claude Code, Gemini CLI, Codex and any other agent that speaks the
[Agent Client Protocol](https://agentclientprotocol.com) can be used anywhere a model can be used: for a single-agent
chat, as the lead of an orchestra, or as any member of one (pick it in the agent's edit dialog). orchopork starts
the agent in the run's worktree, sends it the task with the same context an internal agent would get, streams its
actions to the UI, answers its permission requests (shell commands only if `allow_commands` is on) and file requests
(only inside the worktree), then checkpoints the result. Each agent uses your own login for that tool, and its cost
is not counted against the budget.

Presets (editable in **Models & settings → Agents on this computer**):

| Agent       | Command                                        |
|-------------|------------------------------------------------|
| Claude Code | `npx -y @agentclientprotocol/claude-agent-acp` |
| Gemini CLI  | `gemini --acp`                                 |
| Codex       | `npx -y @agentclientprotocol/codex-acp`        |

**Choosing model family, version and effort.** When you pick an agent (in the model picker or an agent's edit
dialog), orchopork asks it what it offers over ACP and shows those choices, e.g. for Claude Code: *Model* (Sonnet 5,
Opus 5.5, Fable 5.1, Haiku 4.5) and *Effort* (low → max). They are applied as ACP session config options at the start
of every turn and shown on the run. API models get the same treatment: Claude models (except Haiku 4.5) take an
effort level (`output_config.effort`), and Gemini takes low/medium/high (`reasoning_effort`). From the CLI or config
use a query suffix: `acp:claude-code?model=opus&effort=high`, `claude:claude-opus-5-5?effort=xhigh`.

Sign in to each tool once in a terminal (e.g. `claude` → `/login`). The **Test** button runs the ACP handshake.
From the CLI: `orchopork init --actor acp:claude-code`.

How an orchestra executes: an agent acts only after all of its subordinates have reported, so work flows bottom-up.
Commanders read their subordinates' reports and can **`delegate`** work back to a direct subordinate with an order.
`finish` sends a report to the agent's commander, **with the diff of that agent's changes attached automatically**, so
a commander reviews from diffs instead of re-reading files. Teammates under the same commander see each other's
reports (with the list of changed files) as they come in, so later agents build on earlier work instead of
re-exploring it. When the lead finishes, the optional
verification command gates completion. All agents share the run's worktree and act one at a time. The whole command
state is checkpointed with every step, so pause, resume and rewind work exactly as they do for single runs.

From the terminal (classic plan → act → verify → review pipeline):

```sh
orchopork init --actor ollama:qwen2.5-coder:14b --planner claude:claude-sonnet-5 --critic deepseek:deepseek-chat
orchopork providers set-key claude            # reads the key from stdin
orchopork run "Add a --json flag to the list command" --verify "cargo test"
orchopork runs | show <id> | diff <id> | inject <id> "use serde" | resume <id> | rewind <id> 3 | push <id>
```

Ctrl-C during `orchopork run` pauses after the current step. Resume later with `orchopork resume <id>`.

## How a classic run works

```text
plan ──► act ⇄ tool ──finish──► verify ──pass──► review ──approve──► done
          ▲                       │fail            │revise
          └───────────────────────┴────────────────┘
```

| Phase  | Who                                   | What happens                                                                                                                                               |
|--------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| plan   | `planner` (defaults to the actor)     | One call that sees the goal, the file list and README/PROJECT_CONTEXT/STATE docs, and writes a numbered plan.                                               |
| act    | `actor`, or `escalation` when stuck   | One JSON action per turn (`read_file`, `edit_file`, `run_command`, …). The tool result is fed back. Each turn is one checkpoint.                             |
| verify | you                                   | The run's `--verify` command plus every enabled validator skill's `command` must exit 0. On failure the output goes back to the actor.                      |
| review | `critic` (optional)                   | Sees the goal, plan, summary, verification output and the full diff, and answers `approve` or `revise` with feedback.                                       |

**Intervening:** *Inject* adds a message the actor sees on its next turn. It works while running, while paused, and
on a finished run, which re-opens it for a follow-up. *Rewind* resets the worktree to any step, or to the start
(`-1`), and drops later steps. The pre-rewind tree is kept under `refs/orchopork/rewound/…`, so nothing is lost.
*Pause* stops after the step in progress.

**Safety valves** (all under `limits` in `.orchopork/config.yaml`, and editable in Settings):
`steps_per_session` (pause for a human look every N steps), `max_failures`, `escalate_after`, `max_review_rounds`,
`command_timeout_secs`, `allow_commands`, per-role context budgets and `max_output_tokens`.

When a run is done, review the branch and merge it yourself: `git merge orchopork/<id>`. You can also push it from
the dashboard.

### Actor protocol

The actor replies with a short note and one JSON object:

```json
{"tool": "edit_file", "args": {"path": "src/lib.rs", "old": "fn a()", "new": "fn b()"}}
```

This is a text protocol rather than each vendor's native tool calling, so it behaves the same on every backend,
including local models without a tool template. The parser is lenient: fences or not, `name`/`arguments` aliases,
and raw newlines inside JSON strings are all accepted.

File tools are confined to the worktree: no absolute paths, no `..` escapes, no symlinks pointing outside, no
`.git`. `run_command` is a real shell with a timeout; its whole process group is killed when the timeout hits.
Turn it off with `limits.allow_commands: false`.

## Skills

Skills are `*.yaml`, or `*.md` with YAML front matter where the body becomes `prompt_injection`. They live in
`.orchopork/skills/` and hot-reload from Settings or with `orchopork skills reload`. Conflicting skills are refused
when you enable them.

```yaml
# system_modifier: layered onto every role's system prompt, ordered by priority
name: "anti-sycophancy-terse"
version: "1.0.0"
type: "system_modifier"
description: "Enforces zero conversational fluff and direct technical execution."
priority: 10
conflicts_with: []
prompt_injection: |
  Be strictly terse, technical, and utilitarian. ...
```

```yaml
# tool_definition: new actor tools backed by your commands. Arguments arrive as
# environment variables (ORCHOPORK_ARG_<NAME>) and are never interpolated into the command.
name: "cargo-tools"
version: "1.0.0"
type: "tool_definition"
description: "Clippy and targeted tests."
tools:
  - name: clippy
    description: "Run clippy and report warnings."
    command: "cargo clippy --all-targets 2>&1 | tail -n 80"
  - name: test_filter
    description: "Run tests whose name matches a filter."
    parameters: {filter: string}
    command: 'cargo test "$ORCHOPORK_ARG_FILTER" 2>&1 | tail -n 80'
```

```yaml
# validator: substring rules on every actor reply, and/or a command gating `finish`
name: "no-placeholders"
version: "1.0.0"
type: "validator"
description: "Reject placeholder code; require formatting before finishing."
validator:
  forbidden_substrings: ["todo!()", "unimplemented!()"]
  command: "cargo fmt --check"
```

Bundled skills: `anti-sycophancy-terse` and `test-driven-loop` (enabled by default), plus `explain-like-principal`
and `markdown-memory-sync` (off by default, because every enabled modifier costs context on every call).

## Adding models

Everything goes through **＋ Add a model** (in the model picker and on *Models & settings*). Each option is tested
before it is saved:

- **Ollama**: shows what is installed and downloads new models in-app with a progress bar.
- **LM Studio, vLLM, llama.cpp**: local OpenAI-compatible servers, free and never budget-limited.
- **Claude, Gemini, DeepSeek**: paste an API key; the model list is fetched live from the provider.
- **OpenRouter, Groq, Mistral, OpenAI, Together, xAI or any other OpenAI-compatible endpoint**: base URL + key,
  with an optional price per 1M tokens (without one, hosted endpoints are budgeted conservatively). Their models
  appear as `compat:<endpoint>/<model>`.
- **Claude Code, Gemini CLI, Codex or any ACP agent**: detected automatically; the add flow tests the handshake.

## Providers and budget

| Provider   | Kind  | Configure                                                                     |
|------------|-------|-------------------------------------------------------------------------------|
| `ollama`   | local | URL (default `http://127.0.0.1:11434`), `ollama_num_ctx` (default 16384)       |
| `llamacpp` | local | `llamacpp_url`, the OpenAI root of `llama-server` (e.g. `http://127.0.0.1:8080/v1`) |
| `claude`   | cloud | API key                                                                       |
| `deepseek` | cloud | API key                                                                       |
| `gemini`   | cloud | API key (OpenAI-compatible endpoint)                                          |

Models are written `provider:model` (`ollama:qwen2.5-coder:14b`). Prices live in `src/providers/pricing.rs` and
match model ids by longest prefix. Unknown models are priced high, so the budget guard over-estimates rather than
under-estimates.

Ollama's default context window is small, and Ollama silently truncates longer prompts, which breaks agent turns.
orchopork sets `num_ctx` on every request; raise it if your hardware allows.

## Files

```text
.orchopork/
  config.yaml            settings (safe to edit by hand; every key has a default)
  secrets.yaml           API keys + GitHub token, mode 0600 (plaintext at rest)
  skills/                skill files
  skills.enabled.yaml    enabled skill names
  state.db               runs, steps, spend ledger (SQLite, WAL)
  worktrees/<run-id>/    one git worktree per run
  engine.lock            one executing process per workspace
```

`.orchopork/` is added to the repository's `.git/info/exclude` automatically.

## Access from other devices

By default only the machine running orchopork can open it. To use it from your phone, another computer or another
network, turn on **Models & settings → Access from other devices**. It applies at once, no restart needed:

- the server listens on all interfaces (same port, 7878 by default) instead of `127.0.0.1` only;
- other devices get a sign-in page and need the **access token** shown on that card. You can also copy a sign-in
  link (`http://<ip>:7878/?token=…`); it sets a cookie and removes the token from the address bar;
- from another network, forward TCP 7878 on your router to the LAN address the card shows, then open
  `http://<your public IP>:7878`.

The setting is machine-wide and lives in `~/.config/orchopork/server.yaml` (`%APPDATA%\orchopork\server.yaml` on
Windows, or `$ORCHOPORK_CONFIG_DIR`), mode 0600. **New token** signs out every other device. Anyone with the token
can run commands on this machine through the agents, and plain HTTP sends it unencrypted, so on untrusted
networks prefer a VPN (Tailscale, WireGuard) or an SSH tunnel (`ssh -L 7878:127.0.0.1:7878 you@host`), which
need no port forwarding and leave this setting off.

## Security model

- By default the server binds `127.0.0.1` only. It rejects requests whose `Host` is not local, and state-changing
  requests whose `Origin` is not local. This blocks DNS-rebinding and cross-site requests; the API can run commands
  through the agent, so no web page may reach it.
- With access from other devices on, requests from other machines need the access token (an `HttpOnly`,
  `SameSite=Strict` cookie or `Authorization: Bearer`), and state-changing ones must come from the same origin.
  Requests from this machine keep the rules above.
- API keys and the GitHub token never pass through wizard state or any API response. The only secret the API
  returns is the remote access token, on the settings card.
- Nothing is written to disk before you choose a workspace, and `git init` only happens when you ask for it.

## Development

```sh
cargo test          # unit + end-to-end engine tests (real git/SQLite, scripted LLMs)
cargo clippy --all-targets
cargo fmt
```

Requires Rust 1.89+ and `git` on `PATH`.

`src/` layout: `graph/` (engine, prompts, action protocol, tools), `providers/` (gateway, budget, pricing),
`storage/` (SQLite), `skills/`, `git/`, `server/` (HTTP API + wizard), `bootstrap.rs` (workspace wiring),
`config.rs`, `secrets.rs`. The dashboard is a single dependency-free file, `ui/index.html`, embedded in the binary.
