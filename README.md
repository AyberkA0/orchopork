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

How an orchestra executes: an agent acts only after all of its subordinates have reported, so work flows bottom-up.
Commanders read their subordinates' reports, check the files, and can **`delegate`** work back to a direct
subordinate with an order. `finish` sends a report to the agent's commander. When the lead finishes, the optional
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

## Security model

- The server binds `127.0.0.1` only. It rejects requests whose `Host` is not local, and state-changing requests
  whose `Origin` is not local. This blocks DNS-rebinding and cross-site requests; the API can run commands through
  the agent, so no web page may reach it.
- Secrets never pass through wizard state or any API response.
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
