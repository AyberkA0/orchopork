<p align="center">
  <img src="docs/img/logo.png" alt="orchopork logo" width="160">
</p>

<h1 align="center">orchopork</h1>

<p align="center">
  <strong>A harness built for orchestrating many different LLMs as one team, and for measuring whether that team is worth it.</strong>
</p>

<p align="center">
  <a href="https://buymeabitcoffee.vercel.app/btc/bc1qhae9un6j8mel5xcrd4trajm65wfunlltkzelh7?identifier=Buy+Me+a+Coffee"><img src="https://img.shields.io/badge/☕%20Buy%20Me%20a%20Coffee-Donate-f7931a?style=for-the-badge&logo=bitcoin&logoColor=white" alt="Buy Me a Coffee"></a>
</p>

> [!NOTE]
> **An unfinished personal research project.** I built orchopork because I needed it: for my own research into how
> well LLM orchestration actually works, and for my own pipelines. It is shaped by that use first, and shared as
> open source in case it helps others asking the same questions. It is still in progress: several patterns are not
> built yet, and features, file formats and the CLI may change without notice. See
> [why this exists](#why-this-exists) and [where it is going](#where-it-is-going).

orchopork lets Claude, Gemini, DeepSeek, ChatGPT, Mistral, Qwen and any other model, local or cloud, API or CLI
agent, work together on the same task. You compose the orchestration out of nodes, run it, and get detailed
metrics on what each node cost and what it contributed. It is written in Rust, ships as a single binary with an
embedded web UI, and does its work inside isolated git worktrees.

<p align="center">
  <img src="docs/img/fig1-single-vs-team.svg" alt="One goal run two ways: by a single agent, and by a team with a Claude lead and Qwen, DeepSeek, Mistral, Jev and Gemini members; both runs are measured side by side on tokens, cost, time, turns and outcome" width="100%">
</p>

## Why this exists

"Orchestration" is one of the most repeated words around LLMs right now. Put a planner over a few workers, add a
critic, let models debate: it sounds obviously better than one model working alone. Yet it is rarely measured.
Every extra agent re-reads context, every hand-off loses something, and a team of five can easily cost five times
as much for the same result.

orchopork has two goals:

1. **Make real pipelines efficient.** Use expensive models only where they pay off, let cheap local models do the
   bulk of the work, and cap cloud spending with a hard budget.
2. **Study orchestration itself.** Build the orchestration patterns described in the literature from nodes, run
   them on the same tasks with different models, and compare them on tokens, cost, time and outcome. The question
   is not *"can a team of agents do this?"* but *"when does a team beat a single agent, by how much, and how do we
   make that gap bigger?"*

orchopork started as a tool for my own research into these questions, built to run my experiments and my own
pipelines. I am sharing it with the open-source community because the questions are not mine alone: anyone
building with several models runs into them. If you use it to test an orchestration of your own, compare
patterns, or find a case where a team clearly wins or clearly loses, I would like to hear about it; issues and
pull requests are welcome.

## The idea in three layers

```text
  +------------------------------------------------------------------+
  |  Patterns     hierarchy, pipeline, fan-out, debate, vote, ...    |   what the team looks like
  +------------------------------------------------------------------+
  |  Nodes        one agent = one model + role + task + tools        |   who does the work
  +------------------------------------------------------------------+
  |  Engine       worktrees, checkpoints, budget, metrics, rewind    |   what keeps it honest
  +------------------------------------------------------------------+
```

- **Engine.** Every run gets its own branch and `git worktree`; every step is a git commit plus a SQLite row, so
  pause, crash recovery, resume and rewind are exact. Every paid call is reserved against a monthly budget before
  it is sent and recorded after.
- **Nodes.** A node is an agent: a model (any provider, or an external agent such as Claude Code), a role, a task
  and the tools it may use. Nodes do not care which vendor sits behind them; they all speak the same text
  protocol.
- **Patterns.** Nodes are wired into an orchestration. Today that is a chain-of-command tree, which
  also gives fan-out / fan-in and a blackboard-style shared state, plus the classic plan-act-verify-review
  pipeline; the rest of the pattern library is on the roadmap.

## Orchestration patterns

<p align="center">
  <img src="docs/img/fig2-patterns.svg" alt="Eight orchestration patterns: hierarchy, pipeline, fan-out / fan-in and blackboard are available; debate, ensemble voting, router and evaluator loop are planned" width="100%">
</p>

| Pattern           | Status    | Shape                                                                         |
|-------------------|-----------|-------------------------------------------------------------------------------|
| Hierarchy         | available | A lead commands agents, who may command agents. Work flows bottom-up.          |
| Pipeline          | available | plan, act, verify, review, with loops back on failure.                         |
| Fan-out / fan-in  | available | A commander splits the work among subordinates and merges their reports. Agents take turns; truly parallel branches are planned. |
| Debate            | planned   | Agents critique each other for N rounds; a judge decides.                      |
| Ensemble / vote   | planned   | The same task to several models; majority or a scorer picks.                  |
| Router            | planned   | A cheap classifier sends each task to the cheapest model that can handle it.   |
| Evaluator loop    | planned   | A maker and a judge iterate until a bar is met (reflection).                   |
| Blackboard        | available | Agents share one worktree and see each other's reports as they arrive. Turn order follows the tree; opportunistic turns are planned. |

The point of having all of them in one harness is that they can be compared on equal terms: same tasks, same
models, same accounting.

### Mixing patterns within one goal

A real goal is rarely one kind of work. Reading an unfamiliar codebase is wide and parallel; designing an API
benefits from argument; building is a matter of delegation; hardening is a loop of attempts and checks. No single
pattern is the best fit for all of that, so orchopork does not make you pick one for the whole job.

<p align="center">
  <img src="docs/img/fig3-mixed-patterns.svg" alt="One goal, add a payments module, split into four parts run in order: reading the codebase with fan-out / fan-in, designing the API with a debate, building it with a hierarchy, and hardening it with an evaluator loop" width="100%">
</p>

The goal is split into parts, and each part runs with the pattern (and the models) that suit it. Each part hands
its result to the next. Metrics are recorded per agent, so you can see what each part cost and what it delivered,
and swap a pattern out on the next run where it did not pay off.

Today this works within the orchestra tree: every commander's subtree can take its own shape, such as a wide
fan-out of readers under one commander and a deeper chain of command under another, each with its own models.
Debate, voting, the router and the evaluator loop join the mix as they arrive in the pattern library.

## Metrics

Every call is accounted for, per run and per agent:

- input tokens, the share of them read from the prompt cache, and output tokens;
- cost at the provider's real rates (cache reads and writes included), against the monthly cap;
- turns, escalations, failures and review rounds;
- the diff each agent produced, attached to its report.

Every step also records how long it took: the model call and, separately, the tool it ran, so wall-clock time
(which includes waiting for you) and active time are told apart. External agents report their own tokens, and
Claude Code its notional cost, which are shown next to the budget-counted cost but never charged to it.

The run header, each node on the canvas and each agent's panel show these live; the header opens the full
measurements of a run: totals, a timeline with one lane per agent, and per-agent and per-model breakdowns.
**Measurements** (the chart icon in the sidebar, or `#/metrics`) lists every run, grouped by goal, and compares any of them
side by side, with the best value of each row marked and the others as a percentage of the first. *Retry as new*
links the new run to the old one, so the same goal run as a single agent and as several orchestrations lines up
for comparison. Raw numbers are available at `GET /api/runs/<id>/metrics` and `GET /api/metrics?ids=a,b`.

## Where it is going

<p align="center">
  <img src="docs/img/fig4-federation.svg" alt="A coordinating orchopork connected to orchopork nodes on a GPU server, a workstation, a cloud VM and a laptop, each running its own models" width="100%">
</p>

The final version is distributed. An orchopork running on one machine will be able to orchestrate orchopork
instances on other servers, and through them the models those servers run on their own hardware: a 32B model on
a GPU box, a few small models on a workstation, cloud APIs from a VM. A node in the graph will then be "a model on
some machine", and the coordinator will plan with each machine's resources and costs in mind.

Roadmap, roughly in order:

- [x] Single agent and hierarchical orchestra with a visual team editor
- [x] Hybrid local/cloud routing, escalation and a hard monthly budget
- [x] External agents over ACP (Claude Code, Gemini CLI, Codex, Qwen Code, Cursor, Copilot, OpenCode, Goose)
- [x] Headless CLI agents that run with your installed setup (`claude -p`, `codex exec`, `gemini`, `cursor-agent`, `copilot`, `agy`, `aider`)
- [x] Prompt caching and per-agent token/cost metrics
- [x] Agent contracts (owned files, acceptance criteria, verify commands) and team templates
- [x] Fan-out / fan-in and blackboard-style shared state, through the orchestra tree
- [x] Agents ask questions; you answer, or their commander does
- [x] Outputs in the UI: preview images, SVG, web pages, markdown and code a run produced
- [x] Measurements page: every run measured, any runs compared side by side
- [ ] Pattern library: debate, voting, router, evaluator loop
- [ ] Parallel fan-out branches and opportunistic blackboard turns
- [ ] Free-form node graph (any pattern, not only trees)
- [ ] Benchmark mode: one goal, many orchestrations launched together, a comparison report
- [ ] Remote orchopork nodes and cross-server orchestration

## Quick start

```sh
cargo build --release
ollama pull qwen2.5-coder:14b          # or any local model you like
cd /path/to/your/repo
/path/to/orchopork                     # opens http://localhost:7878
```

orchopork only accepts requests from this machine. To reach it from another device, start it with
`orchopork serve --host 0.0.0.0` (keep `--port` as you like) and open `http://<this machine's IP>:7878` there.
Anyone who can reach that port can then make the agents run commands on this machine, so allow the port through
your firewall only on private networks. For more protection, turn on **Settings → Remote access → Require an access
token**: other devices then open the link shown there (such as `http://192.168.1.10:7878/?token=…`) once, which keeps
the token in that browser; scripts can send it as `Authorization: Bearer <token>`. Requests from this machine never
need it. The setting is saved in `~/.orchopork/server.json` and is off by default. The token is kept in
`~/.orchopork/access-token` (or pass `--token`, or set `ORCHOPORK_TOKEN`); delete that file to issue a new one.
Without a token, other devices must use the IP address, not a host name: that is what keeps web pages from reaching
orchopork through DNS rebinding.

The interface is a chat box. When you send a message, orchopork asks **"Activate an agent orchestra?"**

- **No, single agent.** Pick a model (local or cloud). One agent with tools works on the request like a classic
  coding assistant. Replying continues the same conversation. This is also your baseline.
- **Yes, orchestra.** Pick a lead model. The lead drafts a team, and you review it on a canvas before anything
  runs.

### The team editor

```text
                    [ lead ]                       general      > 1000 under command
                   /        \                      major        <= 1000
            [ backend ]   [ tests ]                captain      <= 100
             /      \          \                   corporal     <= 10
      [ api ]   [ storage ]  [ e2e ]               private      none
```

- each agent is a node with an auto-assigned decimal rank, based on how many agents are under its command;
- **drag** an agent onto another to put it under that commander; any agent can command any number of agents;
- **drag from the reserve** (coder, tester, reviewer, researcher, docs writer, refactorer) onto an agent to add a
  subordinate;
- **double-click** an agent to edit its name, role, task, model and commander;
- `Ctrl+Z` undo, `Delete` remove (its subordinates move up), `A` add a subordinate, `Enter` edit; pan by dragging
  the background, zoom with the wheel.

**Approve & deploy** starts it. The same tree then shows live status (who is working, who reported, who has a
question for you, turns, time, tokens and cost per agent), and clicking an agent opens its feed, where you can
message that agent directly.

### Questions from agents

An agent that cannot go on without a decision asks: internal agents with the `ask` tool, external agents (Claude
Code, Codex, …) by ending their reply with a `QUESTION:` line, optionally followed by `- option` lines. Where the
question goes is up to you, per run, and can be changed while it runs (**Questions →** in the toolbar):

- **to you** (the default): the run stops in *needs your answer*, the sidebar lists it under *Waiting for your
  answer*, and a toast (or desktop notification) tells you. Answer on the question card, pick one of the offered
  options, or *Let it decide*; the answer goes to that agent only, and the run continues.
- **to the commander**: in an orchestra, the asking agent's commander answers at once from the mission, its brief and
  the reports so far (one call of its own model, Claude Code included), and the agent continues. A commander that
  cannot decide passes the question on to you. The lead's questions always come to you.

From the CLI: `orchopork answer <run> "…"`.

### Outputs

Whatever a run creates or changes is one click away: an **Outputs** card under the conversation (and under the
lead's feed) shows thumbnails of pictures, SVGs and web pages and lists every other file; **Files** opens a browser
with a live preview (images, SVG, HTML with its scripts in a sandbox, markdown, code with line numbers, PDF, audio,
video) and each file's diff. Files are served only from the run's worktree, under a sandboxing content security
policy, so a generated page can never act as orchopork. Ask for "an SVG of a pelican riding a bicycle" and you see
the pelican, not a path.

### How an orchestra executes

An agent acts only after all of its subordinates have reported, so work flows bottom-up. Commanders read their
subordinates' reports and can **`delegate`** work back to a direct subordinate with an order. `finish` sends a
report to the commander **with the diff of that agent's changes attached**, so commanders review diffs instead of
re-reading files. Teammates under the same commander see each other's reports as they arrive, so later agents build
on earlier work instead of re-exploring it. When the lead finishes, the optional verification command gates
completion.

Each agent's brief can also carry a contract (the lead drafts it, starting from an enabled team template when one
fits, and you can edit it on the canvas):

- **Owns**: files, `folders/` or globs (`src/**/*.rs`) that only this agent's unit may edit. An edit to a file
  another unit owns is refused with the owner's name, so agents report what must change instead of overwriting each
  other; shared files such as manifests are best left to the lead, who may edit anything.
- **Done when**: the acceptance criterion its commander reviews the report against.
- **Verify**: a command (e.g. a targeted test) that must exit 0 before the report is accepted; on failure the agent
  gets the output and tries again. The lead's verify command joins the final verification. Being model-proposed,
  verify commands run only while `allow_commands` is on.

All agents share the run's worktree and act one at a time. The whole command state is checkpointed with every
step, so pause, resume and rewind work exactly as they do for single runs.

### Token economy: a strong lead with cheaper workers

A typical setup is one Opus lead with two Sonnet workers. In the team editor, **Workers** sets the model of every
agent below the lead that has no model of its own. The engine is built so such a team spends fewer tokens than the
lead model working alone:

- **Prompt caching (Claude).** Each turn's context is split into a stable part and a volatile tail. The stable part
  (the mission, the repository as it was when the run started, the agent's role and its conversation so far) is
  sent with cache breakpoints and re-read at a fraction of the input price. What changes every turn goes after the
  last breakpoint. Commanders, who wait between turns, use the 1-hour cache.
- **A window that moves in steps.** When the conversation outgrows the context budget, old turns are dropped a
  third of the budget at a time, so the cached prefix survives many turns in a row.
- **Reports carry diffs**, and teammates see each other's reports: fewer turns spent re-reading files.

Cached tokens are billed at their real rates (reads 0.1x, or 0.05x on Opus 5.5; writes 1.25x, 2x for 1 hour); the
budget guard still reserves the uncached worst case.

### Working with many tasks

- **Sidebar** groups conversations by what needs you: *Pinned*, *Running*, *Needs you* (finished or stopped since
  you last looked), then *Today / This week / Older*. The tab title shows how many tasks are waiting.
- **`Ctrl/Cmd K`** jumps to any conversation, recent project or action.
- **Background updates.** When a task you are not looking at finishes or stops, a toast links to it; desktop
  notifications can be turned on from the sidebar.
- **Drafts are kept.** Unsent text in any composer, and a team the lead drafted, survive navigation, reloads and
  project switches.
- **Rewind from the UI.** *Rewind here* on any step, *Edit & resend* on any of your messages, *Start over* to reset
  the branch, *Retry as new* to run the same goal again with another model or another orchestration while keeping
  the original. Rewound states stay under `refs/orchopork/rewound/`.

## Models and agents

Everything goes through **Add a model** (in the model picker and on *Models & settings*). Each option is tested
before it is saved.

| Source                                   | Kind  | How                                                                     |
|------------------------------------------|-------|-------------------------------------------------------------------------|
| Ollama                                   | local | Lists installed models, downloads new ones in-app with progress.        |
| LM Studio, vLLM, llama.cpp               | local | Local OpenAI-compatible servers; free and never budget-limited.         |
| Claude, Gemini, DeepSeek                 | cloud | API key; the model list is fetched live from the provider.              |
| OpenAI, Mistral, OpenRouter, Groq, xAI, Together, ... | cloud | Any OpenAI-compatible endpoint: base URL + key, optional price per 1M tokens. Models appear as `compat:<endpoint>/<model>`. |
| Claude Code, Codex, Gemini CLI, Antigravity, Qwen Code, Cursor, Copilot, OpenCode, Goose, Aider | agent | Your installed CLI, run headless, or any [Agent Client Protocol](https://agentclientprotocol.com) agent. Detected automatically; one click to add. |

Models are written `provider:model`, e.g. `ollama:qwen2.5-coder:14b`. Prices live in `src/providers/pricing.rs` and
match model ids by longest prefix. Unknown models are priced high, so the budget guard over-estimates rather than
under-estimates.

Ollama's default context window is small and it silently truncates longer prompts, which breaks agent turns.
orchopork sets `num_ctx` on every request (default 16384); raise it if your hardware allows.

### External agents

Coding agents can be used anywhere a model can: as a single agent, as the lead of an orchestra, or as any member of
one. orchopork starts the agent in the run's worktree, sends it the task with the same context an internal agent
would get, streams its actions to the UI, then checkpoints the result. Each agent uses your own login for that
tool, and its cost is not counted against the budget; the tokens (and, for Claude Code, the cost) it reports are
recorded in the per-agent metrics.

There are two ways to connect one, and several tools offer both:

- **Headless CLI.** orchopork runs the CLI you already use in its own non-interactive mode. Nothing else to install,
  and the tool behaves as it does in your terminal: your login, settings, MCP servers, plugins and memory files
  (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`). Its own permission system applies; orchopork maps `allow_commands` onto it.
  Untick **my config** on the agent to make Claude Code ignore your user settings and MCP servers for that agent.
- **ACP.** The [Agent Client Protocol](https://agentclientprotocol.com): orchopork answers the agent's permission
  requests (shell commands only if `allow_commands` is on) and file requests (only inside the worktree) itself.

| Agent         | Headless CLI (what orchopork runs)                          | ACP                                            |
|---------------|-------------------------------------------------------------|------------------------------------------------|
| Claude Code   | `claude -p --output-format stream-json --verbose`           | `npx -y @agentclientprotocol/claude-agent-acp` |
| Codex         | `codex exec --json --sandbox workspace-write -`             | `npx -y @agentclientprotocol/codex-acp`        |
| Gemini CLI    | `gemini --output-format stream-json --approval-mode …`      | `gemini --acp`                                 |
| Qwen Code     | `qwen --output-format stream-json --approval-mode …`        | `qwen --acp`                                   |
| Cursor        | `cursor-agent -p --output-format stream-json "<task>"`      | `cursor-agent acp`                             |
| Copilot       | `copilot -p "<task>"`                                       | `copilot --acp --stdio`                        |
| Antigravity   | `agy -p --input-format stream-json --output-format stream-json` | —                                          |
| Aider         | `aider --message-file <task file> --yes-always --no-auto-commits` | —                                        |
| OpenCode      | —                                                           | `opencode acp`                                 |
| Goose         | —                                                           | `goose acp`                                    |

**Workspace trust.** Every run works in a fresh worktree no tool has seen before, and several CLIs hold back in
an untrusted folder (Gemini CLI and Qwen Code drop to "ask before every action", Cursor stops at a trust prompt).
orchopork settles it before the agent starts: with the tool's own flag where it has one (`--skip-trust` for Gemini
CLI, `--trust` for Cursor), and for tools that keep a trust list (Gemini CLI and Qwen Code, also over ACP) by adding
the folder that holds orchopork's worktrees to it, once. Nothing else is trusted: not your checkout, not your home.
Claude Code skips its trust dialog in `-p` mode, and Codex runs with `--skip-git-repo-check` in its own sandbox.

Antigravity is installed with Google's own script (see [antigravity.google/cli](https://antigravity.google/docs/cli/headless/))
and signed in once by running `agy` in a terminal. agy has no switch that allows only shell commands: with
`allow_commands` on, orchopork passes `--dangerously-skip-permissions`; with it off, reads and writes in the worktree
still work and anything else follows your own `permissions.allow` rules in `~/.gemini/antigravity-cli/settings.json`.

**Add a model** lists every agent it finds on this machine (on PATH, and in the usual install folders such as
`~/.local/bin` or npm's global folder, which a GUI-started process often does not have on PATH), with a guided
install and sign-in for the rest. Any other agent works too: pick **Other agent**, choose ACP or a CLI dialect,
and give the command. For a CLI, the task goes to stdin, or put `{prompt}` (the task as one argument) or
`{prompt_file}` (a temporary file outside the worktree) in the arguments. With the `generic` dialect, stdout is
the report.

In `.orchopork/config.yaml` an agent looks like this:

```yaml
external_agents:
  - id: claude-code-cli
    name: Claude Code (CLI)
    command: claude
    args: [--add-dir, ../shared]   # your own flags; orchopork adds -p, the output format and permissions
    protocol: cli                  # or acp (the default)
    dialect: claude                # claude | codex | gemini | qwen | cursor | copilot | antigravity | aider | generic
    inherit_config: true           # false: --setting-sources project,local --strict-mcp-config
    env: {CLAUDE_CODE_MAX_OUTPUT_TOKENS: "32000"}
```

When you pick an agent, orchopork shows what it can be tuned with: over ACP it asks the agent (model family,
version, effort); for CLIs it offers the known choices (Claude Code's `opus`/`sonnet`/`haiku` and effort, Codex's
reasoning effort). API models get the same treatment: Claude takes an effort level, Gemini takes low/medium/high.
From the CLI or config, use a query suffix: `acp:claude-code-cli?model=opus&effort=high`,
`claude:claude-opus-5-5?effort=xhigh`. For a CLI agent, any other option is passed on as `--key value`
(`agent:` works as a synonym of `acp:`).

## The classic pipeline (CLI)

```text
plan ──► act ⇄ tool ──finish──► verify ──pass──► review ──approve──► done
          ▲                       │fail            │revise
          └───────────────────────┴────────────────┘
```

```sh
orchopork init --actor ollama:qwen2.5-coder:14b --planner claude:claude-sonnet-5 --critic deepseek:deepseek-chat
orchopork providers set-key claude            # reads the key from stdin
orchopork run "Add a --json flag to the list command" --verify "cargo test"
orchopork runs | show <id> | diff <id> | inject <id> "use serde" | answer <id> "yes" | resume <id> | rewind <id> 3 | push <id>
```

| Phase  | Who                                 | What happens                                                                                          |
|--------|-------------------------------------|-------------------------------------------------------------------------------------------------------|
| plan   | `planner` (defaults to the actor)   | One call that sees the goal, the file list and project docs, and writes a numbered plan.              |
| act    | `actor`, or `escalation` when stuck | One JSON action per turn (`read_file`, `edit_file`, `run_command`, ...). Each turn is one checkpoint. |
| verify | you                                 | The `--verify` command and every enabled validator skill must exit 0. Failures go back to the actor.  |
| review | `critic` (optional)                 | Sees goal, plan, summary, verification output and the full diff; answers `approve` or `revise`.       |

The actor is a cheap model by default. After repeated failures a turn is escalated to a stronger one; when the
budget is exhausted, cloud roles fall back to the local actor. `Ctrl-C` pauses after the current step.

**Intervening.** *Inject* adds a message the actor sees on its next turn (also on a finished run, which re-opens
it). *Rewind* resets the worktree to any step and keeps the pre-rewind tree under `refs/orchopork/rewound/`.
*Pause* stops after the step in progress.

**Safety valves** live under `limits` in `.orchopork/config.yaml` and in Settings: `steps_per_session`,
`max_failures`, `escalate_after`, `max_review_rounds`, `command_timeout_secs`, `allow_commands`, per-role context
budgets and `max_output_tokens`.

When a run is done, review the branch and merge it yourself (`git merge orchopork/<id>`), or push it from the
dashboard.

### Actor protocol

Agents reply with a short note and one JSON object:

```json
{"tool": "edit_file", "args": {"path": "src/lib.rs", "old": "fn a()", "new": "fn b()"}}
```

This is a text protocol rather than each vendor's native tool calling, so every backend behaves the same, including
local models without a tool template. That is what lets a Qwen node and a Claude node sit in the same graph. The
parser is lenient: fences or not, `name`/`arguments` aliases, and raw newlines inside JSON strings are accepted.

File tools are confined to the worktree: no absolute paths, no `..`, no symlinks pointing outside, no `.git`.
`run_command` is a real shell with a timeout that kills the whole process group; turn it off with
`limits.allow_commands: false`.

## Skills

Tone, tools, output checks and team templates are declarative files in `.orchopork/skills/` (`*.yaml`, or `*.md` with YAML front
matter whose body becomes `prompt_injection`). They hot-reload from Settings or with `orchopork skills reload`, and
conflicting skills are refused when you enable them.

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
# tool_definition: new tools backed by your commands. Arguments arrive as
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
# validator: substring rules on every reply, and/or a command gating `finish`
name: "no-placeholders"
version: "1.0.0"
type: "validator"
description: "Reject placeholder code; require formatting before finishing."
validator:
  forbidden_substrings: ["todo!()", "unimplemented!()"]
  command: "cargo fmt --check"
```

```yaml
# team_template: a starting-point team the orchestra designer adapts when the
# mission fits `description`. Costs context only in the one team-design call.
name: "team-bugfix"
version: "1.0.0"
type: "team_template"
description: "Fixing a reported bug: reproduce it with a failing test, fix the cause, prove the fix."
team:
  - {id: lead, role: integrates and reviews, task: "...", done_when: "..."}
  - {id: reproducer, parent: lead, role: writes the failing test, task: "...", done_when: "..."}
  - {id: fixer, parent: lead, role: fixes the cause, task: "...", owns: ["src/"]}
```

**Who a skill is for.** Any skill can say `applies_to: [...]`; without it, it applies everywhere. Entries are
`solo` (the single-agent actor), `planner`, `critic`, `designer`, `lead`, `commander` (any orchestra agent with
subordinates), `worker` (any agent below the lead), or `role:<text>` (an orchestra agent whose id, name or role
contains the text). A leading `!` excludes: `applies_to: ["worker", "!role:doc"]` reaches every worker except the
docs writer. This keeps, say, test-first rules away from the lead and the reviewer, and saves their context. A
`validator` whose `command` should check one agent's report (e.g. a `role:test` validator running the test suite)
gates that agent's `finish`; an unscoped validator only gates the final finish, so a full test suite is not rerun at
every report.

Bundled skills: `anti-sycophancy-terse`, `test-driven-loop` (for the solo actor and workers, not the docs writer)
and the team templates `team-bugfix`, `team-feature` and `team-refactor` (enabled by default), plus
`explain-like-principal` and `markdown-memory-sync` (solo actor and lead only), which are off by default because
every enabled modifier costs context on every call.

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

- By default the server binds `127.0.0.1` only. It rejects requests whose `Host` is not local, and state-changing
  requests whose `Origin` is not local. This blocks DNS rebinding and cross-site requests; the API can run commands
  through the agent, so no web page may reach it.
- With `--host`, every request that is not from this machine needs the access token, and changes must be
  same-origin.
- Files a run produced are served from its worktree only, with `X-Content-Type-Options: nosniff` and a sandboxing
  `Content-Security-Policy`, so a generated HTML or SVG file runs (if at all) in an opaque origin.
- Secrets never pass through wizard state or any API response.
- Nothing is written to disk before you choose a workspace, and `git init` only happens when you ask for it.

Cross-server orchestration will need its own authenticated transport; it will not be done by opening this port.

## Development

```sh
cargo test          # unit + end-to-end engine tests (real git/SQLite, scripted LLMs)
cargo clippy --all-targets
cargo fmt
```

Requires Rust 1.89+ and `git` on `PATH`.

```text
src/
  graph/        engine, orchestra, questions, prompts, action protocol, tools
  providers/    gateway, budget, pricing
  storage/      SQLite checkpoints and spend ledger
  skills/       loader, composition, model
  git/          worktrees and branches through hardened git calls
  server/       HTTP API, setup wizard, access control, outputs and measurements
  agents/       external agents: ACP, headless CLIs, presets, workspace trust
ui/index.html   the whole dashboard, dependency-free, embedded in the binary
```

## License

MIT
