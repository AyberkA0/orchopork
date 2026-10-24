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
- **Yes, orchestra**: pick a lead model. The Lead designs a **flow** of agents, and you review it on a canvas
  before anything runs (see [Orchestra flows](#orchestra-flows)). The designer is told the workspace's limits (the
  internet maximum, whether commands may run, whether the Lead can see images), the role playbooks available
  (including your own `roles:` skills) and the README; it gives every agent a playbook and the least internet it
  needs, keeps implementation out of the Lead's task, adds link abilities only where they pay off (consult from a
  researcher, test into a tester, approve into a reviewer), and picks the Goal's judge. Levels at or above the
  workspace's are shown as the default, and without commands the verification command is dropped:
  - work flows **left to right**, from the **Lead** (start) through the agents to the **Goal** (end);
  - **drag a node's right port** onto another node to connect them; **drag from the palette** (Dispatcher, coder,
    tester, visual tester, reviewer, researcher, docs writer, refactorer) onto the canvas to add it. Every socket takes one link: a
    new link replaces the one the socket had, so the flow is one line. For parallel work use a **Dispatcher** (below);
  - **teams**: select agents and press `Ctrl+G` (or right-click → Make a team) to show them as one node, like a node
    group in Blender; `Tab` or a double-click opens and closes it, `Alt+G` ungroups. It is only a view: at run time
    the agents work as if they were outside. A team is one line of agents (one first, one last agent), and links to
    and from it go to its first and from its last agent;
  - **double-click** an agent to edit it (role, task, model and its **shared notebook**), and a link to choose what
    it lets the two agents do;
  - **select** several nodes by dragging a box over the background (like on the desktop) or with `Shift`/`Ctrl`-click,
    then drag them together, `Delete` them or `Ctrl+D` duplicate them; `Ctrl+A` selects all, `Esc` clears;
  - **right-click** a node (or the selection), a link or the background for its options: edit, pick a notebook,
    duplicate, remove, add an agent here, arrange, fit;
  - **Notebooks** (toolbar button) creates, renames and removes the flow's shared notebooks;
  - problems (a cycle, two links on one socket, an agent with no path to the Goal, …) show as you edit and block
    deploying. The Goal is optional: without one the run ends when every node has finished;
  - `Ctrl+Z` undo, `Enter` edit; pan with the middle mouse button or `Space` + drag, zoom with the wheel. On touch
    screens drag to pan, pinch to zoom, tap a selected node to edit it and long-press for its options.

  **Deploy Orchestra** starts it. The same canvas then shows live status (who is working or reviewing, what is
  done, turns and cost per agent, links lighting up as work passes), and clicking a node opens its feed, where
  you can message that agent directly.

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

A typical setup is one Opus lead with two Sonnet workers. In the flow editor's toolbar, **Workers** sets the model of every
agent after the Lead that has no model of its own (with an Opus or Fable lead, Sonnet is suggested when it is
available). The engine is built so such a team spends fewer tokens than the lead model working alone:

- **Prompt caching (Claude).** Every turn's context is split into a stable part and a volatile tail. The stable
  part is the mission and the repository as it was when the run started (identical for every agent, cached once
  for all of them), the agent's role, and its conversation so far; it is sent with cache breakpoints and re-read at
  a fraction of the input price. What changes every turn (who has reported, which files changed) goes after the
  last breakpoint. Flow agents, who often wait for others between turns, use the 1-hour cache.
- **A window that moves in steps.** When the conversation outgrows the context budget, the oldest turns are folded
  into a summary a third of the budget at a time rather than one per turn, so the cached prefix survives many turns
  in a row.
- **Reports carry diffs**, and teammates see each other's reports: fewer turns spent re-reading files.
- The run header, each agent's node and its panel show input tokens, the share read from cache, and output tokens.
  Cached tokens are billed at their real rates (reads 0.1x, or 0.05x on Opus 5.5; writes 1.25x, 2x for 1 hour);
  the budget guard still reserves the uncached worst case.

### Agents on this computer (ACP)

Claude Code, Gemini CLI, Codex and any other agent that speaks the
[Agent Client Protocol](https://agentclientprotocol.com) can be used anywhere a model can be used: for a single-agent
chat, as the lead of an orchestra, or as any member of one (pick it in the agent's edit dialog). orchopork starts
the agent in the run's worktree, sends it the task with the same context an internal agent would get, streams its
actions to the UI, answers its permission requests (shell commands only if `allow_commands` is on, and through the
run's [autonomy](#autonomy-and-approvals) mode) and file requests
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

### Orchestra flows

An orchestra is a graph with one **Lead**, any number of **agents**, optional **Dispatchers**, at most one **Goal**
and optional **notebooks**. The flow is one line: each node has at most one link in and one link out, and its
agents act one at a time on the run's worktree.

- **Every link carries the work forward**: when an agent calls `finish`, its report goes to the next agent, with
  the diff of its changes attached (or the report only, set per link to save tokens).
- **Dispatchers run work in parallel.** Besides its link in and its link out (*then*), a Dispatcher has any number
  of branch sockets. Connect one to an agent or a team: the branch is that line of agents, and its last agent reports
  back to the Dispatcher (a branch cannot link on elsewhere; no Dispatchers inside branches). When the flow reaches
  it, every branch gets its own git worktree, forked from the run's, and the branches work at the same time. The
  Dispatcher's policy decides when the flow goes on: **wait for all** branches, the **first to finish** (the others
  are cancelled), or **the Dispatcher decides**: its model reads each report and goes on (cancelling the branches
  still running) or waits for more. For **code**, the finished branches are merged into the run's worktree and the
  Dispatcher's model resolves merge conflicts; for **research**, only the reports and notebook notes are kept and
  the branches' file changes are discarded. Its combined report (per branch: finished, merged, cancelled) goes to
  the next node. Branch progress is checkpointed like everything else, so pausing, resuming and rewinding pick the
  branches up where they were; if no branch finishes, the run pauses and resuming retries them. A Dispatcher's
  model must be a model, not an external (ACP) agent.
- **Link abilities** (double-click a link):

  | Ability | What it does |
  |---|---|
  | Consult | either agent may ask the other a question; the other answers in a short read-only session |
  | Suggest | either agent may ask the other for improvement suggestions on the current changes |
  | Test | the first agent may ask the second to test something; it runs it and reports pass / fail |
  | Report | when the second agent finishes, its report goes back to the first, who accepts it or sends it back |
  | Approve | before the second agent starts, it reviews the first one's work and approves it or sends it back |

  Consult / Suggest / Test calls and Report / Approve rounds have per-link limits; a reviewer can read and run
  things but not edit. **Click a link** to see every message that crossed it: handoffs with their diffs,
  questions and answers (with the work the other agent did to answer), test results, verdicts and notes.
- **Vision models work alongside text models.** A node whose model can see images (Claude, Gemini, GPT-4o-class
  models, and Ollama models that report the `vision` capability, e.g. `qwen2.5vl`, `gemma3`, `llava`; force it with
  `?vision=true` on the model reference) shows 👁 on the canvas and gets `view_image`. Over a **Consult** or **Suggest**
  link, a text model can pass screenshots in `images` to such an agent and get its reading back; the agent list tells
  each model who can see. When the Goal's judge can see, it is also shown the page the engine opened.
- **Notebooks** are shared memory for the agents that use them (`notebook_write` / `notebook_read`), for this run
  only. They are not nodes on the canvas: create them in the **Notebooks** window and pick each agent's notebook
  (at most one) in its dialog or right-click menu. An agent shows its notebook on its card; on a running flow the
  Notebooks window lists what was written. Notes rewind with the run.
- **The Goal** holds the success criteria and decides whether they are met: the agent connected to it asks
  itself, an independent judge model decides, or only a verification command does. A command, when set, always
  runs first and must pass. If the goal is not reached, the **Lead** gets the reasons and coordinates the fix:
  `rerun` sends an agent back with an instruction (it and everything after it run again), or the Lead fixes it
  itself. After the Goal's max checks the run pauses for you.
- **Dynamic agents** (option on the Lead): the Lead may add agents between two nodes (`spawn_agent`) or remove one
  (`retire_agent`) while the run executes. In Semi-auto and Manual mode each change waits for your approval.
- **Re-orchestrate** (run header): cancels the task (its branch and history stay) and has the Lead design a new
  flow for a new goal, from a clean start or from the cancelled task's code. In Full auto it starts right away;
  otherwise you review the new flow first.
- Sending a message to an agent that already finished reopens it and everything after it. Each agent's context is
  its own turns plus what reached it over its links; when that outgrows the budget, the oldest turns are folded
  into a running summary instead of being dropped.

- **Preflight** (**Check** in the editor's toolbar, and before every deploy or save): each model is called once, reporting
  whether it loads, takes native tool calls and sees images, and the design is checked for mistakes that only show
  later (visual criteria with a judge that cannot see, tasks that mention an agent they have no link to, notebooks
  shared by one agent, vague criteria). Problems are marked ⚠ on the nodes.
- **Stuck agents hand back.** An agent that uses up `limits.node_turns` (default 25) on one assignment, or keeps
  getting the same error, stops and hands the problem to the Lead with the errors and what it tried; the Lead reruns
  it with an instruction or changes the plan. Identical calls with nothing changed since are not run again, the same
  failure more than three times is written to the shared notebooks as a dead end, and repeated errors come with a
  strong hint to consult a teammate. A stuck Lead (or a stuck single agent past its failures) pauses the run.
- **Lessons** (task menu → Learn lessons from this task): the Lead's model turns dead ends, failed Goal checks and
  hand-backs into short "when X, do Y" lessons; the ones you keep go into the `project-lessons` skill that every
  agent of later tasks reads.
- **Live output**: replies from Ollama and OpenAI-compatible endpoints stream into the task view (and the acting
  node's panel) while they are written, reasoning in italics. **■ Kill** stops a task mid-step.
- Every file-changing step shows `+added −removed` lines and opens its own diff; nodes show their totals and
  context use (`ctx 7k/16k`).

Tasks from the old hierarchical orchestra stay viewable, read-only.

#### Role playbooks: every agent knows its job and the team's rules

Every orchestra agent gets two bundled skills on top of the general ones, so it knows what its role is *and* how it
fits into the chain (see [Skills](#skills)):

- `team-protocol` for everyone: stay in your lane, the handoff report and diff are all the next agent gets, don't
  redo finished work, limited turns per assignment, ask for help after two failed attempts, finish with *Done /
  Verified / Open / For the next agent* (and `BLOCKED:` or `PARTIAL:` when it cannot finish).
- one **role playbook** chosen by the agent's *Role playbook* (double-click → Role playbook): `coder`, `tester`,
  `visual-tester` (screenshots at desktop and phone sizes, motion via frame sequences; best with a vision model, else
  it consults one), `reviewer`, `researcher`, `docs`, `refactorer`; the Lead and a Dispatcher get `lead` and `dispatcher` by their kind.
  Left on *Auto*, the playbook is guessed from the agent's name and role text; an agent that matches nothing gets the
  team rules only. The Lead can set it when it designs the flow or adds an agent (`archetype`).

**A Lead that out-works its team** was the most common failure, so the engine enforces the split, not only the
prompt. A Lead that has agents after it (or may add some) can read, search and look things up, but cannot edit
files or run commands; it hands the work over and, when the Goal check fails, sends the responsible agent back with
`rerun`. The tail of its context lists the turns each agent has taken, and after 6 turns of its own it is told to
hand off now. A Lead alone in its flow can always edit, and *Hands-on* in the Lead's dialog turns the restriction
off. An external (ACP) Lead is told the same, and its edit and command permission requests and protocol file
writes are refused during its first assignment (it has no `rerun`, so after a failed Goal check it may fix things
itself); an agent's own tools that never ask permission can only be asked to comply.

#### Internet access per agent

Settings → Execution Safety → *Internet access for agents* sets the most any agent may reach; in an orchestra each
agent can narrow it for itself (double-click → Internet access, shown as `web·docs` on its card). Levels:

| Level | Web tools | Browser / screenshots |
|---|---|---|
| None (`off`) | none | none |
| Local pages only (`local`) | none | this project's files and `localhost` servers |
| Documentation (`docs`) | `web_search`, `web_fetch` | local pages only |
| Full (`full`, default) | `web_search`, `web_fetch` | any URL |

An agent's level is the lower of its own and the workspace's, and also applies to the sessions it is consulted in. A
tool the level removes is not offered, and a call to it anyway is refused. Without `docs` the shell is also refused
obvious downloads (`curl`, `wget`, PowerShell's web cmdlets, `ssh`, `git clone/fetch/pull/push`). This controls the
agent's tools; it is not a firewall: a build or install command (`npm install`, `cargo build`) can still reach
package registries, and ACP agents bring their own web tools: they are told their level and their `fetch`
permission requests are refused, nothing more. In config: `limits.network: off|local|docs|full` (with
`limits.allow_network: false` as a master switch), per node `network:` and `archetype:` in the flow.

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
`command_timeout_secs`, `allow_commands`, `allow_network`, per-role context budgets and `max_output_tokens`.

When a run is done, review the branch and merge it yourself: `git merge orchopork/<id>`. You can also push it from
the dashboard, or merge it there (**Merge**, after a confirmation); a merge that conflicts is aborted, so
your checkout is never left half-merged.

### Actor protocol

Every turn is exactly one tool call. Tools are offered through the provider's native tool calling (Ollama,
OpenAI-compatible servers, Claude), and the call is recorded as one JSON object:

```json
{"tool": "edit_file", "args": {"path": "src/lib.rs", "old": "fn a()", "new": "fn b()"}}
```

Models or endpoints that reject native tools (an Ollama model without a tool template, `llama-server` without
`--jinja`) are switched to the plain text protocol automatically: the model writes that JSON object itself. The
parser is lenient: fences or not, `name`/`arguments` aliases, raw newlines inside JSON strings, and plain read
commands such as `read_file src/main.rs` are all accepted.

| Tools | |
|---|---|
| Files | `list_files`, `list_dir`, `read_file`, `search`, `write_file`, `edit_file` (`replace_all`), `move_file`, `delete_file` |
| Commands | `run_command` (`cwd`, `timeout_secs`), `start_process` / `process_output` / `stop_process` for dev servers and watchers |
| Web | `web_search`, `web_fetch`, `browser`: a headless Edge/Chrome that runs JavaScript and returns the rendered text, console errors and an optional screenshot; repo files like `index.html` are served over http so ES modules work (`ORCHOPORK_BROWSER` overrides the browser path) |
| Screenshots & images | `screenshot_page` (a page in the headless browser), `screenshot_sequence` (several frames of a page a fixed time apart, reporting which frames changed: is the animation really moving?), `screenshot_screen` (the whole screen, once or several times in a row; off unless enabled in Settings), `view_image` (one or several images at once; only for models that can see images). Screenshots go to `.orchopork-shots/` and are never checkpointed |

File tools are confined to the worktree: no absolute paths, no `..` escapes, no symlinks pointing outside, no
`.git`. `run_command` is a real shell with a timeout; its whole process group is killed when the timeout hits.
Background processes are killed with their process tree when stopped or when the run is deleted. Turn commands off
with `limits.allow_commands: false` and the web tools with `limits.allow_network: false` (or set a finer
`limits.network` level, see [Internet access per agent](#internet-access-per-agent)).

### Autonomy and approvals

Each run has an autonomy mode, chosen when you start it and switchable at any time from the run's header:

| Mode | Asks before |
|---|---|
| Full auto | nothing |
| Semi-auto | commands, background processes, deletions and skill tools (file edits run freely) |
| Manual | every change: file writes and edits too (reads never ask) |

A call that needs approval pauses the run on an approval card showing the command or the content/diff. You can
allow it, allow that tool for the rest of the run, or decline with a note the agent reads. Switching to a mode that
would not have asked releases the waiting call. On the CLI: `orchopork run "…" --autonomy semi` and answer
`y` / `n` / `a` (always) or type a reason.

The dashboard's **Browser** pane shows the run's files (or any dev server URL) in a sandboxed frame, so pages the
agent wrote cannot call the dashboard API.

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

A skill with `roles` applies only to orchestra agents of that role, and not to single-agent chats: `roles: [tester]`
for one role key (`lead`, `coder`, `tester`, `visual-tester`, `reviewer`, `researcher`, `docs`, `refactorer`, `dispatcher`, or a key of
your own that you give an agent as its `archetype`), `roles: ["*"]` for every orchestra agent; without `roles` a skill
applies everywhere. Copy a bundled `role-*.yaml` to tune a playbook for your project (the file in
`.orchopork/skills/` is yours; updates never overwrite it, and a new bundled default skill is switched on once).

Bundled skills: `anti-sycophancy-terse`, `test-driven-loop` and `no-false-claims` (enabled by default; the last one
makes the engine reject reports that claim more than the agent checked), the orchestra playbooks `team-protocol`,
`role-lead`, `role-coder`, `role-tester`, `role-visual-tester`, `role-reviewer`, `role-researcher`, `role-docs-writer`, `role-refactorer`
and `role-dispatcher` (enabled by default; an agent receives only the team rules and its own role, about 1k tokens,
cached with the rest of its stable prompt), plus `explain-like-principal` and
`markdown-memory-sync` (off by default, because every enabled modifier costs context on every call). Skill tools may
not reuse the name of a built-in or orchestra tool (`read_file`, `browser`, `consult`, …).

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

## Security model

- `orchopork serve` listens on `0.0.0.0:7878` by default so other devices can reach it, but only with an access
  token (created in Settings, stored in `~/.orchopork/tokens.json`); tokens can only be managed from the machine
  itself. Use `--host 127.0.0.1` to accept local connections only.
- Without a token, a request is accepted only when both its `Host` and the connecting peer are local, and a
  state-changing request also needs a local `Origin`. This blocks DNS-rebinding and cross-site requests; the API can
  run commands through the agent, so no web page may reach it.
- Run previews are served with CSP `sandbox` (an opaque origin), and requests from sandboxed pages (`Origin: null`)
  never reach the API, even with a token in the page's URL.
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
