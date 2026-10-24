//! Role prompts and the context each role sees. Skills are layered on top
//! of these (see `skills::compose`); nothing tone- or persona-related lives
//! here.

use crate::fsutil::clip;
use crate::providers::{Message, Role};
use crate::storage::{Memory, Step, StepKind};

pub const PLANNER: &str = "You are the planning phase of an autonomous coding agent that works inside a git repository.
Write a concise, numbered plan (typically 3-10 steps) for achieving the goal. Each step should name the files or
areas to inspect or change and what \"done\" means for it. Finish with a line starting with \"Verification:\" that
says how to check the result (tests, build, commands). Do not write the code itself.";

pub const CRITIC: &str = "You are the review phase of an autonomous coding agent. You get the goal, the plan, the
agent's own summary, verification output and the diff of everything it changed. Decide whether the change fully and
correctly achieves the goal. Look for bugs, missing pieces, broken or missing tests, and unrelated changes. Do not
request stylistic rewrites.
Reply with exactly one JSON object: {\"verdict\": \"approve\" | \"revise\", \"feedback\": \"...\"}. When revising,
list concrete, actionable problems.";

pub fn actor_system(tools: &str) -> String {
    format!(
        "You are the execution phase of an autonomous coding agent. You work inside a git repository (the current
directory) and make progress one tool call at a time; each result comes back to you in the next message.

Reply with a short note on what you will do next, then exactly one action as a JSON object:
```json
{{\"tool\": \"<name>\", \"args\": {{ ... }}}}
```
Rules:
- Every reply MUST include exactly one tool call JSON object. Describing changes in prose does not change any file;
  only tool calls do. Do not hide the call inside thinking; put it in your visible reply.
- One action per reply. Never invent tool results; wait for them.
- Messages from the user override the plan. If the user declines a call, do not repeat it.

Files:
- read_file before you edit a file. Copy `old` for edit_file exactly from what read_file showed, without the line
  numbers, and include enough surrounding lines to be unique.
- edit_file changes a part of an existing file. To create a file, or to rewrite most of one, use write_file with the
  COMPLETE new content. Never call edit_file with an empty `old`. Never write files through run_command
  (`cat <<EOF`, `echo >`, `printf >`).
- After a change that did not help, look at the file again before the next change: repeated edits easily leave
  duplicated or half-replaced code behind.
- Inside JSON strings, escape newlines as \\n and double quotes as \\\".

Commands and processes:
- run_command waits for the command to end (tests, builds, scripts). Servers and watchers never end: start them with
  start_process, read them with process_output, stop them with stop_process when done.

Web pages and images:
- Open pages you build with the browser tool (a repo file like index.html needs no server; a localhost URL works only
  for a server you started) and read the console errors and failed requests: a 404 for a texture, model or script
  means that part is missing even when the console is clean. Open the page once after each change, not again and
  again without changing anything.
- To judge how something looks, take a screenshot (screenshot_page); if your model can see images it is shown to you,
  otherwise consult a teammate who can see images and pass the screenshot in images. To check that something moves
  or changes over time, take a screenshot_sequence: it reports which frames differ.
- web_search / web_fetch find documentation, correct library URLs and error explanations.

When you are stuck:
- A failed call is information: read the error and change your approach. Never repeat the same call with the same
  arguments; identical repeats are not run.
- If the same error comes back a second time, or you do not understand an error, or you lack knowledge about a
  library or technique, STOP retrying. Get help first: consult a teammate who can help (see \"Who can help you\"),
  or web_search the exact error message. Guessing again wastes turns.
- Verify your work (build, tests, the page) before calling finish, and finish only when your part is done.

Tools:
{tools}"
    )
}

/// Planner input: the goal, any user messages so far, and an overview of
/// the repository (file list plus key docs).
pub fn planner_input(goal: &str, notes: &[&str], files: &[String], docs: &[(String, String)]) -> String {
    let mut s = format!("# Goal\n{}\n", goal.trim());
    if !notes.is_empty() {
        s.push_str("\n# Notes from the user\n");
        for n in notes {
            s.push_str(&format!("- {}\n", n.trim()));
        }
    }
    s.push_str(&file_overview(files, 300));
    for (name, body) in docs {
        s.push_str(&format!("\n# {name}\n{}\n", clip(body, 4_000)));
    }
    s
}

pub fn critic_input(
    goal: &str,
    plan: &str,
    summary: &str,
    verification: Option<&str>,
    diff_stat: &str,
    diff: &str,
    budget_chars: usize,
) -> String {
    let diff = if diff.trim().is_empty() { "(no changes)".to_string() } else { clip(diff, budget_chars / 2) };
    format!(
        "# Goal\n{}\n\n# Plan\n{}\n\n# Agent's summary\n{}\n\n# Verification\n{}\n\n# Changed files\n{}\n\n# Diff\n```diff\n{}\n```\n",
        goal.trim(),
        clip(plan, 6_000),
        clip(summary.trim(), 6_000),
        verification.map(|v| clip(v, 8_000)).unwrap_or_else(|| "(none configured)".into()),
        if diff_stat.trim().is_empty() { "(none)" } else { diff_stat },
        diff,
    )
}

pub fn file_overview(files: &[String], max: usize) -> String {
    let mut s = format!("\n# Repository files ({})\n", files.len());
    if files.is_empty() {
        s.push_str("(empty repository)\n");
    }
    for f in files.iter().take(max) {
        s.push_str(f);
        s.push('\n');
    }
    if files.len() > max {
        s.push_str(&format!("[{} more; use list_files]\n", files.len() - max));
    }
    s
}

/// What changed in the tree since the run started, for the volatile tail of
/// a turn (see `CompletionRequest::tail`). `name_status` is
/// `git diff --name-status` output.
pub fn changes_note(name_status: &str) -> String {
    let body = name_status.trim();
    if body.is_empty() {
        return "# Files changed since the start\n(none yet)\n".into();
    }
    format!("# Files changed since the start (A added, M modified, D deleted)\n{}\n", clip(body, 3_000))
}

/// The first message of every turn: `shared` is identical for every agent
/// of the run (mission, repository overview), `own` is this agent's part.
/// Both must stay byte-identical across turns so they can be cached;
/// anything that changes per turn belongs in the request's `tail`.
#[derive(Clone)]
pub struct Pinned {
    pub shared: String,
    pub own: String,
}

/// A transcript group (a tool call with its result, a message, …) and the
/// sequence number of the step it came from.
pub type Group = (i64, Vec<Message>);

/// Fitting a conversation into its budget.
pub enum Window {
    Fits(Vec<Message>),
    /// These oldest groups must be folded into the agent's memory summary
    /// (covering steps up to `upto`) before the rest fits.
    Fold { fold: Vec<Vec<Message>>, upto: i64 },
}

/// Like `windowed`, but instead of dropping what does not fit, asks for it
/// to be summarized. Groups already covered by `memory` are left out and
/// the summary goes into the pinned message, where it stays byte-stable
/// (cacheable) until the next fold.
pub fn window(mut pinned: Pinned, groups: Vec<Group>, budget_chars: usize, memory: Option<&Memory>) -> Window {
    let groups: Vec<Group> = groups.into_iter().filter(|(seq, _)| memory.is_none_or(|m| *seq > m.upto)).collect();
    if let Some(m) = memory.filter(|m| !m.text.trim().is_empty()) {
        pinned.own.push_str(&format!(
            "\n# Your memory of earlier turns (a summary; re-read files for exact content)\n{}\n",
            m.text.trim()
        ));
    }
    let sizes: Vec<usize> = groups.iter().map(|(_, g)| g.iter().map(|m| m.content.len()).sum()).collect();
    let room = budget_chars.saturating_sub(pinned.shared.len() + pinned.own.len());
    let omitted = stable_cut(&sizes, room);
    if omitted == 0 {
        return Window::Fits(windowed(pinned, groups.into_iter().map(|(_, g)| g).collect(), usize::MAX));
    }
    let upto = groups[omitted - 1].0;
    Window::Fold { fold: groups.into_iter().take(omitted).map(|(_, g)| g).collect(), upto }
}

/// The single-agent actor's stable first message.
pub fn actor_pinned(goal: &str, plan: &str, files: &[String]) -> Pinned {
    let shared = format!("# Goal\n{}\n\n# Plan\n{}\n{}", goal.trim(), clip(plan, 8_000), file_overview(files, 150));
    Pinned { shared, own: String::new() }
}

/// The single-agent actor's transcript groups.
pub fn actor_groups(steps: &[Step], obs_chars: usize) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    for s in steps {
        let obs = s.observation.as_deref().unwrap_or("");
        let g = match s.kind {
            StepKind::Plan => continue,
            StepKind::Act => {
                let tool = s.meta["tool"].as_str().unwrap_or("action");
                let status = if s.meta["ok"].as_bool().unwrap_or(false) { "ok" } else { "error" };
                vec![
                    Message::assistant(s.output.clone()),
                    Message::user(format!("Result of {tool} ({status}):\n{}", clip(obs, obs_chars))),
                ]
            }
            StepKind::Inject => vec![Message::user(format!("{}:\n{}", inject_label(s), s.output))],
            StepKind::Verify => {
                vec![Message::user(format!("Verification after your finish:\n{}\n\n{}", s.output, clip(obs, obs_chars)))]
            }
            StepKind::Review => {
                if s.meta["verdict"].as_str() != Some("revise") {
                    continue;
                }
                vec![Message::user(format!("The reviewer requested changes:\n{}", s.output))]
            }
        };
        groups.push((s.seq, g));
    }
    groups
}

/// Transcript text for the summarizer.
pub fn render_groups(groups: &[Vec<Message>], per_message: usize) -> String {
    let mut out = String::new();
    for g in groups {
        for m in g {
            let who = if m.role == Role::Assistant { "Agent" } else { "Input" };
            out.push_str(&format!("[{who}]\n{}\n\n", clip(&m.content, per_message)));
        }
    }
    out
}

pub const SUMMARIZER: &str = "You maintain the working memory of an AI coding agent whose conversation no longer fits its \
context. Merge the previous summary and the new transcript into one updated summary for the agent itself, written in the \
second person (\"You ...\"). Keep what it needs to continue: decisions and why, facts learned (file paths, function names, \
APIs, commands and their results), what is finished, what failed and why, open problems and next steps, and any \
instructions from the user or other agents. Drop chit-chat and file contents that can be re-read. Plain text, terse bullet \
points, no preamble.";

/// The actor's conversation: a pinned first message (goal, plan, file
/// overview), then as many of the most recent turns as fit in
/// `budget_chars`. Older turns are dropped whole — a tool call is never
/// separated from its result — and the pinned message says how many.
///
/// `files` should be the file list at the run's base commit (stable); pass
/// what changed since then in the request tail via `changes_note`.
///
/// The result always starts and ends with a user message and never has two
/// consecutive messages with the same role.
pub fn actor_messages(
    goal: &str,
    plan: &str,
    files: &[String],
    steps: &[Step],
    budget_chars: usize,
    obs_chars: usize,
) -> Vec<Message> {
    let groups = actor_groups(steps, obs_chars).into_iter().map(|(_, g)| g).collect();
    windowed(actor_pinned(goal, plan, files), groups, budget_chars)
}

/// How many of the oldest groups to drop so the rest fit in `room` chars.
///
/// The cut moves in coarse steps (a third of the room at a time) instead of
/// one group per turn: dropping a single old turn every turn would change
/// the start of the conversation on every call and defeat prompt caching.
/// With steps, the prefix stays identical for many turns in a row, and what
/// is kept is always between two thirds of the room and all of it.
fn stable_cut(sizes: &[usize], room: usize) -> usize {
    let total: usize = sizes.iter().sum();
    if total <= room || sizes.len() <= 1 {
        return 0;
    }
    let step = (room / 3).max(1);
    let target = (total - room).div_ceil(step) * step;
    let mut cum = 0;
    for (i, s) in sizes.iter().enumerate() {
        cum += s;
        if cum >= target {
            return (i + 1).min(sizes.len() - 1);
        }
    }
    sizes.len() - 1
}

/// Pinned first message + the newest `groups` that fit in `budget_chars`
/// (groups are never split; see `stable_cut` for which are dropped). The
/// shared part of the pinned message is marked as its own cache prefix.
/// Starts and ends with a user message; no two consecutive messages share a
/// role.
pub fn windowed(pinned: Pinned, groups: Vec<Vec<Message>>, budget_chars: usize) -> Vec<Message> {
    let sizes: Vec<usize> = groups.iter().map(|g| g.iter().map(|m| m.content.len()).sum()).collect();
    let room = budget_chars.saturating_sub(pinned.shared.len() + pinned.own.len());
    let omitted = stable_cut(&sizes, room);
    let kept = &groups[omitted..];

    let split = pinned.shared.len();
    let mut first = pinned.shared;
    first.push_str(&pinned.own);
    if omitted > 0 {
        first.push_str(&format!(
            "\n[{omitted} earlier turns are omitted to save context. Re-read files rather than relying on memory.]\n"
        ));
    }
    // No "start now" line on the first turn: the pinned message must read the
    // same on every turn to stay cacheable (the turn's tail follows it).
    let mut msgs = vec![Message { cache_split: Some(split), ..Message::user(first) }];
    for g in kept {
        msgs.extend(g.iter().cloned());
    }
    let mut merged = merge_same_role(msgs);
    if merged.last().is_some_and(|m| m.role == Role::Assistant) {
        merged.push(Message::user("Continue."));
    }
    merged
}

fn merge_same_role(msgs: Vec<Message>) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(msgs.len());
    for m in msgs {
        match out.last_mut() {
            Some(prev) if prev.role == m.role => {
                prev.content.push_str("\n\n");
                prev.content.push_str(&m.content);
            }
            _ => out.push(m),
        }
    }
    out
}

/// How a message from the user is introduced to the agent.
pub fn inject_label(s: &Step) -> &'static str {
    if s.meta.get("answer_to").is_some() { "Answer from the user" } else { "Message from the user" }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::RunState;
    use serde_json::json;

    fn step(seq: i64, kind: StepKind, output: &str, obs: Option<&str>, meta: serde_json::Value) -> Step {
        Step {
            run_id: "r".into(),
            seq,
            kind,
            output: output.into(),
            observation: obs.map(str::to_string),
            meta,
            state: RunState::default(),
            git_commit: "abcdef1".into(),
            cost_usd: 0.0,
            created_at: 0,
        }
    }

    #[test]
    fn transcript_alternates_and_keeps_tool_results_with_their_calls() {
        let steps = vec![
            step(0, StepKind::Plan, "1. do it", None, json!({})),
            step(1, StepKind::Act, "reading", Some("contents"), json!({"tool": "read_file", "ok": true})),
            step(2, StepKind::Inject, "use tabs", None, json!({})),
            step(3, StepKind::Act, "writing", Some("created"), json!({"tool": "write_file", "ok": true})),
        ];
        let m = actor_messages("goal", "1. do it", &["a.rs".into()], &steps, 100_000, 1_000);
        let roles: Vec<Role> = m.iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::User, Role::Assistant, Role::User, Role::Assistant, Role::User]);
        assert!(m[0].content.contains("# Goal\ngoal") && m[0].content.contains("a.rs"));
        assert!(m[2].content.contains("Result of read_file (ok)") && m[2].content.contains("use tabs"));
    }

    #[test]
    fn the_cut_moves_in_steps_so_the_prefix_stays_cacheable() {
        let turn = |i: i64| {
            step(
                i,
                StepKind::Act,
                &format!("turn {i}"),
                Some(&"y".repeat(1_000)),
                json!({"tool": "read_file", "ok": true}),
            )
        };
        let firsts: Vec<String> = (20..40)
            .map(|n| {
                let steps: Vec<Step> = (0..n).map(turn).collect();
                let m = actor_messages("goal", "plan", &[], &steps, 12_000, 10_000);
                m[1].content.clone() // the oldest kept turn
            })
            .collect();
        // Twenty growing transcripts, but the oldest kept turn changes only
        // a few times: most consecutive turns share their whole prefix.
        let changes = firsts.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(changes <= 6, "the window start moved {changes} times in 20 turns");
    }

    #[test]
    fn shared_part_of_the_pinned_message_is_a_cache_prefix() {
        let m = windowed(Pinned { shared: "MISSION\n".into(), own: "YOU\n".into() }, vec![], 1_000);
        assert_eq!(m[0].cache_split, Some("MISSION\n".len()));
        assert!(m[0].content.starts_with("MISSION\nYOU\n"));
    }

    #[test]
    fn old_turns_are_dropped_whole_under_a_tight_budget() {
        let big = "x".repeat(5_000);
        let steps: Vec<Step> = (0..10)
            .map(|i| step(i, StepKind::Act, &format!("turn {i}"), Some(&big), json!({"tool": "read_file", "ok": true})))
            .collect();
        let m = actor_messages("goal", "plan", &[], &steps, 12_000, 10_000);
        assert!(m[0].content.contains("earlier turns are omitted"));
        assert!(m.iter().any(|x| x.content == "turn 9"));
        assert!(!m.iter().any(|x| x.content == "turn 0"));
        assert_eq!(m.last().unwrap().role, Role::User);
        for w in m.windows(2) {
            assert_ne!(w[0].role, w[1].role);
        }
    }
}
