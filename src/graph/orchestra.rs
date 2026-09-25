//! Agent orchestras: a tree of agents organised like a decimal military
//! chain of command. Every agent owns a task; commanders also own the work
//! of their subordinates.
//!
//! Scheduling is a depth-first command stack stored in `RunState`
//! (`stack` = chain of command down to the agent acting now, `done` =
//! finished agents), so it checkpoints, resumes and rewinds like any run:
//!
//! - An agent acts only after all of its subordinates reported `done`
//!   (their reports arrive in its transcript), so work flows bottom-up.
//! - A commander may `delegate` work back to a direct subordinate with an
//!   order; that subordinate becomes active again and reports back.
//! - `finish` delivers a report to the commander. When the lead finishes,
//!   the run's verification commands gate completion; a failure goes back
//!   to the lead.
//!
//! All agents share the run's worktree, one acting at a time.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::json;

use super::{Engine, Inner, Stop, call_meta, prompts, protocol, verify_commands};
use crate::config::{Config, ModelRef};
use crate::error::{Error, Result};
use crate::fsutil::clip;
use crate::git::GitRepo;
use crate::graph::tools::ToolBox;
use crate::providers::{CompletionRequest, Message};
use crate::storage::{AgentSpec, Phase, Run, RunSpec, RunState, Step, StepKind};

/// Decimal ranks by the number of agents under command (the whole subtree).
pub fn rank(spec: &RunSpec, id: &str) -> &'static str {
    match subtree_size(spec, id) {
        0 => "private",
        1..=10 => "corporal (squad leader)",
        11..=100 => "captain (company commander)",
        101..=1000 => "major (battalion commander)",
        _ => "general",
    }
}

fn subtree_size(spec: &RunSpec, id: &str) -> usize {
    spec.children(id).map(|c| 1 + subtree_size(spec, &c.id)).sum()
}

const EXTERNAL_SUFFIX: &str = "\n\n---\nWork directly in the current directory (the shared git working tree). Do only your task. When you are done, end with a concise report for your commander: what you changed, how you verified it, and anything left open.";

const DESIGNER: &str = "You design teams of AI agents for software tasks, organised like a decimal military chain of
command: exactly one lead agent at the top; any agent may command subordinates; keep each commander's direct reports
at 10 or fewer. Use the fewest agents that make sense: 1 for a trivial task, typically 3-8, deeper trees only for large
missions. Leaf agents do concrete work (read code, edit files, run tests). Commanders integrate and review their
subordinates' work, send it back when it is wrong, and report upward. All agents share one working tree and act one at
a time, so give each a clearly separated responsibility and tell dependent agents what to expect.
Reply with only a JSON object:
{\"agents\": [{\"id\": \"lead\", \"name\": \"Lead\", \"role\": \"short role\", \"task\": \"concrete, self-contained task\", \"parent\": null}, ...]}
Ids are short snake_case; `parent` is the commander's id (null only for the lead).";

impl Engine {
    /// Asks `lead` to design a team for `goal`. Always returns a valid tree:
    /// if the reply cannot be used, a single lead agent owning the goal.
    pub async fn propose_team(&self, goal: &str, lead: &ModelRef) -> Result<Vec<AgentSpec>> {
        let goal = goal.trim();
        if goal.is_empty() {
            return Err(Error::InvalidRequest("goal is empty".into()));
        }
        let ws = &self.inner.ws;
        let files = ws.repo.ls_files().await.unwrap_or_default();
        let input = format!("# Mission\n{goal}\n{}", prompts::file_overview(&files, 200));
        let req = CompletionRequest {
            system: DESIGNER.into(),
            messages: vec![Message::user(input)],
            max_tokens: ws.config().limits.max_output_tokens,
            effort: None,
        };
        let out = ws.gateway.complete(lead, &req, "team-design").await?;
        Ok(parse_team(&out.text, goal))
    }
}

/// Lowercase, non-alphanumeric → `_`: the canonical form for both an
/// agent's `id` and any `parent` field naming one, so the two compare equal
/// regardless of how the model cased or punctuated them.
fn sanitize_id(s: &str) -> String {
    s.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

/// Lenient team parsing + repair: sanitised unique ids, a single root,
/// dangling parents re-attached to the root.
pub fn parse_team(text: &str, goal: &str) -> Vec<AgentSpec> {
    let fallback = || {
        vec![AgentSpec {
            id: "lead".into(),
            name: "Lead".into(),
            role: "does the whole mission".into(),
            task: goal.to_string(),
            parent: None,
            model: None,
        }]
    };
    let Some(list) = protocol::json_objects(text)
        .into_iter()
        .rev()
        .find_map(|o| o.get("agents").and_then(|a| a.as_array()).cloned())
    else {
        return fallback();
    };
    let mut agents: Vec<AgentSpec> = Vec::new();
    let mut ids = HashSet::new();
    for (i, v) in list.iter().enumerate() {
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
        // Skip empty-task entries before reserving an id slot, so a skipped
        // agent cannot bump a later, real agent's id to a fallback.
        let task = s("task");
        if task.is_empty() {
            continue;
        }
        let mut id = sanitize_id(&s("id"));
        if id.is_empty() || ids.contains(&id) {
            id = format!("agent_{}", i + 1);
        }
        ids.insert(id.clone());
        let name = if s("name").is_empty() { id.clone() } else { s("name") };
        // `parent` names another agent's `id`, so it must be sanitised the
        // same way or a valid parent looks unknown and gets reattached to
        // the root below.
        let parent = Some(sanitize_id(&s("parent"))).filter(|p| !p.is_empty() && p != "null");
        agents.push(AgentSpec { id, name, role: s("role"), task, parent, model: None });
    }
    if agents.is_empty() {
        return fallback();
    }
    let root = agents.iter().find(|a| a.parent.is_none()).map(|a| a.id.clone()).unwrap_or(agents[0].id.clone());
    let known: HashSet<String> = agents.iter().map(|a| a.id.clone()).collect();
    for a in &mut agents {
        if a.id == root {
            a.parent = None;
        } else if a.parent.as_ref().is_none_or(|p| !known.contains(p) || *p == a.id) {
            a.parent = Some(root.clone());
        }
    }
    let spec = RunSpec { agents: agents.clone(), ..Default::default() };
    if spec.validate_agents().is_err() {
        // Cycles: flatten everything under the root.
        for a in &mut agents {
            if a.id != root {
                a.parent = Some(root.clone());
            }
        }
    }
    agents
}

impl Inner {
    pub(super) async fn orchestra_turn(
        &self,
        cfg: &Config,
        run: &Run,
        steps: &[Step],
        mut state: RunState,
    ) -> Result<Option<Stop>> {
        let spec = &run.spec;
        let root = spec.root().ok_or_else(|| Error::InvalidRequest("orchestra has no lead".into()))?.id.clone();

        if state.phase == Phase::Verify {
            let (ok, summary, details) = self.run_verification(cfg, run).await?;
            let mut next = RunState { failures: 0, escalate: false, ..state };
            if ok {
                next.phase = Phase::Done;
            } else {
                next.phase = Phase::Act;
                next.done.retain(|d| *d != root);
                next.stack.clear();
            }
            let meta = json!({ "ok": ok, "agent": root });
            self.append(run, StepKind::Verify, &summary, Some(&details), meta, &next, 0.0).await?;
            return Ok(None);
        }

        // A message injected at agent X while X had already reported (X is
        // in `done`) would otherwise sit unread forever: nothing re-reads
        // `done` once it's set. Reopen X and its whole chain of command (so
        // the report bubbles back up) whenever such a message is newer than
        // X's last turn. Idempotent: once reopened, X leaves `done` and this
        // no longer matches on later turns.
        let mut last_act: HashMap<&str, i64> = HashMap::new();
        for s in steps {
            if s.kind == StepKind::Act
                && let Some(a) = s.meta["agent"].as_str()
            {
                last_act.insert(a, s.seq);
            }
        }
        for s in steps {
            if s.kind != StepKind::Inject {
                continue;
            }
            let target = s.meta["agent"].as_str().unwrap_or(&root);
            let pending = last_act.get(target).is_none_or(|&seq| s.seq > seq);
            if pending && state.done.iter().any(|d| d.as_str() == target) {
                let mut cur = Some(target.to_string());
                while let Some(id) = cur {
                    state.done.retain(|d| *d != id);
                    cur = spec.agent(&id).and_then(|a| a.parent.clone());
                }
                state.stack.clear();
            }
        }

        // Descend the chain of command to the first agent with no
        // unfinished subordinates.
        if state.stack.is_empty() {
            state.stack.push(root.clone());
        }
        while let Some(top) = state.stack.last().cloned() {
            match spec.children(&top).find(|c| !state.done.contains(&c.id)) {
                Some(c) => state.stack.push(c.id.clone()),
                None => break,
            }
        }
        let me = spec.agent(state.stack.last().expect("non-empty")).cloned().ok_or_else(|| {
            Error::InvalidRequest("the command stack names an agent that is not in the orchestra".into())
        })?;
        let subordinates: Vec<&AgentSpec> = spec.children(&me.id).collect();

        // Escalation and the failure streak are per-agent, but `RunState` is
        // shared for the whole tree: reset both when the acting agent
        // changes, so one agent's failures never escalate a different
        // agent's next turn.
        let last_actor = steps.iter().rev().find(|s| s.kind == StepKind::Act).and_then(|s| s.meta["agent"].as_str());
        if last_actor != Some(me.id.as_str()) {
            state.failures = 0;
            state.escalate = false;
        }

        let base = me.model.clone().or_else(|| spec.model.clone()).or_else(|| cfg.routing.actor.clone());
        let base = base.ok_or_else(|| Error::InvalidRequest(format!("no model for agent {}", me.name)))?;
        // Escalation swaps in a different (internal) model, so it never
        // applies when the agent's own model is an external ACP agent.
        let (model, escalated) = match (&cfg.routing.escalation, state.escalate) {
            (Some(e), true) if base.provider != crate::providers::ProviderId::Acp => (e.clone(), true),
            _ => (base, false),
        };

        if model.provider == crate::providers::ProviderId::Acp {
            let files = GitRepo::new(&run.worktree).ls_files().await?;
            let mut pin = pinned(run, spec, &me, &subordinates, &state, &files);
            if !subordinates.is_empty() {
                pin.push_str("\nYou cannot delegate in this mode: review your subordinates' work in the files and fix or complete what is missing yourself.\n");
            }
            let groups = transcript(spec, &me, &root, steps, cfg.limits.tool_output_chars);
            let messages = prompts::windowed(pin, groups, cfg.limits.cloud_context_chars);
            let prompt = format!("{}{}", super::external::as_prompt(&messages), EXTERNAL_SUFFIX);
            let out = self.external_turn(cfg, run, &model, &me.name, prompt).await?;
            let mut meta = super::external::meta(&model, &out);
            meta["agent"] = json!(me.id);
            let mut next = RunState { failures: 0, escalate: false, ..state };
            let text =
                if out.text.is_empty() { format!("({} finished without a report)", me.name) } else { out.text.clone() };
            if !out.cancelled {
                meta["summary"] = json!(clip(&text, 6_000));
                next.done.push(me.id.clone());
                next.stack.pop();
                if me.parent.is_none() {
                    let skills = self.ws.skills.compose("")?;
                    next.phase = if verify_commands(run, &skills).is_empty() { Phase::Done } else { Phase::Verify };
                }
            }
            let obs = super::external::tool_lines(&out);
            self.append(run, StepKind::Act, &text, Some(&obs), meta, &next, 0.0).await?;
            return Ok(out.cancelled.then_some(Stop::Paused(None)));
        }

        let skills = self.ws.skills.compose("")?;
        let toolbox = ToolBox::new(Path::new(&run.worktree), &cfg.limits, skills.tools.clone())?;
        let mut tools = toolbox.describe();
        if !subordinates.is_empty() {
            tools.push_str(
                "- delegate {\"agent\": string, \"instruction\": string}: send work back to one of your direct subordinates (by id); they act on it and report back to you.\n",
            );
        }
        let system = format!("{}\n\n{}", prompts::actor_system(&tools), skills.system).trim_end().to_string();
        let files = GitRepo::new(&run.worktree).ls_files().await?;
        let pinned = pinned(run, spec, &me, &subordinates, &state, &files);
        let groups = transcript(spec, &me, &root, steps, cfg.limits.tool_output_chars);
        let messages = prompts::windowed(pinned, groups, cfg.context_chars(&model));
        self.log(&run.id, format!("{} is working", me.name));
        let out = self.call(cfg, run, &model, system, messages).await?;

        let mut meta = call_meta(&out);
        meta["agent"] = json!(me.id);
        meta["escalated"] = json!(escalated);
        let mut next = state.clone();
        let violations = skills.check_output(&out.text);
        let (observation, ok) = if !violations.is_empty() {
            let list: Vec<String> = violations.iter().map(|v| format!("{} (reply {})", v.skill, v.detail)).collect();
            (format!("Your reply was rejected by validator skills: {}. Redo it.", list.join("; ")), false)
        } else {
            match protocol::parse_action(&out.text) {
                Err(e) => (
                    format!(
                        "Could not parse an action: {e}. Reply with a short note and exactly one JSON object, e.g. {{\"tool\": \"read_file\", \"args\": {{\"path\": \"README.md\"}}}}."
                    ),
                    false,
                ),
                Ok(a) if a.tool == "finish" => {
                    let summary = a.str_arg("summary").or_else(|| a.str_arg("report")).unwrap_or("").trim().to_string();
                    let summary = if summary.is_empty() { out.text.clone() } else { summary };
                    meta["tool"] = json!("finish");
                    meta["summary"] = json!(summary);
                    next.done.push(me.id.clone());
                    next.stack.pop();
                    let msg = match me.parent.as_deref().and_then(|p| spec.agent(p)) {
                        Some(c) => format!("Report delivered to {}.", c.name),
                        None => {
                            next.phase =
                                if verify_commands(run, &skills).is_empty() { Phase::Done } else { Phase::Verify };
                            "Mission report delivered.".to_string()
                        }
                    };
                    (msg, true)
                }
                Ok(a) if a.tool == "delegate" => {
                    meta["tool"] = json!("delegate");
                    let wanted = a.str_arg("agent").or_else(|| a.str_arg("to")).unwrap_or("").trim().to_lowercase();
                    let target =
                        subordinates.iter().find(|s| s.id.to_lowercase() == wanted || s.name.to_lowercase() == wanted);
                    let instruction = a.str_arg("instruction").or_else(|| a.str_arg("task")).unwrap_or("").trim();
                    match (target, instruction.is_empty()) {
                        (Some(t), false) => {
                            meta["target"] = json!(t.id);
                            meta["instruction"] = json!(instruction);
                            next.done.retain(|d| *d != t.id);
                            next.stack.push(t.id.clone());
                            (format!("Order sent to {}; their report will follow.", t.name), true)
                        }
                        (None, _) => {
                            let ids: Vec<&str> = subordinates.iter().map(|s| s.id.as_str()).collect();
                            (format!("No direct subordinate {wanted:?}; yours are: {}", ids.join(", ")), false)
                        }
                        (_, true) => ("delegate needs a non-empty \"instruction\"".into(), false),
                    }
                }
                Ok(a) => {
                    meta["tool"] = json!(a.tool);
                    let r = toolbox.execute(&a).await;
                    (r.output, r.ok)
                }
            }
        };
        meta["ok"] = json!(ok);
        if ok {
            next.failures = 0;
            next.escalate = false;
        } else {
            next.failures += 1;
            next.escalate = cfg.routing.escalation.is_some() && next.failures >= cfg.limits.escalate_after;
        }
        self.append(run, StepKind::Act, &out.text, Some(&observation), meta, &next, out.cost_usd).await?;
        if next.failures >= cfg.limits.max_failures {
            return Ok(Some(Stop::Paused(Some(format!(
                "{} failed {} turns in a row; message them with guidance or change their model, then resume",
                me.name, next.failures
            )))));
        }
        Ok(None)
    }
}

fn pinned(
    run: &Run,
    spec: &RunSpec,
    me: &AgentSpec,
    subs: &[&AgentSpec],
    state: &RunState,
    files: &[String],
) -> String {
    let mut s = format!(
        "# Mission (from the user)\n{}\n\n# You\n{} (id {}), rank {}. Role: {}\nYour task: {}\n",
        run.goal.trim(),
        me.name,
        me.id,
        rank(spec, &me.id),
        if me.role.is_empty() { "-" } else { &me.role },
        me.task.trim()
    );
    match me.parent.as_deref().and_then(|p| spec.agent(p)) {
        Some(c) => s.push_str(&format!(
            "You report to {} ({}). Do your task only, then call finish with a precise report of what you did and anything they must know.\n",
            c.name, c.role
        )),
        None => s.push_str(
            "You lead the orchestra and answer to the user. Call finish only when the whole mission is complete.\n",
        ),
    }
    if !subs.is_empty() {
        s.push_str("\n# Your subordinates (their reports are below)\n");
        for c in subs {
            let status = if state.done.contains(&c.id) { "reported" } else { "working" };
            s.push_str(&format!("- {} [{}]: {} ({status}) — {}\n", c.id, c.name, c.role, clip(&c.task, 300)));
        }
        s.push_str(
            "Check their work in the files. If something is wrong or missing, use delegate with a precise instruction instead of redoing it yourself; integrate and verify, then finish.\n",
        );
    }
    let others: Vec<String> = spec
        .agents
        .iter()
        .filter(|a| a.id != me.id && a.parent.as_deref() != Some(&me.id) && Some(a.id.as_str()) != me.parent.as_deref())
        .map(|a| format!("{} ({})", a.name, a.role))
        .collect();
    if !others.is_empty() {
        s.push_str(&format!("\nOther agents working in the same tree: {}.\n", others.join(", ")));
    }
    s.push_str(&prompts::file_overview(files, 150));
    s
}

/// What `me` has seen: its own turns, orders addressed to it, reports from
/// its direct subordinates, user messages for it, and (lead only)
/// verification results.
fn transcript(spec: &RunSpec, me: &AgentSpec, root: &str, steps: &[Step], obs_chars: usize) -> Vec<Vec<Message>> {
    let mut groups = Vec::new();
    let name = |id: &str| spec.agent(id).map(|a| a.name.clone()).unwrap_or_else(|| id.to_string());
    for s in steps {
        let agent = s.meta["agent"].as_str().unwrap_or(root);
        match s.kind {
            StepKind::Act if agent == me.id => {
                let tool = s.meta["tool"].as_str().unwrap_or("action");
                let status = if s.meta["ok"].as_bool().unwrap_or(false) { "ok" } else { "error" };
                groups.push(vec![
                    Message::assistant(s.output.clone()),
                    Message::user(format!(
                        "Result of {tool} ({status}):\n{}",
                        clip(s.observation.as_deref().unwrap_or(""), obs_chars)
                    )),
                ]);
            }
            StepKind::Act if s.meta["tool"] == "delegate" && s.meta["target"] == me.id.as_str() => {
                groups.push(vec![Message::user(format!(
                    "Order from {}:\n{}",
                    name(agent),
                    s.meta["instruction"].as_str().unwrap_or("")
                ))]);
            }
            StepKind::Act
                if s.meta["tool"] == "finish"
                    && spec.agent(agent).and_then(|a| a.parent.as_deref()) == Some(&me.id) =>
            {
                groups.push(vec![Message::user(format!(
                    "Report from {} ({}):\n{}",
                    name(agent),
                    agent,
                    s.meta["summary"].as_str().unwrap_or(&s.output)
                ))]);
            }
            StepKind::Inject if agent == me.id => {
                groups.push(vec![Message::user(format!("Message from the user:\n{}", s.output))]);
            }
            StepKind::Verify if me.id == root => groups.push(vec![Message::user(format!(
                "Verification after your finish:\n{}\n\n{}",
                s.output,
                clip(s.observation.as_deref().unwrap_or(""), obs_chars)
            ))]),
            _ => {}
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn team_parsing_repairs_bad_trees() {
        let t = r#"Here: {"agents": [
            {"id": "Lead", "name": "Lead", "role": "r", "task": "t", "parent": null},
            {"id": "a", "name": "A", "task": "ta", "parent": "lead"},
            {"id": "a", "name": "Dup", "task": "td", "parent": "ghost"},
            {"id": "b", "name": "B", "task": "", "parent": "lead"}
        ]}"#;
        let a = parse_team(t, "goal");
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].id, "lead");
        assert_eq!(a[2].id, "agent_3");
        assert_eq!(a[2].parent.as_deref(), Some("lead"));
        assert!(RunSpec { agents: a, ..Default::default() }.validate_agents().is_ok());
        assert_eq!(parse_team("no json", "g")[0].task, "g");
    }

    #[test]
    fn parent_ids_are_sanitised_like_ids() {
        // "Tech-Lead" as an id sanitises to "tech_lead"; a child naming it
        // as `parent` with the original casing/punctuation must resolve to
        // the same (non-root) agent instead of being treated as unknown and
        // reattached to the root.
        let t = r#"{"agents": [
            {"id": "ceo", "name": "CEO", "role": "r", "task": "lead the company", "parent": null},
            {"id": "Tech-Lead", "name": "Lead", "role": "r", "task": "lead the team", "parent": "ceo"},
            {"id": "dev", "name": "Dev", "role": "r", "task": "write code", "parent": "Tech-Lead"}
        ]}"#;
        let a = parse_team(t, "goal");
        assert_eq!(a.len(), 3);
        assert_eq!(a[0].id, "ceo");
        assert_eq!(a[1].id, "tech_lead");
        assert_eq!(a[1].parent.as_deref(), Some("ceo"));
        assert_eq!(a[2].id, "dev");
        assert_eq!(a[2].parent.as_deref(), Some("tech_lead"), "dev must stay under tech_lead, not fall to the root");
    }

    #[test]
    fn ranks_follow_the_decimal_system() {
        let mut agents = vec![AgentSpec {
            id: "r".into(),
            name: "r".into(),
            role: String::new(),
            task: "t".into(),
            parent: None,
            model: None,
        }];
        for i in 0..12 {
            agents.push(AgentSpec { id: format!("s{i}"), parent: Some("r".into()), ..agents[0].clone() });
        }
        let spec = RunSpec { agents, ..Default::default() };
        assert_eq!(rank(&spec, "s0"), "private");
        assert_eq!(rank(&spec, "r"), "captain (company commander)");
    }
}
