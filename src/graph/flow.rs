//! Flow orchestras: a left-to-right graph of agents that starts at the Lead
//! and ends at the Goal (see `storage::flowgraph` for the model).
//!
//! Scheduling is derived from checkpointed state and the step log, so
//! pause, resume and rewind work exactly as for single runs:
//!
//! - Nodes run one at a time in the graph's execution order; a node starts
//!   once all of its predecessors finished (`FlowState::done`).
//! - `finish` hands the node's report (and diff) to its successors.
//! - Review gates run before the flow moves on: on a `report` edge the
//!   upstream agent accepts the downstream agent's report or sends it back;
//!   on an `approve` edge the downstream agent approves the upstream work
//!   before starting, or sends it back. Pending reviews are computed from
//!   the step log, never stored.
//! - `consult`, `ask_suggestions` and `request_test` open a short side
//!   session with the other agent inside the caller's turn.
//! - Notebooks are shared notes: entries are the `notebook_write` steps.
//! - When every predecessor of the Goal finished, the Goal check runs
//!   (commands, then the configured judge). A failure goes back to the Lead
//!   with the reasons; the Lead reruns agents or fixes it and finishes again.
//! - Sending an agent back (review, rerun, a user message) reopens it and
//!   every node after it on the way to the Goal.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering;

use serde_json::{Value, json};

use super::approvals::{self, Decision};
use super::protocol::{self, Action};
use super::tools::{READ_ONLY, TESTING, ToolBox, ToolClass};
use super::{
    Engine, Inner, Stop, attach_latest_image, call_meta, identical_failures, image_of, note_images, note_lines, prompts,
    repeated_call, stuck_hint, verify_commands,
};
use crate::config::{Config, ModelRef, NetAccess};
use crate::error::{Error, Result};
use crate::fsutil::clip;
use crate::git::GitRepo;
use crate::providers::{CompletionRequest, Image, Message, ProviderId, ToolDef};
use crate::skills::Audience;
use crate::storage::{
    EdgeContext, FlowEdge, FlowGraph, FlowNode, GoalJudge, NodeKind, Phase, ReviewKind, Run, RunState, Step, StepKind,
    flowgraph::auto_layout,
};

/// Turns a consulted / reviewing / judging agent gets inside a side session.
const SIDE_TURNS: usize = 8;
/// A reviewer who can see images may look at them too.
const VIEWING: [&str; 12] = [
    "list_files",
    "list_dir",
    "read_file",
    "search",
    "web_search",
    "web_fetch",
    "browser",
    "process_output",
    "run_command",
    "start_process",
    "stop_process",
    "view_image",
];

/// What a Lead who delegates may use: look, search and read, never change.
const LEADING: [&str; 9] = [
    "list_files",
    "list_dir",
    "read_file",
    "search",
    "web_search",
    "web_fetch",
    "browser",
    "process_output",
    "view_image",
];

/// Own turns after which the Lead is told, every turn, to hand off.
const LEAD_PLAN_TURNS: usize = 6;

/// Whether `me` is a Lead who leaves the work to the agents: the flow has
/// agents (or may get some), and the Lead's `hands_on` is off. Such a Lead
/// cannot edit files or run commands; it investigates, hands off, and sends
/// agents back with `rerun`.
fn lead_delegates(g: &FlowGraph, me: &FlowNode) -> bool {
    me.kind == NodeKind::Lead
        && !g.lead_settings().hands_on
        && (g.lead_settings().dynamic_agents || g.nodes.iter().any(|n| matches!(n.kind, NodeKind::Agent | NodeKind::Parallel)))
}

/// What an external (ACP) agent is told, and refused, because of its
/// internet level. We can only refuse what it asks permission for.
fn acp_restrictions(net: NetAccess) -> (Vec<String>, String) {
    let mut deny = Vec::new();
    let mut text = String::new();
    if net < NetAccess::Docs {
        deny.push("fetch".to_string());
        text.push_str("\nInternet: you may not search the web or fetch URLs in this role");
        text.push_str(if net == NetAccess::Off { "." } else { " (local pages of this project are fine)." });
    }
    (deny, text)
}

/// Longest notebook entry.
const NOTE_CHARS: usize = 2_000;

const DESIGNER: &str = r##"You design a team of AI agents (a "flow") for a software task. The user reviews your design on a canvas, then it runs.

# How a flow runs
- Work flows left to right in ONE line: Lead -> agents -> Goal. Every node has at most one edge in and one edge out. Agents act one at a time on one shared working tree; when one finishes, its report and the diff of its changes go to the next.
- The Lead investigates and plans only. With agents after it, it can read, search and look things up, but it cannot edit files or run commands: it hands the work over in its finish report, and when the Goal check fails it sends the responsible agent back with a precise instruction. So the Lead's task says what to find out and how to brief the team, never what to implement. Every piece of work, even a one-line change, belongs to an agent.
- Every agent follows a role playbook picked by its "archetype", plus shared team rules (stay in your lane, report with evidence, ask for help when stuck). An agent has about 25 turns per assignment: one agent per coherent piece of work that fits in that.
- The Goal checks the result against the success criteria; failures go back to the Lead.
- Every agent re-reads its context on every turn: each extra agent costs real tokens. Add an agent only when it does work nobody else in the line can do as well.

# Choosing the team (typical shapes, adapt them)
- Small, clear change: coder. Add a tester if behaviour must be checked beyond the coder's own tests.
- Feature: coder -> tester; researcher -> coder -> tester when the code or a library is unfamiliar; split coders by module only if the modules are truly separate.
- Bug: researcher (reproduce and locate, no edits) -> coder (fix plus a regression test); or one coder when the cause is obvious.
- Refactor: refactorer -> reviewer. Documentation only: docs. A question to answer: researcher (several independent questions: a Dispatcher with research branches).
- Add a reviewer for risky changes (security, data, concurrency, public interfaces, large diffs); add a docs writer only when user-facing behaviour or documented commands change.
- Each agent's task must stand on its own: what to deliver, which files or modules it owns, how to verify it, and what not to touch. Name concrete paths from the file list when you can.

# Archetypes (playbooks)
coder: implements and proves it works. tester: independent verification against the criteria; writes and runs tests, never fixes production code. visual-tester: checks what users see (rendering, layout at phone and desktop sizes, animation) with screenshots; use it for pages, games, charts and other UI, best with a model that can see images. reviewer: finds real defects in the change, ranks them, gives a verdict. researcher: finds out and reports with evidence, edits nothing. docs: makes documentation match the code. refactorer: behaviour-preserving cleanup in tested steps. Give EVERY agent one of these, or one of the extra role keys listed under "Workspace" when one fits better.

# Internet ("network" per agent)
Levels: "off" (no web tools), "local" (browser and screenshots for this project's pages and local servers only), "docs" (also web_search and web_fetch, for documentation and error lookups), "full" (also the browser on any URL). Give each agent the least it needs; the workspace maximum is stated below and cannot be exceeded. Usually: a researcher reading external docs "docs" ("full" only to inspect live websites); a coder "docs" when it uses an unfamiliar library or API, else "off", or "local" when it builds pages it should open; a tester or visual-tester of pages or web apps "local"; reviewer and refactorer "off"; docs writer "off". Omit "network" to use the workspace setting.

# Links (edge "caps"), only where they pay off
- consult: either agent may ask the other a question. Put it between an agent and an earlier one whose knowledge it may need (researcher -> coder).
- test: the first may ask the second to run checks while it works (coder -> tester).
- approve: before the second starts, it reviews the first's work and approves it or sends it back. Use it on the link into a reviewer, so the review becomes a gate.
- report: when the second finishes, its report goes back to the first, who accepts it or sends it back. Use it on Lead -> agent for high-stakes work the Lead must sign off.
- suggest: improvement ideas from the other agent; rarely worth it.
- "context": "report" sends only the report, without the diff: use it when the next agent works elsewhere and the diff would only cost tokens. Default "report_diff".
- Notebook nodes are shared memory (one per agent): use one when agents that are not neighbours share interfaces or decisions (both sides of an API, parallel branches). Give it a "task" saying what to write there.

# Parallel work
Only when parts are truly independent (separate modules, separate research questions): a Dispatcher node {"id": "par", "kind": "parallel", "name": "Dispatcher", "task": "how to judge and combine the results", "policy": "wait_all" | "first_success" | "manager", "work": "code" | "research"}. It sits in the line like an agent; each branch starts with an edge {"from": "par", "to": "x", "branch": true} and is a line of agents whose last agent has no edge out (it reports back to the Dispatcher). Branches run at the same time in separate copies of the project; code branches are merged afterwards. Never put two branches on the same files.

# Lead options
"lead": {"dynamic_agents": true, "max_dynamic": 2} on the Lead node lets it add agents while running: use it when the scope is unclear until the code has been read.

# The Goal
- "criteria": a detailed, verifiable checklist of 4-8 numbered items, each an observable fact plus how to check it (e.g. "1. index.html opens in a browser with no console errors (open it)", "2. the waves move over time (vertex heights change between frames)", "3. npm test passes (run it)"). Check behaviour and results, never exact source text (no required strings or grep patterns). Spell out what the user asked for in concrete terms; never just repeat the request.
- "verify_command": the project's test or build command when it has (or will have) one, else null.
- "judge": "agent" (the last agent asks itself; cheap), "judge" (an independent model; best when the last agent did the work itself, or for criteria about looks, UX or wording), or "command" (only when verify_command checks every criterion).

Reply with only a JSON object:
{"goal": {"criteria": "1. ...\n2. ...", "verify_command": null, "judge": "agent"},
 "nodes": [{"id": "lead", "kind": "lead", "name": "Lead", "role": "...", "task": "what to investigate and how to brief the team"},
           {"id": "dev", "kind": "agent", "name": "...", "role": "...", "archetype": "coder", "network": "off", "task": "..."},
           {"id": "qa", "kind": "agent", "name": "...", "role": "...", "archetype": "tester", "network": "local", "task": "..."},
           {"id": "notes", "kind": "notebook", "name": "...", "task": "what to record"}],
 "edges": [{"from": "lead", "to": "dev"}, {"from": "dev", "to": "qa", "caps": {"test": true}},
           {"from": "qa", "to": "goal"}, {"from": "notes", "to": "dev"}, {"from": "notes", "to": "qa"}]}
Ids are short snake_case. The Goal node is added for you as id "goal"; connect the last agent to it. Omit caps you do not use."##;

/// What the designer must know about the workspace: limits it cannot
/// exceed, the role keys it may use, and the key project docs.
pub(super) struct DesignContext {
    pub net: NetAccess,
    pub commands: bool,
    pub lead_sees: bool,
    /// (role key, what the playbook says it is for), beyond `ROLE_KEYS`.
    pub extra_roles: Vec<(String, String)>,
    pub docs: Vec<(String, String)>,
}

fn designer_input(goal: &str, files: &[String], c: &DesignContext) -> String {
    let mut s = format!("# Mission\n{goal}\n\n# Workspace\n");
    s.push_str(&format!(
        "- Internet maximum for agents: {} ({}).\n",
        c.net.as_str(),
        c.net.describe().split(':').next().unwrap_or("")
    ));
    if c.net < NetAccess::Docs {
        s.push_str("- No agent can search the web: a researcher can only study the repository.\n");
    }
    s.push_str(if c.commands {
        "- Agents may run shell commands (tests, builds, servers).\n"
    } else {
        "- Agents may NOT run shell commands: nothing can be built or tested by running it, so prefer criteria that can be checked by reading or opening pages, and leave verify_command null.\n"
    });
    s.push_str(if c.lead_sees {
        "- The Lead's model can see images.\n"
    } else {
        "- The Lead's model is text only.\n"
    });
    if !c.extra_roles.is_empty() {
        s.push_str("- Extra role keys (archetypes) with playbooks in this workspace:\n");
        for (k, d) in &c.extra_roles {
            s.push_str(&format!("  - {k}: {}\n", clip(d, 160)));
        }
    }
    s.push_str(&prompts::file_overview(files, 200));
    for (name, body) in &c.docs {
        s.push_str(&format!("\n# {name}\n{}\n", clip(body, 3_000)));
    }
    s
}

impl Engine {
    /// Asks `lead` to design a flow for `goal`. Always returns a valid
    /// graph: if the reply cannot be used, the Lead alone.
    pub async fn propose_flow(&self, goal: &str, lead: &ModelRef) -> Result<FlowGraph> {
        let goal = goal.trim();
        if goal.is_empty() {
            return Err(Error::InvalidRequest("goal is empty".into()));
        }
        let ws = &self.inner.ws;
        let cfg = ws.config();
        let files = ws.repo.ls_files().await.unwrap_or_default();
        let mut docs = Vec::new();
        for name in ["README.md", "PROJECT_CONTEXT.md", "CONTRIBUTING.md"] {
            if let Ok(body) = tokio::fs::read_to_string(ws.root.join(name)).await {
                docs.push((name.to_string(), body));
            }
        }
        let mut extra_roles: Vec<(String, String)> = Vec::new();
        for sk in ws.skills.enabled_skills() {
            for r in sk.roles.iter().map(|r| r.to_lowercase()) {
                if r != "*" && !crate::storage::ROLE_KEYS.contains(&r.as_str()) && !extra_roles.iter().any(|(k, _)| *k == r) {
                    extra_roles.push((r, sk.description.clone()));
                }
            }
        }
        let ctx = DesignContext {
            net: cfg.limits.net(),
            commands: cfg.limits.allow_commands,
            lead_sees: ws.gateway.supports_vision(lead).await,
            extra_roles,
            docs,
        };
        let req = CompletionRequest {
            system: DESIGNER.into(),
            messages: vec![Message::user(designer_input(goal, &files, &ctx))],
            max_tokens: cfg.limits.max_output_tokens,
            ..Default::default()
        };
        let out = ws.gateway.complete(lead, &req, "team-design").await?;
        let roles: Vec<&str> = ctx.extra_roles.iter().map(|(k, _)| k.as_str()).collect();
        let mut g = parse_flow_with(&out.text, goal, &roles);
        // A level at or above the workspace's is the workspace's: show it as such.
        for n in &mut g.nodes {
            if n.network.is_some_and(|x| x >= ctx.net) {
                n.network = None;
            }
        }
        if !ctx.commands
            && let Some(gs) = g.nodes.iter_mut().find_map(|n| n.goal.as_mut())
        {
            gs.verify_command = None;
            if gs.judge == GoalJudge::Command {
                gs.judge = GoalJudge::Agent;
            }
        }
        Ok(g)
    }
}

const CRITERIA: &str = "You turn a software task's success criteria into a detailed, verifiable checklist for the \
agent that decides whether the work is done. Write 4-8 numbered items. Each is one observable fact about the result \
plus how to check it (open the page in a browser and read the console, run the program or its tests, call the endpoint). \
Check behaviour and results, never the exact text of the source: do not require particular strings, identifiers or \
grep patterns in files, which only invites gaming the check. Make vague wishes concrete (\"looks realistic\" -> which \
visual properties, \"works\" -> which behaviour). Keep the user's intent and language; do not add unrelated \
requirements. Reply with only the checklist.";

impl Engine {
    /// Rewrites `criteria` (or, if empty, the goal) as a detailed checklist.
    pub async fn detail_criteria(&self, goal: &str, criteria: &str, model: &ModelRef) -> Result<String> {
        let ws = &self.inner.ws;
        let files = ws.repo.ls_files().await.unwrap_or_default();
        let input = format!(
            "# Task\n{}\n\n# Current success criteria\n{}\n{}",
            goal.trim(),
            if criteria.trim().is_empty() { "(none yet)" } else { criteria.trim() },
            prompts::file_overview(&files, 120)
        );
        let req = CompletionRequest {
            system: CRITERIA.into(),
            messages: vec![Message::user(input)],
            max_tokens: ws.config().limits.max_output_tokens,
            ..Default::default()
        };
        let out = ws.gateway.complete(model, &req, "criteria").await?;
        let text = out.text.trim().trim_matches('`').trim().to_string();
        if text.is_empty() {
            return Err(Error::Provider("the model returned no criteria".into()));
        }
        Ok(text)
    }

    /// Replaces the flow of a run that is not executing. Recorded as a step
    /// (so a rewind undoes it); nodes whose definition or inputs changed
    /// run again, with everything after them.
    pub async fn edit_flow(&self, run_id: &str, mut graph: FlowGraph) -> Result<Run> {
        if self.is_active(run_id) {
            return Err(Error::Conflict("pause the run before editing its flow".into()));
        }
        let ws = self.workspace().clone();
        let run = ws.store.get_run(run_id).await?;
        let head = ws.store.head_step(run_id).await?;
        let state = head.as_ref().map(|h| h.state.clone()).unwrap_or_default();
        let old = current_graph(&run, &state)
            .ok_or_else(|| Error::Conflict("this run has no flow to edit (a removed hierarchical orchestra)".into()))?;
        graph.normalize();
        check_orchestra(&run.spec, &graph, run.verify_command.as_deref())?;
        auto_layout(&mut graph);

        let same = |a: &FlowNode, b: &FlowNode| {
            a.kind == b.kind
                && a.task == b.task
                && a.role == b.role
                && a.model == b.model
                && a.goal == b.goal
                && a.lead == b.lead
                && a.parallel == b.parallel
        };
        let inputs = |g: &FlowGraph, id: &str| {
            let mut v: Vec<String> =
                g.edges.iter().filter(|e| e.to == id).map(|e| format!("{:?}", (e.from.as_str(), e.caps, e.context, e.branch))).collect();
            v.sort();
            v
        };
        let mut changed = Vec::new();
        let mut added = Vec::new();
        for n in &graph.nodes {
            match old.node(&n.id) {
                None => added.push(n.id.clone()),
                Some(o) if !same(o, n) || inputs(&old, &n.id) != inputs(&graph, &n.id) => changed.push(n.id.clone()),
                _ => {}
            }
        }
        let removed: Vec<String> = old.nodes.iter().filter(|n| graph.node(&n.id).is_none()).map(|n| n.id.clone()).collect();
        let renamed: Vec<String> = graph
            .nodes
            .iter()
            .filter(|n| old.node(&n.id).is_some_and(|o| o.name != n.name || o.pos != n.pos) && !changed.contains(&n.id))
            .map(|n| n.id.clone())
            .collect();
        if changed.is_empty() && added.is_empty() && removed.is_empty() && renamed.is_empty() {
            return Ok(run);
        }

        let mut next = state.clone();
        next.flow.graph = Some(graph.clone());
        next.flow.patch = Default::default();
        next.flow.done.retain(|d| graph.node(d).is_some());
        // A changed branch agent runs again with its whole branch: from its
        // Dispatcher, which starts the branches afresh.
        let mut rerun: Vec<String> = changed.clone();
        for id in changed.iter().chain(&added).chain(&removed) {
            if let Some((d, _)) = graph.branch_of(id).or_else(|| old.branch_of(id)) {
                rerun.push(d);
            }
        }
        for id in &rerun {
            let down = graph.downstream(id);
            next.flow.done.retain(|d| !down.contains(d));
        }
        if let Some(p) = &next.flow.parallel
            && rerun.contains(&p.node)
        {
            // The branches restart; the Dispatcher's turn cleans up the old ones.
            next.flow.done.retain(|d| *d != p.node);
        }
        // A changed Goal is checked again.
        if let Some(goal) = graph.goal()
            && changed.contains(&goal.id)
        {
            next.flow.done.retain(|d| *d != goal.id);
            next.flow.goal_rounds = 0;
        }
        let reopened = next.flow.done.len() < state.flow.done.len() || !added.is_empty();
        if reopened && next.phase == Phase::Done {
            next.phase = Phase::Act;
        }
        next.failures = 0;
        next.escalate = false;
        let mut parts = Vec::new();
        let names = |ids: &[String]| ids.iter().map(|i| name_of(&graph, i)).collect::<Vec<_>>().join(", ");
        let old_names = |ids: &[String]| ids.iter().map(|i| name_of(&old, i)).collect::<Vec<_>>().join(", ");
        if !changed.is_empty() {
            parts.push(format!("changed {}", names(&changed)));
        }
        if !added.is_empty() {
            parts.push(format!("added {}", names(&added)));
        }
        if !removed.is_empty() {
            parts.push(format!("removed {}", old_names(&removed)));
        }
        if parts.is_empty() {
            parts.push("rearranged the canvas".into());
        }
        let text = format!("The user edited the flow: {}.", parts.join("; "));
        self.inner.record_edit(&run, &text, json!({ "agent": "", "graph_edit": { "changed": changed, "added": added, "removed": removed } }), &next).await?;
        // Work to redo: show it as paused (with a Resume) rather than done.
        let run = if next.phase != Phase::Done && (run.status == crate::storage::RunStatus::Done || reopened) {
            ws.store
                .set_run_status(run_id, crate::storage::RunStatus::Paused, Some("the flow was edited; resume to run the changed agents"))
                .await?
        } else {
            ws.store.get_run(run_id).await?
        };
        self.inner.emit(super::Event::Run { run: run.clone() });
        Ok(run)
    }
}

/// Words that make a criterion visual (checked by looking).
const VISUAL: &[&str] = &[
    "look", "visual", "colour", "color", "realistic", "beautiful", "screenshot", "render", "appear", "shows ", "image",
    "görün", "renk", "gerçekçi", "güzel", "görsel", "ekran",
];

impl Engine {
    /// Checks a flow before it runs: every model is called once (loads,
    /// takes native tools, sees images), and the design is checked for
    /// mistakes that only show later (a judge that cannot see visual
    /// criteria, tasks that rely on links that do not exist, …).
    pub async fn preflight(&self, spec: &crate::storage::RunSpec, verify: Option<&str>) -> Result<Value> {
        let cfg = self.workspace().config();
        let gw = self.workspace().gateway.clone();
        let mut errors: Vec<String> = Vec::new();
        let mut warnings: Vec<Value> = Vec::new();
        let Some(g) = spec.graph.clone() else { return Err(Error::InvalidRequest("no flow graph".into())) };
        let mut g = g;
        g.normalize();
        if let Err(e) = check_orchestra(spec, &g, verify) {
            errors.push(e.to_string());
        }
        // Models, each tested once.
        let mut tested: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        let mut nodes = serde_json::Map::new();
        let mut targets: Vec<(String, ModelRef)> = Vec::new();
        for n in g.nodes.iter().filter(|n| n.is_worker()) {
            match self.inner.node_model(&cfg, spec, &g, n) {
                Ok(m) => targets.push((n.id.clone(), m)),
                Err(e) => errors.push(e.to_string()),
            }
        }
        let gs = g.goal_settings();
        if g.goal().is_some()
            && gs.judge == GoalJudge::Judge
            && let Some(m) = gs.judge_model.clone().or_else(|| spec.model.clone())
        {
            targets.push(("goal".into(), m));
        }
        for (id, m) in &targets {
            let key = m.to_string();
            if !tested.contains_key(&key) {
                let r = self.test_model(m).await;
                tested.insert(key.clone(), r);
            }
            let mut r = tested[&key].clone();
            r["vision"] = json!(gw.supports_vision(m).await);
            r["model"] = json!(key);
            if r["ok"] == false {
                errors.push(format!("{}: {} cannot be used: {}", name_of(&g, id), key, r["error"].as_str().unwrap_or("")));
            } else if r["native_tools"] == false {
                warnings.push(json!({ "node": id, "text": format!("{}: {} has no native tool calling; it will use the slower, more error-prone text protocol", name_of(&g, id), key) }));
            }
            nodes.insert(id.clone(), r);
        }
        let sees = |id: &str| nodes.get(id).is_some_and(|r| r["vision"] == true);
        // A judge that cannot see, facing visual criteria.
        let criteria = gs.criteria.to_lowercase();
        let visual = VISUAL.iter().any(|w| criteria.contains(w));
        if visual && g.goal().is_some() && gs.judge != GoalJudge::Command {
            let judges: Vec<String> = match gs.judge {
                GoalJudge::Agent => g.goal().map(|goal| g.preds(&goal.id).iter().map(|s| s.to_string()).collect()).unwrap_or_default(),
                _ => vec!["goal".into()],
            };
            if judges.iter().all(|j| !sees(j)) {
                warnings.push(json!({ "node": g.goal().map(|n| n.id.clone()), "text": "The Goal has visual criteria but its judge cannot see images: make an agent that can see the last one before the Goal, or choose an independent judge with a vision model." }));
            }
        }
        if g.goal().is_some() && !gs.criteria.contains('\n') && gs.criteria.len() < 80 {
            warnings.push(json!({ "node": g.goal().map(|n| n.id.clone()), "text": "The Goal's success criteria are vague: make them a checklist (✨ Make detailed)." }));
        }
        // Tasks that rely on talking to someone without a link that allows it.
        for n in g.nodes.iter().filter(|n| n.is_worker()) {
            let task = n.task.to_lowercase();
            let l = links(&g, &n.id);
            let can_ask: Vec<&String> = l.consult.iter().chain(l.suggest.iter()).map(|(_, id)| id).collect();
            for other in g.nodes.iter().filter(|o| o.is_worker() && o.id != n.id) {
                let named = task.contains(&other.name.to_lowercase()) || task.contains(&other.id.to_lowercase());
                if named && !can_ask.contains(&&other.id) {
                    warnings.push(json!({ "node": n.id, "text": format!("{}'s task mentions {} but no Consult/Suggest link lets them talk: tick Consult on the link between them.", n.name, other.name) }));
                }
            }
            if (task.contains("consult") || task.contains("danış")) && can_ask.is_empty() {
                warnings.push(json!({ "node": n.id, "text": format!("{}'s task talks about consulting, but none of its links has Consult ticked.", n.name) }));
            }
            let visual = n.role_key() == "visual-tester" || task.contains("screenshot") || task.contains("image") || task.contains("görüntü");
            if visual && !sees(&n.id) && !can_ask.iter().any(|id| sees(id)) {
                warnings.push(json!({ "node": n.id, "text": format!("{} works with images but cannot see them and cannot consult anyone who can.", n.name) }));
            }
        }
        for nb in g.nodes.iter().filter(|n| n.kind == NodeKind::Notebook) {
            let users = g.edges.iter().filter(|e| e.other(&nb.id).is_some()).count();
            if users < 2 {
                warnings.push(json!({ "node": nb.id, "text": format!("Notebook {} is connected to {users} agent(s): it only helps when shared by two or more.", nb.name) }));
            }
        }
        Ok(json!({ "ok": errors.is_empty(), "errors": errors, "warnings": warnings, "nodes": nodes }))
    }

    /// One tiny call: does the model load and answer, and take native tools?
    async fn test_model(&self, m: &ModelRef) -> Value {
        if m.provider == ProviderId::Acp {
            return json!({ "ok": true, "native_tools": null, "note": "external agent (not tested)" });
        }
        let gw = self.workspace().gateway.clone();
        let req = CompletionRequest {
            system: "You are being tested. Call the finish tool with summary \"ok\".".into(),
            messages: vec![Message::user("Reply by calling finish with summary \"ok\".")],
            max_tokens: 64,
            tools: vec![super::tools::finish_def()],
            ..Default::default()
        };
        let t = std::time::Instant::now();
        match tokio::time::timeout(std::time::Duration::from_secs(240), gw.complete(m, &req, "preflight")).await {
            Ok(Ok(_)) => json!({ "ok": true, "native_tools": gw.native_tools(m), "ms": t.elapsed().as_millis() as u64 }),
            Ok(Err(e)) => json!({ "ok": false, "error": e.to_string() }),
            Err(_) => json!({ "ok": false, "error": "no answer within 4 minutes" }),
        }
    }

    /// Proposes lessons for future tasks from what went wrong in `run_id`:
    /// dead ends, failed Goal checks, hand-backs.
    pub async fn suggest_lessons(&self, run_id: &str) -> Result<Vec<String>> {
        let ws = self.workspace().clone();
        let run = ws.store.get_run(run_id).await?;
        let steps = ws.store.steps(run_id).await?;
        let mut facts = Vec::new();
        for s in &steps {
            for (_, e) in notes_in(s) {
                if s.meta["dead_end"] == true {
                    facts.push(format!("Dead end: {e}"));
                }
            }
            if s.meta["tool"] == "handback" {
                facts.push(format!("Stuck: {}", clip(&s.output, 800)));
            }
            if s.kind == StepKind::Verify && s.meta["ok"] == false {
                facts.push(format!("Goal check failed: {}", clip(s.meta["missing"].as_str().unwrap_or(&s.output), 800)));
            }
            if s.observation.as_deref().is_some_and(|o| o.contains("\n\nSTUCK:"))
                && let Some(e) = super::error_sig(s.observation.as_deref().unwrap_or(""), s.meta["ok"].as_bool())
            {
                facts.push(format!("Repeated error: {e}"));
            }
        }
        facts.dedup();
        if facts.is_empty() {
            return Ok(vec![]);
        }
        let model = run
            .spec
            .model
            .clone()
            .or_else(|| ws.config().routing.actor.clone())
            .ok_or_else(|| Error::InvalidRequest("no model to write lessons".into()))?;
        let (_, patch) = GitRepo::new(&run.worktree).diff_from(&run.base_commit).await.unwrap_or_default();
        let input = format!(
            "# Task\n{}\n\n# What went wrong\n{}\n\n# The final changes (excerpt)\n{}\n\n\
             Write at most 5 lessons that would have prevented these problems in future tasks in this project. \
             Each lesson is one line: \"When <situation>, <what to do>\" with the concrete fix (exact URLs, names, commands). \
             Only lessons that generalize beyond this task; skip anything you are not sure of. Reply with the lines only.",
            run.goal,
            facts.iter().take(30).map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n"),
            clip(&patch, 6_000)
        );
        let req = CompletionRequest {
            system: "You distil hard-won lessons from failed attempts of AI coding agents.".into(),
            messages: vec![Message::user(input)],
            max_tokens: 1_000,
            ..Default::default()
        };
        let out = ws.gateway.complete(&model, &req, run_id).await?;
        Ok(out
            .text
            .lines()
            .map(|l| l.trim().trim_start_matches(['-', '*', '•']).trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')').trim())
            .filter(|l| l.len() > 12)
            .take(5)
            .map(str::to_string)
            .collect())
    }

    /// Adds `lessons` to the project's lessons skill (created and enabled
    /// on first use), which every agent of later runs reads.
    pub fn save_lessons(&self, lessons: &[String]) -> Result<usize> {
        let reg = &self.workspace().skills;
        let path = reg.dir().join("project-lessons.yaml");
        let mut all: Vec<String> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_yaml::from_str::<crate::skills::model::Skill>(&t).ok())
            .map(|s| s.prompt_injection.lines().filter_map(|l| l.trim().strip_prefix("- ").map(str::to_string)).collect())
            .unwrap_or_default();
        for l in lessons.iter().map(|l| l.trim()).filter(|l| !l.is_empty()) {
            if !all.iter().any(|x| x == l) {
                all.push(l.to_string());
            }
        }
        let skill = crate::skills::model::Skill {
            name: "project-lessons".into(),
            version: "1.0.0".into(),
            kind: crate::skills::model::SkillType::SystemModifier,
            description: "Lessons learned in earlier tasks of this project.".into(),
            prompt_injection: format!(
                "Lessons learned in earlier tasks of this project (follow them):\n{}",
                all.iter().map(|l| format!("- {l}")).collect::<Vec<_>>().join("\n")
            ),
            priority: 30,
            conflicts_with: vec![],
            roles: vec![],
            tools: vec![],
            validator: None,
        };
        crate::fsutil::write_atomic(&path, serde_yaml::to_string(&skill)?.as_bytes(), false)?;
        reg.reload()?;
        reg.set_enabled("project-lessons", true)?;
        Ok(all.len())
    }
}

/// Rules every orchestra must satisfy (at creation and after edits).
pub fn check_orchestra(spec: &crate::storage::RunSpec, g: &FlowGraph, verify_command: Option<&str>) -> Result<()> {
    g.validate()?;
    if spec.model.is_none() && g.nodes.iter().any(|n| n.is_worker() && n.model.is_none()) {
        return Err(Error::InvalidRequest("choose a lead model for the orchestra".into()));
    }
    // A Dispatcher decides and resolves merge conflicts with tool calls: an
    // external agent (ACP) cannot do that for it.
    for n in g.nodes.iter().filter(|n| n.kind == NodeKind::Parallel) {
        let m = n.model.as_ref().or(spec.worker_model.as_ref()).or(spec.model.as_ref());
        if m.is_some_and(|m| m.provider == ProviderId::Acp) {
            return Err(Error::InvalidRequest(format!("{} needs a model, not an external agent: choose one in its settings", n.name)));
        }
    }
    let gs = g.goal_settings();
    let has_cmd = gs.verify_command.as_deref().is_some_and(|c| !c.trim().is_empty())
        || verify_command.is_some_and(|c| !c.trim().is_empty());
    if g.goal().is_some() && gs.judge == GoalJudge::Command && !has_cmd {
        return Err(Error::InvalidRequest("the Goal decides by command only, but no verification command is set".into()));
    }
    Ok(())
}

/// The flow as it stands at `state`: the user's latest edit (or the run's
/// original graph) with the Lead's runtime changes applied.
pub fn current_graph(run: &Run, state: &RunState) -> Option<FlowGraph> {
    state.flow.graph.as_ref().or(run.spec.graph.as_ref()).map(|g| g.patched(&state.flow.patch))
}

/// Notes a step wrote to notebooks: (notebook, entry).
pub fn notes_in(s: &Step) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = s.meta["notes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|n| Some((n["notebook"].as_str()?.to_string(), n["entry"].as_str()?.to_string())))
        .collect();
    let w = &s.meta["notebook_write"];
    if let (Some(nb), Some(e)) = (w["notebook"].as_str(), w["entry"].as_str()) {
        v.push((nb.to_string(), e.to_string()));
    }
    v
}

/// What re-orchestrating produced: a started run (full autonomy) or a
/// draft for the user to review.
pub enum Reorchestration {
    Started(Run),
    Draft { spec: crate::storage::RunSpec, base: Option<String>, verify: Option<String> },
}

impl Engine {
    /// Cancels `run_id` and has its Lead design a new flow for `goal`. The
    /// new run starts from the repository's HEAD, or with `continue_current`
    /// from the cancelled run's latest checkpoint. In full autonomy it
    /// starts right away; otherwise the draft is returned for approval.
    pub async fn reorchestrate(&self, run_id: &str, goal: &str, continue_current: bool) -> Result<Reorchestration> {
        let ws = self.workspace().clone();
        let old = ws.store.get_run(run_id).await?;
        if old.spec.mode != crate::storage::RunMode::Orchestra {
            return Err(Error::InvalidRequest("only orchestras can be re-orchestrated".into()));
        }
        let lead = old
            .spec
            .graph
            .as_ref()
            .and_then(|g| g.lead().and_then(|l| l.model.clone()))
            .or_else(|| old.spec.model.clone())
            .or_else(|| ws.config().routing.actor.clone())
            .ok_or_else(|| Error::InvalidRequest("no model for the Lead".into()))?;
        // Design first: if that fails, the old run is left as it was.
        let graph = self.propose_flow(goal, &lead).await?;
        let why = "cancelled: replaced by a new orchestra";
        if self.is_active(run_id) {
            // Stopped now rather than after its step, so the code the new
            // run continues from is final (the step in flight is dropped).
            self.kill(run_id).await?;
            let run = ws.store.set_run_status(run_id, crate::storage::RunStatus::Paused, Some(why)).await?;
            self.inner.emit(super::Event::Run { run });
        } else {
            self.cancel(run_id, why).await?;
        }
        let base = if continue_current { GitRepo::new(&old.worktree).head().await? } else { None };
        let spec = crate::storage::RunSpec {
            mode: crate::storage::RunMode::Orchestra,
            model: old.spec.model.clone().or(Some(lead)),
            worker_model: old.spec.worker_model.clone(),
            agents: vec![],
            autonomy: old.spec.autonomy,
            graph: Some(graph),
        };
        if old.spec.autonomy == crate::storage::Autonomy::Full {
            let run = self.create_run_at(goal, old.verify_command.as_deref(), spec, base.as_deref()).await?;
            return Ok(Reorchestration::Started(run));
        }
        Ok(Reorchestration::Draft { spec, base, verify: old.verify_command.clone() })
    }
}

impl Engine {
    /// Per worker node: its context window (tokens, when known), its
    /// transcript budget (chars), the prompt size of its last turn, and the
    /// lines its file tools added and removed.
    pub async fn node_stats(&self, run: &Run, g: &FlowGraph, steps: &[Step]) -> serde_json::Map<String, Value> {
        let cfg = self.workspace().config();
        let mut out = serde_json::Map::new();
        for n in g.nodes.iter().filter(|n| n.is_worker()) {
            let Ok(m) = self.inner.node_model(&cfg, &run.spec, g, n) else { continue };
            let window = match m.provider {
                ProviderId::Ollama => Some(cfg.ollama_num_ctx as u64),
                _ => None,
            };
            let mine = || steps.iter().filter(|s| s.kind == StepKind::Act && s.meta["agent"] == n.id.as_str());
            let last = mine().rev().find_map(|s| s.meta["prompt_tokens"].as_u64());
            let (added, removed) = mine().fold((0u64, 0u64), |(a, r), s| {
                (a + s.meta["lines"]["added"].as_u64().unwrap_or(0), r + s.meta["lines"]["removed"].as_u64().unwrap_or(0))
            });
            out.insert(n.id.clone(), json!({
                "window_tokens": window,
                "budget_chars": cfg.context_chars(&m),
                "last_prompt_tokens": last,
                "added": added,
                "removed": removed,
            }));
        }
        out
    }

    /// Nodes of `g` whose model accepts images (for the dashboard's badge).
    pub async fn vision_nodes(&self, run: &Run, g: &FlowGraph) -> Vec<String> {
        let cfg = self.workspace().config();
        let mut v = Vec::new();
        for n in g.nodes.iter().filter(|n| n.is_worker()) {
            if let Ok(m) = self.inner.node_model(&cfg, &run.spec, g, n)
                && self.workspace().gateway.supports_vision(&m).await
            {
                v.push(n.id.clone());
            }
        }
        v
    }
}

/// Lowercase, non-alphanumeric → `_`.
pub fn sanitize_id(s: &str) -> String {
    let id: String = s.trim().to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    id.trim_matches('_').to_string()
}

fn kind_of(s: &str) -> NodeKind {
    match s.trim().to_lowercase().as_str() {
        "lead" | "start" => NodeKind::Lead,
        "goal" | "end" => NodeKind::Goal,
        "notebook" | "memory" | "notes" => NodeKind::Notebook,
        "parallel" | "dispatcher" => NodeKind::Parallel,
        _ => NodeKind::Agent,
    }
}

/// The Lead alone, straight to the Goal.
pub fn minimal_flow(goal: &str) -> FlowGraph {
    let mut g = FlowGraph::default();
    g.nodes.push(FlowNode::new("lead", NodeKind::Lead, "Lead", goal));
    g.nodes.push(FlowNode::new("goal", NodeKind::Goal, "Goal", ""));
    g.edges.push(FlowEdge::new("lead", "goal"));
    if let Some(gs) = g.nodes[1].goal.as_mut() {
        gs.criteria = goal.to_string();
    }
    auto_layout(&mut g);
    g
}

/// Lenient flow parsing + repair: one Lead, one Goal, known endpoints, no
/// cycles, every agent between the Lead and the Goal.
pub fn parse_flow(text: &str, goal: &str) -> FlowGraph {
    parse_flow_with(text, goal, &[])
}

/// `parse_flow`, also accepting the workspace's own role keys `roles` as
/// archetypes.
pub fn parse_flow_with(text: &str, goal: &str, roles: &[&str]) -> FlowGraph {
    let Some(obj) =
        protocol::json_objects(text).into_iter().rev().find(|o| o.get("nodes").and_then(Value::as_array).is_some())
    else {
        return minimal_flow(goal);
    };
    let mut g = FlowGraph::default();
    let mut ids = HashSet::new();
    let mut rename = std::collections::HashMap::new();
    for (i, v) in obj["nodes"].as_array().into_iter().flatten().enumerate() {
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").trim().to_string();
        let mut kind = kind_of(&s("kind"));
        if kind == NodeKind::Goal {
            continue; // added below
        }
        if kind == NodeKind::Lead && g.lead().is_some() {
            kind = NodeKind::Agent;
        }
        let raw = s("id");
        let mut id = sanitize_id(&raw);
        if id.is_empty() || id == "goal" || ids.contains(&id) {
            id = format!("agent_{}", i + 1);
        }
        let task = s("task");
        if kind == NodeKind::Agent && task.is_empty() {
            continue;
        }
        ids.insert(id.clone());
        rename.insert(sanitize_id(&raw), id.clone());
        let name = if s("name").is_empty() { id.clone() } else { s("name") };
        let mut n = FlowNode::new(&id, kind, &name, &task);
        n.role = s("role");
        if kind == NodeKind::Agent {
            let arch = s("archetype").to_lowercase();
            let known = crate::storage::ROLE_KEYS.contains(&arch.as_str()) || roles.contains(&arch.as_str());
            if known && arch != "lead" && arch != "dispatcher" {
                n.archetype = arch;
            }
            // Spelled out, so the editor shows which playbook it got.
            if n.archetype.is_empty() {
                n.archetype = n.role_key();
            }
        }
        n.network = serde_json::from_value(json!(s("network"))).ok();
        if let Some(ls) = n.lead.as_mut() {
            let lv = v.get("lead").unwrap_or(v);
            ls.dynamic_agents = lv.get("dynamic_agents").and_then(Value::as_bool).unwrap_or(false);
            if let Some(m) = lv.get("max_dynamic").and_then(Value::as_u64) {
                ls.max_dynamic = m.clamp(1, 10) as u32;
            }
        }
        if let Some(ps) = n.parallel.as_mut() {
            ps.policy = serde_json::from_value(json!(s("policy"))).unwrap_or_default();
            ps.work = serde_json::from_value(json!(s("work"))).unwrap_or_default();
        }
        g.nodes.push(n);
    }
    if g.lead().is_none() {
        match g.nodes.iter_mut().find(|n| n.kind == NodeKind::Agent) {
            Some(n) => {
                n.kind = NodeKind::Lead;
                n.lead = Some(Default::default());
            }
            None => return minimal_flow(goal),
        }
    }
    let mut goal_node = FlowNode::new("goal", NodeKind::Goal, "Goal", "");
    if let Some(gs) = goal_node.goal.as_mut() {
        gs.criteria = obj
            .get("goal")
            .and_then(|v| v.get("criteria").and_then(Value::as_str).or(v.as_str()))
            .map(str::to_string)
            .unwrap_or_else(|| goal.to_string());
        gs.verify_command = obj
            .get("goal")
            .and_then(|v| v.get("verify_command"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|c| !c.is_empty() && *c != "null")
            .map(str::to_string);
        gs.judge = obj
            .get("goal")
            .and_then(|v| v.get("judge"))
            .and_then(|j| serde_json::from_value(j.clone()).ok())
            .unwrap_or_default();
        if gs.judge == GoalJudge::Command && gs.verify_command.is_none() {
            gs.judge = GoalJudge::Agent;
        }
    }
    g.nodes.push(goal_node);
    let resolve = |raw: &str| {
        let id = sanitize_id(raw);
        if id == "goal" || id == "end" { Some("goal".to_string()) } else { rename.get(&id).cloned() }
    };
    for v in obj.get("edges").and_then(Value::as_array).into_iter().flatten() {
        let (Some(from), Some(to)) = (
            v.get("from").and_then(Value::as_str).and_then(resolve),
            v.get("to").and_then(Value::as_str).and_then(resolve),
        ) else {
            continue;
        };
        let mut e = FlowEdge::new(&from, &to);
        e.branch = v.get("branch").and_then(Value::as_bool).unwrap_or(false)
            && g.node(&from).is_some_and(|n| n.kind == NodeKind::Parallel);
        if let Some(c) = v.get("caps") {
            let b = |k: &str| c.get(k).and_then(Value::as_bool).unwrap_or(false);
            e.caps = crate::storage::EdgeCaps {
                consult: b("consult"),
                suggest: b("suggest"),
                report: b("report"),
                test: b("test"),
                approve: b("approve"),
            };
        }
        if let Some(c) = v.get("context").and_then(|c| serde_json::from_value(c.clone()).ok()) {
            e.context = c;
        }
        let limit = |k: &str| v.get(k).and_then(Value::as_u64).map(|n| n.clamp(1, 10) as u32);
        if let Some(n) = limit("call_limit") {
            e.call_limit = n;
        }
        if let Some(n) = limit("loop_limit") {
            e.loop_limit = n;
        }
        // One notebook per agent: that is what the editor lets you choose.
        let agent = match (g.node(&e.from).map(|n| n.kind), g.node(&e.to).map(|n| n.kind)) {
            (Some(NodeKind::Notebook), _) => Some(e.to.clone()),
            (_, Some(NodeKind::Notebook)) => Some(e.from.clone()),
            _ => None,
        };
        if let Some(a) = agent
            && !g.notebooks_of(&a).is_empty()
        {
            continue;
        }
        // Keep an edge only if the graph stays valid-shaped with it.
        let mut trial = g.clone();
        trial.edges.push(e.clone());
        if edge_allowed(&trial, &e) {
            g.edges.push(e);
        }
    }
    repair(&mut g);
    g.normalize();
    auto_layout(&mut g);
    if g.validate().is_err() {
        return minimal_flow(goal);
    }
    g
}

/// Endpoint kinds are legal, no duplicate, one link per socket, and (for
/// flow edges) no cycle. `g` already contains `e`.
fn edge_allowed(g: &FlowGraph, e: &FlowEdge) -> bool {
    let (Some(a), Some(b)) = (g.node(&e.from).map(|n| n.kind), g.node(&e.to).map(|n| n.kind)) else { return false };
    if e.from == e.to || g.edges.iter().filter(|x| x.from == e.from && x.to == e.to).count() > 1 {
        return false;
    }
    match (a, b) {
        (NodeKind::Notebook, NodeKind::Notebook)
        | (NodeKind::Notebook, NodeKind::Goal | NodeKind::Parallel)
        | (NodeKind::Parallel, NodeKind::Notebook)
        | (NodeKind::Goal, _) => false,
        (NodeKind::Notebook, _) | (_, NodeKind::Notebook) => true,
        (_, NodeKind::Lead) => false,
        _ if e.branch && b != NodeKind::Agent => false,
        _ => {
            g.preds(&e.to).len() <= 1
                && g.flow_edges().filter(|x| x.from == e.from && !x.branch).count() <= 1
                && g.order().len() == g.nodes.iter().filter(|n| n.kind != NodeKind::Notebook).count()
        }
    }
}

/// Makes the main line one line: the Lead, then every node outside the
/// Dispatchers' branches in flow order, then the Goal. Existing links
/// between neighbours keep their settings; branches stay as they are, but
/// end at their last agent.
fn repair(g: &mut FlowGraph) {
    let mut in_branch: HashSet<String> = HashSet::new();
    for d in g.nodes.iter().filter(|n| n.kind == NodeKind::Parallel) {
        for h in g.branch_heads(&d.id) {
            in_branch.extend(g.branch_chain(&h));
        }
    }
    // Branch agents link only inside their branch.
    let inside = |id: &str| in_branch.contains(id);
    let snap = g.clone();
    g.edges.retain(|e| e.branch || !inside(&e.from) || inside(&e.to) || !snap.is_flow(e));
    // Dispatchers without branches are useless.
    let empty: Vec<String> =
        g.nodes.iter().filter(|n| n.kind == NodeKind::Parallel && g.branch_heads(&n.id).is_empty()).map(|n| n.id.clone()).collect();
    g.nodes.retain(|n| !empty.contains(&n.id));
    g.edges.retain(|e| !empty.contains(&e.from) && !empty.contains(&e.to));
    // The line as designed from the Lead on, then whatever was left out,
    // then the Goal.
    let lead = g.lead().map(|n| n.id.clone()).unwrap_or_default();
    let fits = |id: &str| !in_branch.contains(id) && g.node(id).is_some_and(|n| !matches!(n.kind, NodeKind::Goal | NodeKind::Notebook));
    let mut main = vec![lead.clone()];
    let mut cur = lead;
    while let Some(e) = g.flow_edges().find(|e| e.from == cur && !e.branch && fits(&e.to) && !main.contains(&e.to)) {
        cur = e.to.clone();
        main.push(cur.clone());
    }
    for id in g.order() {
        if fits(&id) && !main.contains(&id) {
            main.push(id);
        }
    }
    main.extend(g.goal().map(|n| n.id.clone()));
    let old = g.edges.clone();
    let snap = g.clone();
    g.edges.retain(|e| e.branch || !snap.is_flow(e) || inside(&e.from));
    for w in main.windows(2) {
        let e = old.iter().find(|e| e.from == w[0] && e.to == w[1] && !e.branch).cloned().unwrap_or_else(|| FlowEdge::new(&w[0], &w[1]));
        g.edges.push(e);
    }
}

/// A review gate that must run before the flow moves on.
#[derive(Debug, Clone)]
pub(super) struct Review {
    kind: ReviewKind,
    pub(super) reviewer: String,
    pub(super) subject: String,
    edge: FlowEdge,
}

fn agent_of<'a>(s: &'a Step, lead: &'a str) -> &'a str {
    s.meta["agent"].as_str().unwrap_or(lead)
}

fn is_finish(s: &Step, id: &str) -> bool {
    s.kind == StepKind::Act && s.meta["agent"] == id && s.meta["tool"] == "finish"
}

pub(super) fn last_finish(steps: &[Step], id: &str) -> Option<i64> {
    steps.iter().rev().find(|s| is_finish(s, id)).map(|s| s.seq)
}

/// The first review gate whose decision is still missing: for each finished
/// node (in flow order), reports owed upstream, then approvals owed by the
/// agents after it.
pub(super) fn pending_review(g: &FlowGraph, steps: &[Step], done: &[String]) -> Option<Review> {
    for subject in g.order() {
        if !done.contains(&subject) {
            continue;
        }
        let Some(fin) = last_finish(steps, &subject) else { continue };
        let decided = |e: &FlowEdge| {
            steps.iter().any(|s| {
                s.seq > fin && s.meta["review"]["edge"] == e.id.as_str() && s.meta["review"]["subject"] == subject.as_str()
            })
        };
        let worker = |id: &str| g.node(id).is_some_and(FlowNode::is_worker);
        for e in g.flow_edges() {
            if e.to == subject && e.caps.report && worker(&e.from) && !decided(e) {
                return Some(Review { kind: ReviewKind::Report, reviewer: e.from.clone(), subject, edge: e.clone() });
            }
        }
        for e in g.flow_edges() {
            if e.from == subject && e.caps.approve && worker(&e.to) && !decided(e) {
                return Some(Review { kind: ReviewKind::Approve, reviewer: e.to.clone(), subject, edge: e.clone() });
            }
        }
    }
    None
}

/// Reopens `id` and every node after it (the Goal included).
fn reopen(g: &FlowGraph, state: &mut RunState, id: &str) {
    let down = g.downstream(id);
    state.flow.done.retain(|d| !down.contains(d));
    if state.phase == Phase::Done {
        state.phase = Phase::Act;
    }
}

/// A user message for a node that already finished reopens it (and what
/// follows it). Idempotent: once reopened, it no longer matches.
fn reopen_for_messages(g: &FlowGraph, steps: &[Step], state: &mut RunState, lead: &str) {
    for s in steps.iter().filter(|s| s.kind == StepKind::Inject) {
        let target = agent_of(s, lead);
        let last_act = steps.iter().rev().find(|x| x.kind == StepKind::Act && x.meta["agent"] == target).map(|x| x.seq);
        if last_act.is_none_or(|a| s.seq > a) && state.flow.done.iter().any(|d| d == target) {
            reopen(g, state, target);
        }
    }
}

/// The step after which `me`'s current assignment began: its last finish,
/// hand-back, instruction, user message, verdict on its work or flow edit.
fn assignment_boundary(steps: &[Step], me: &str) -> i64 {
    steps
        .iter()
        .rev()
        .find(|s| {
            let m = &s.meta;
            (m["agent"] == me && (m["tool"] == "finish" || m["tool"] == "handback"))
                || m["rerun"]["agent"] == me
                || m["review"]["subject"] == me
                || (s.kind == StepKind::Inject && (m["agent"] == me || m["graph_edit"].is_object()))
        })
        .map_or(-1, |s| s.seq)
}

/// The commit `me`'s current assignment started from.
fn assignment_start(me: &str, steps: &[Step], base: &str) -> String {
    let mut start = base.to_string();
    let mut prev = base.to_string();
    let mut open = false;
    // A branch agent's steps checkpoint its branch's worktree: follow only
    // that branch's steps.
    let branch = steps.iter().rev().find(|s| s.meta["agent"] == me && s.meta["branch"].is_object()).map(|s| s.meta["branch"]["head"].clone());
    for s in steps {
        if let Some(b) = &branch
            && s.meta["branch"]["head"] != *b
        {
            continue;
        }
        let commit = s.meta["branch"]["commit"].as_str().unwrap_or(&s.git_commit).to_string();
        if s.kind == StepKind::Act && s.meta["agent"] == me {
            if !open {
                start = prev.clone();
                open = true;
            }
            if s.meta["tool"] == "finish" {
                open = false;
            }
        }
        prev = commit;
    }
    if open { start } else { prev }
}

async fn node_changes(run: &Run, me: &str, steps: &[Step], patch_chars: usize) -> Option<Value> {
    let start = assignment_start(me, steps, &run.base_commit);
    let (stat, patch) = GitRepo::new(&run.worktree).diff_from(&start).await.ok()?;
    Some(json!({ "stat": stat.trim(), "patch": clip(&patch, patch_chars) }))
}

/// A finished node's report as the next node receives it.
pub(super) fn report_text(s: &Step, ctx: EdgeContext) -> String {
    let mut out = s.meta["summary"].as_str().unwrap_or(&s.output).to_string();
    let stat = s.meta["changes"]["stat"].as_str().unwrap_or("");
    if stat.is_empty() {
        if s.meta.get("changes").is_some() {
            out.push_str("\n\n(no file changes)");
        }
        return out;
    }
    out.push_str(&format!("\n\nFiles changed:\n{stat}"));
    let patch = s.meta["changes"]["patch"].as_str().unwrap_or("");
    if ctx == EdgeContext::ReportDiff && !patch.trim().is_empty() {
        out.push_str(&format!("\n\nDiff:\n```diff\n{patch}\n```"));
    }
    out
}

pub(super) fn name_of(g: &FlowGraph, id: &str) -> String {
    g.node(id).map(|n| n.name.clone()).unwrap_or_else(|| id.to_string())
}

/// Who `me` can talk to over which capability.
struct Links {
    consult: Vec<(FlowEdge, String)>,
    suggest: Vec<(FlowEdge, String)>,
    test: Vec<(FlowEdge, String)>,
    notebooks: Vec<String>,
}

fn links(g: &FlowGraph, me: &str) -> Links {
    let mut l = Links { consult: vec![], suggest: vec![], test: vec![], notebooks: vec![] };
    for (e, other) in g.links(me) {
        if !other.is_worker() {
            continue;
        }
        if e.caps.consult {
            l.consult.push((e.clone(), other.id.clone()));
        }
        if e.caps.suggest {
            l.suggest.push((e.clone(), other.id.clone()));
        }
        if e.caps.test && e.from == me {
            l.test.push((e.clone(), other.id.clone()));
        }
    }
    l.notebooks = g.notebooks_of(me).into_iter().map(|n| n.id.clone()).collect();
    l
}

pub(super) fn tool(name: &str, description: &str, props: Value, required: &[&str]) -> ToolDef {
    ToolDef {
        name: name.into(),
        description: description.into(),
        parameters: json!({ "type": "object", "properties": props, "required": required }),
    }
}

/// Flow-specific tools for `me` (descriptions for the prompt + schemas).
fn flow_tools(g: &FlowGraph, me: &FlowNode, review: Option<&Review>, dynamic_left: u32, sees: &HashSet<String>) -> (String, Vec<ToolDef>) {
    let mut text = String::new();
    let mut defs = Vec::new();
    let mut add = |line: String, def: ToolDef| {
        text.push_str(&line);
        defs.push(def);
    };
    if let Some(r) = review {
        let (accept, reject) = match r.kind {
            ReviewKind::Report => ("accept the report", "send them back with a precise instruction"),
            ReviewKind::Approve => ("approve the work and start your own", "send it back with the reasons"),
        };
        add(
            format!("- review {{\"accept\": bool, \"feedback\": string}}: your decision on {}'s work: {accept}, or {reject}.\n", name_of(g, &r.subject)),
            tool(
                "review",
                "Decide on the work under review: accept it, or send it back with precise feedback.",
                json!({ "accept": { "type": "boolean" }, "feedback": { "type": "string", "description": "why, and exactly what to change when sending it back" } }),
                &["accept", "feedback"],
            ),
        );
        return (text, defs);
    }
    let l = links(g, &me.id);
    let names = |v: &[(FlowEdge, String)]| v.iter().map(|(_, id)| format!("{id} ({})", name_of(g, id))).collect::<Vec<_>>().join(", ");
    let agent_prop = |v: &[(FlowEdge, String)]| {
        json!({ "type": "string", "description": format!("one of: {}", v.iter().map(|(_, id)| id.as_str()).collect::<Vec<_>>().join(", ")) })
    };
    let images_prop = json!({ "type": "array", "items": { "type": "string" }, "description": "repo-relative image paths (e.g. screenshots) to show them; only for agents who can see images" });
    let seers = |v: &[(FlowEdge, String)]| {
        let who: Vec<String> = v.iter().filter(|(_, id)| sees.contains(id)).map(|(_, id)| name_of(g, id)).collect();
        if who.is_empty() { String::new() } else { format!(" {} can see images: pass screenshots in images.", who.join(", ")) }
    };
    if !l.consult.is_empty() {
        add(
            format!("- consult {{\"agent\": string, \"question\": string, \"images\"?: [string]}}: ask {} a question; they answer from their role and the code (read-only).{}\n", names(&l.consult), seers(&l.consult)),
            tool("consult", "Ask another agent a question (optionally about images); they answer from their role and the code.", json!({ "agent": agent_prop(&l.consult), "question": { "type": "string" }, "images": images_prop.clone() }), &["agent", "question"]),
        );
    }
    if !l.suggest.is_empty() {
        add(
            format!("- ask_suggestions {{\"agent\": string, \"focus\": string, \"images\"?: [string]}}: ask {} for improvement suggestions on the current changes.{}\n", names(&l.suggest), seers(&l.suggest)),
            tool("ask_suggestions", "Ask another agent for improvement suggestions on the current changes (optionally on images).", json!({ "agent": agent_prop(&l.suggest), "focus": { "type": "string", "description": "what to look at" }, "images": images_prop }), &["agent"]),
        );
    }
    if !l.test.is_empty() {
        add(
            format!("- request_test {{\"agent\": string, \"what\": string}}: ask {} to test something; they run it and report pass/fail.\n", names(&l.test)),
            tool("request_test", "Ask another agent to test something; they run it and report pass or fail.", json!({ "agent": agent_prop(&l.test), "what": { "type": "string" } }), &["agent", "what"]),
        );
    }
    if !l.notebooks.is_empty() {
        let nb = l.notebooks.iter().map(|id| format!("{id} ({})", name_of(g, id))).collect::<Vec<_>>().join(", ");
        let nb_prop = json!({ "type": "string", "description": format!("one of: {}", l.notebooks.join(", ")) });
        add(
            format!("- notebook_write {{\"notebook\"?: string, \"entry\": string}}: add a short note to a shared notebook ({nb}) for the other agents connected to it: decisions, interfaces, gotchas, results.\n"),
            tool("notebook_write", "Add a short note to a shared notebook for the other agents connected to it.", json!({ "notebook": nb_prop.clone(), "entry": { "type": "string" } }), &["entry"]),
        );
        add(
            "- notebook_read {\"notebook\"?: string, \"query\"?: string}: read the notes (optionally only those containing query).\n".into(),
            tool("notebook_read", "Read the notes of a shared notebook.", json!({ "notebook": nb_prop, "query": { "type": "string" } }), &[]),
        );
    }
    if me.kind == NodeKind::Lead {
        add(
            "- rerun {\"agent\": string, \"instruction\": string}: send an agent back to work with a precise instruction; it and every agent after it run again.\n".into(),
            tool("rerun", "Send an agent back to work with a precise instruction; it and every agent after it run again.", json!({ "agent": { "type": "string" }, "instruction": { "type": "string" } }), &["agent", "instruction"]),
        );
        if g.lead_settings().dynamic_agents && dynamic_left > 0 {
            add(
                format!("- spawn_agent {{\"name\": string, \"role\": string, \"archetype\"?: string, \"network\"?: string, \"task\": string, \"after\"?: string, \"before\"?: string}}: add a new agent to the flow between two nodes (default: after you, before the Goal). {dynamic_left} more allowed.\n"),
                tool("spawn_agent", "Add a new agent to the flow between two nodes.", json!({
                    "name": { "type": "string" }, "role": { "type": "string" }, "task": { "type": "string" },
                    "archetype": { "type": "string", "description": "role playbook: coder, tester, reviewer, researcher, docs or refactorer" },
                    "network": { "type": "string", "description": "internet: off, local, docs or full (default: the workspace's)" },
                    "after": { "type": "string", "description": "node id it follows (default: lead)" },
                    "before": { "type": "string", "description": "node id it precedes (default: goal)" }
                }), &["name", "task"]),
            );
        }
        if g.lead_settings().dynamic_agents {
            add(
                "- retire_agent {\"agent\": string}: remove an agent from the flow; its neighbours are connected directly.\n".into(),
                tool("retire_agent", "Remove an agent from the flow; its neighbours are connected directly.", json!({ "agent": { "type": "string" } }), &["agent"]),
            );
        }
    }
    let l = links(g, &me.id);
    if l.notebooks.is_empty() {
        add(
            "- finish {\"summary\": string}: your part is done; the report (with the diff of your changes) goes to the next agents.\n".into(),
            tool("finish", "Your part is done; the report goes to the next agents.", json!({ "summary": { "type": "string", "description": "what you did, how you verified it, what the next agents must know" } }), &["summary"]),
        );
    } else {
        add(
            "- finish {\"summary\": string, \"notes\": string}: your part is done; the report (with the diff) goes to the next agents, and `notes` go to the shared notebook for everyone connected to it: facts they need (decisions, interfaces, file layout, commands, pitfalls). Required.\n".into(),
            tool("finish", "Your part is done; the report goes to the next agents and the notes to the shared notebook.", json!({
                "summary": { "type": "string", "description": "what you did, how you verified it, what the next agents must know" },
                "notes": { "type": "string", "description": "facts for the shared notebook: decisions, interfaces, file layout, commands, pitfalls" }
            }), &["summary", "notes"]),
        );
    }
    (text, defs)
}

/// The stable first message of `me`'s turns.
fn flow_pinned(
    run: &Run,
    g: &FlowGraph,
    me: &FlowNode,
    files: &[String],
    sees: &HashSet<String>,
    net: NetAccess,
    delegates: bool,
) -> prompts::Pinned {
    let shared = format!("# Mission (from the user)\n{}\n{}\n", run.goal.trim(), prompts::file_overview(files, 150));
    let mut s = format!(
        "# You\n{} (id {}). Role: {}\nYour task: {}\nInternet access: {}\n",
        me.name,
        me.id,
        if me.role.is_empty() { "-" } else { &me.role },
        if me.task.trim().is_empty() { run.goal.trim() } else { me.task.trim() },
        net.describe()
    );
    let list = |ids: Vec<&str>| ids.iter().map(|id| name_of(g, id)).collect::<Vec<_>>().join(", ");
    let preds = g.preds(&me.id);
    let succs = g.succs(&me.id);
    let goal = g.goal().map(|n| n.id.clone()).unwrap_or_default();
    if me.kind == NodeKind::Lead && delegates {
        s.push_str(&format!(
            "You are the Lead: the flow starts with you, and the work is NOT yours. You cannot edit files or run commands; the agents after you ({}) do that. \
             Understand the mission (read, search, look things up), then call finish with a handoff that gives each agent a clear piece of work: \
             what to produce, which files or modules it owns, how to verify it, and the decisions and constraints already settled. \
             If a Goal check fails you get the reasons: send the responsible agent back with rerun and a precise instruction, then finish again.\n",
            list(succs.clone())
        ));
    } else if me.kind == NodeKind::Lead {
        s.push_str(&format!(
            "You are the Lead: the flow starts with you. Plan the work, then call finish with a handoff report for the next agents ({}). \
             If a Goal check fails, you get the reasons: coordinate the fix (rerun agents with precise instructions, or fix it yourself), then finish again.\n",
            list(succs.clone())
        ));
    } else if me.kind == NodeKind::Parallel {
        s.push_str("You are a Dispatcher: your branches of agents work in parallel; you judge their reports and merge their work.\n");
    } else {
        let branch = g.branch_of(&me.id);
        let to = match (&branch, succs.is_empty()) {
            (Some((d, _)), true) => format!("{} (your branch ends with you)", name_of(g, d)),
            _ => list(succs.clone()),
        };
        s.push_str(&format!(
            "Your input comes from {}; when you finish, your report and the diff of your changes go to {to}. Do your task only, \
             then call finish with a short report: what you did, how you verified it, and what the next agents must know. Do not paste code into it.\n",
            list(preds.clone()),
        ));
        if let Some((d, head)) = &branch {
            let chain = g.branch_chain(head).iter().map(|id| name_of(g, id)).collect::<Vec<_>>().join(" → ");
            let research = g.node(d).is_some_and(|n| n.parallel_settings().work == crate::storage::ParallelWork::Research);
            s.push_str(&format!(
                "You work in a parallel branch ({chain}) started by {}: other branches work on other parts at the same time, \
                 each in its own copy of the project. Stay within your branch's task and files; {}\n",
                name_of(g, d),
                if research {
                    "this is research: put your findings in your report (and the shared notebook, if you have one); file changes made in this branch are discarded."
                } else {
                    "the Dispatcher merges the finished branches, so avoid editing files another branch owns."
                }
            ));
        }
    }
    if succs.contains(&goal.as_str()) {
        let gs = g.goal_settings();
        if !gs.criteria.trim().is_empty() {
            s.push_str(&format!("Your work goes to the Goal check. Success criteria:\n{}\n", gs.criteria.trim()));
        }
    }
    let l = links(g, &me.id);
    let mut helpers: Vec<(String, Vec<&str>)> = Vec::new();
    for (list, what) in [(&l.consult, "consult"), (&l.suggest, "ask_suggestions"), (&l.test, "request_test")] {
        for (_, id) in list {
            match helpers.iter_mut().find(|(h, _)| h == id) {
                Some((_, w)) => w.push(what),
                None => helpers.push((id.clone(), vec![what])),
            }
        }
    }
    if !helpers.is_empty() {
        s.push_str("\n# Who can help you\nWhen you are stuck, get help from them instead of guessing:\n");
        for (id, how) in &helpers {
            let n = g.node(id).expect("linked");
            let model = n
                .model
                .as_ref()
                .map(|m| m.to_string())
                .unwrap_or_else(|| "the team's default model".into());
            s.push_str(&format!(
                "- {} ({id}): {}. Model: {model}.{} Task: {}. Ask with: {}.\n",
                n.name,
                if n.role.is_empty() { "agent" } else { &n.role },
                if sees.contains(id) { " Can see images: pass screenshots in images." } else { " Text only: cannot see images." },
                clip(n.task.trim(), 240),
                how.join(", ")
            ));
        }
    }
    let nbs = g.notebooks_of(&me.id);
    if !nbs.is_empty() {
        for nb in nbs {
            let others: Vec<String> = g
                .edges
                .iter()
                .filter_map(|e| e.other(&nb.id))
                .filter(|o| *o != me.id)
                .map(|o| name_of(g, o))
                .collect();
            s.push_str(&format!(
                "\nShared notebook \"{}\" ({}){}: the team's shared memory for this run. Notes the others write reach you here; \
                 when you finish, leave notes they need (decisions, interfaces, file layout, commands, pitfalls).\n",
                nb.name,
                nb.id,
                if others.is_empty() { String::new() } else { format!(", shared with {}", others.join(", ")) }
            ));
            if !nb.task.trim().is_empty() {
                s.push_str(&format!("Its purpose: {}\n", nb.task.trim()));
            }
        }
    }
    let team: Vec<String> = g
        .order()
        .iter()
        .filter_map(|id| g.node(id))
        .filter(|n| n.is_worker() && n.id != me.id)
        .map(|n| {
            let mut t = format!("{} [{}]: {}", n.name, n.id, if n.role.is_empty() { clip(&n.task, 120) } else { n.role.clone() });
            let key = n.role_key();
            let extra: Vec<String> = [
                (!key.is_empty() && n.kind == NodeKind::Agent).then(|| key.clone()),
                n.network.map(|x| format!("internet {}", x.as_str())),
            ]
            .into_iter()
            .flatten()
            .collect();
            if !extra.is_empty() {
                t.push_str(&format!(" ({})", extra.join(", ")));
            }
            t
        })
        .collect();
    if !team.is_empty() {
        s.push_str(&format!("\nThe team, in flow order: {}.\n", team.join("; ")));
    }
    prompts::Pinned { shared, own: s }
}

/// What changed since `me`'s last turn. Sent after the cache breakpoint.
fn flow_tail(
    g: &FlowGraph,
    me: &FlowNode,
    state: &RunState,
    steps: &[Step],
    review: Option<&Review>,
    name_status: &str,
    delegates: bool,
) -> String {
    let mut s = String::from("# Status now\n");
    let st: Vec<String> = g
        .order()
        .iter()
        .filter_map(|id| g.node(id))
        .filter(|n| n.is_worker())
        .map(|n| format!("{} {}", n.id, if state.flow.done.contains(&n.id) { "done" } else if n.id == me.id { "working (you)" } else { "waiting" }))
        .collect();
    s.push_str(&format!("Flow: {}.\n", st.join(", ")));
    if delegates && review.is_none() {
        // Who has done how much: a Lead that out-works the team is told so.
        let lead = g.lead().map(|n| n.id.as_str()).unwrap_or_default();
        let turns = |id: &str| steps.iter().filter(|x| x.kind == StepKind::Act && agent_of(x, lead) == id).count();
        let team: Vec<String> = g.order().iter().filter_map(|id| g.node(id)).filter(|n| n.is_worker()).map(|n| format!("{} {}", n.id, turns(&n.id))).collect();
        s.push_str(&format!("Turns taken so far: {}.\n", team.join(", ")));
        let since = assignment_boundary(steps, &me.id);
        let mine = steps.iter().filter(|x| x.seq > since && x.kind == StepKind::Act && agent_of(x, lead) == me.id).count();
        if mine >= LEAD_PLAN_TURNS {
            s.push_str(&format!(
                "You have used {mine} turns on your own. Stop investigating and call finish with your handoff now: the agents do the work, and they can look up details themselves.\n"
            ));
        }
    }
    if let Some(r) = review {
        s.push_str(&format!(
            "You are reviewing {}'s work ({}). Decide with the review tool.\n",
            name_of(g, &r.subject),
            if r.kind == ReviewKind::Report { "their report back to you" } else { "approval before you start" }
        ));
    }
    let l = links(g, &me.id);
    if !l.notebooks.is_empty() {
        let n = steps.iter().flat_map(notes_in).filter(|(nb, _)| l.notebooks.contains(nb)).count();
        s.push_str(&format!(
            "Shared notebooks: {} ({n} notes so far; notes from others appear in your conversation, add yours with finish's notes).\n",
            l.notebooks.join(", ")
        ));
    }
    s.push_str(&prompts::changes_note(name_status));
    s
}

/// Everything `me` has seen, in order, grouped by step.
fn flow_groups(g: &FlowGraph, me: &FlowNode, lead: &str, steps: &[Step], obs_chars: usize) -> Vec<prompts::Group> {
    let mut groups = Vec::new();
    let name = |id: &str| name_of(g, id);
    let preds = g.preds(&me.id);
    let siblings: HashSet<String> = preds
        .iter()
        .flat_map(|p| g.succs(p))
        .filter(|s| *s != me.id)
        .map(str::to_string)
        .collect();
    for s in steps {
        let agent = agent_of(s, lead);
        let mut msgs: Vec<Message> = Vec::new();
        match s.kind {
            StepKind::Act if agent == me.id => {
                let tool = s.meta["tool"].as_str().unwrap_or("action");
                let status = if s.meta["ok"].as_bool().unwrap_or(false) { "ok" } else { "error" };
                msgs.push(Message::assistant(s.output.clone()));
                msgs.push(Message::user(format!(
                    "Result of {tool} ({status}):\n{}",
                    clip(s.observation.as_deref().unwrap_or(""), obs_chars)
                )));
            }
            StepKind::Act => {
                if s.meta["tool"] == "dispatch" && g.edge(agent, &me.id).is_some_and(|e| e.branch) {
                    msgs.push(Message::user(format!(
                        "{} ({agent}) started parallel branches; yours starts with you. Handoff it passed on:\n{}",
                        name(agent),
                        s.meta["summary"].as_str().unwrap_or(&s.output)
                    )));
                }
                if s.meta["tool"] == "finish" {
                    if let Some(e) = g.edge(agent, &me.id).filter(|e| g.is_flow(e)) {
                        let mut t = format!("Handoff from {} ({agent}):\n{}", name(agent), report_text(s, e.context));
                        if e.caps.approve && me.is_worker() {
                            t.push_str("\n\nBefore you start, review this work: call review with accept=true to approve it, or accept=false with precise reasons to send it back.");
                        }
                        msgs.push(Message::user(t));
                    } else if let Some(e) = g.edge(&me.id, agent).filter(|e| e.caps.report) {
                        msgs.push(Message::user(format!(
                            "{} ({agent}) reports back to you:\n{}\n\nReview it: call review with accept=true, or accept=false with a precise instruction to send them back.",
                            name(agent),
                            report_text(s, e.context)
                        )));
                    } else if siblings.contains(agent) {
                        msgs.push(Message::user(format!(
                            "Teammate {} ({agent}) finished:\n{}",
                            name(agent),
                            report_text(s, EdgeContext::Report)
                        )));
                    }
                }
                let r = &s.meta["review"];
                if r["subject"] == me.id.as_str() {
                    let fb = r["feedback"].as_str().unwrap_or("");
                    msgs.push(Message::user(if r["accept"] == true {
                        format!("{} accepted your work.{}", name(agent), if fb.is_empty() { String::new() } else { format!(" Note: {fb}") })
                    } else {
                        format!("{} sent your work back:\n{fb}\nFix it, then finish again.", name(agent))
                    }));
                }
                let c = &s.meta["comm"];
                if c["with"] == me.id.as_str() {
                    msgs.push(Message::user(format!(
                        "Earlier, {} asked you ({}): {}\nYou answered: {}",
                        name(agent),
                        c["cap"].as_str().unwrap_or(""),
                        clip(c["request"].as_str().unwrap_or(""), 1_500),
                        clip(c["response"].as_str().unwrap_or(""), 3_000)
                    )));
                }
                let mine: Vec<String> = g.notebooks_of(&me.id).into_iter().map(|n| n.id.clone()).collect();
                for (nb, entry) in notes_in(s) {
                    if mine.contains(&nb) {
                        msgs.push(Message::user(format!("Shared note in {} from {}:\n{entry}", name(&nb), name(agent))));
                    }
                }
                if s.meta["tool"] == "handback" && me.kind == NodeKind::Lead && agent != me.id {
                    msgs.push(Message::user(format!(
                        "{}\n\nDecide how to unblock it: rerun {agent} with a precise instruction (the fix, or a different approach), \
                         add an agent that can do it if you may, or change the plan. Then finish so the flow continues.",
                        s.output
                    )));
                }
                if s.meta["rerun"]["agent"] == me.id.as_str() {
                    msgs.push(Message::user(format!(
                        "Instruction from {} (you run again):\n{}",
                        name(agent),
                        s.meta["rerun"]["instruction"].as_str().unwrap_or("")
                    )));
                }
            }
            StepKind::Inject if s.meta["graph_edit"].is_object() => {
                let e = &s.meta["graph_edit"];
                let hit = |k: &str| e[k].as_array().is_some_and(|a| a.iter().any(|x| x == me.id.as_str()));
                if hit("changed") || hit("added") {
                    msgs.push(Message::user(format!(
                        "The user changed your part of the flow. Your task now: {}",
                        if me.task.trim().is_empty() { "(as before)" } else { me.task.trim() }
                    )));
                } else if me.kind == NodeKind::Lead {
                    msgs.push(Message::user(s.output.clone()));
                }
            }
            StepKind::Inject if agent == me.id => msgs.push(Message::user(format!("Message from the user:\n{}", s.output))),
            StepKind::Verify if me.kind == NodeKind::Lead || s.meta["judged"] == me.id.as_str() => {
                msgs.push(Message::user(format!(
                    "Goal check:\n{}\n\n{}",
                    s.output,
                    clip(s.observation.as_deref().unwrap_or(""), obs_chars)
                )));
            }
            _ => {}
        }
        if !msgs.is_empty() {
            groups.push((s.seq, msgs));
        }
    }
    groups
}

/// Who acts next and whether as a reviewer (for the dashboard). Mirrors
/// `flow_turn`'s scheduling.
pub fn next_actor(g: &FlowGraph, steps: &[Step], state: &RunState) -> Option<(String, Option<String>)> {
    if state.phase == Phase::Done {
        return None;
    }
    if let Some(p) = &state.flow.parallel {
        return Some((p.node.clone(), None));
    }
    if let Some(r) = main_review(g, steps, &state.flow.done) {
        return Some((r.reviewer, Some(r.subject)));
    }
    next_main(g, &state.flow.done).map(|id| (id, None))
}

/// The next node of the main line whose inputs are done. Branch agents are
/// not in it: they run inside their Dispatcher's turn.
fn next_main(g: &FlowGraph, done: &[String]) -> Option<String> {
    g.order().into_iter().find(|id| {
        !done.contains(id) && g.branch_of(id).is_none() && g.preds(id).iter().all(|p| done.iter().any(|d| d == p))
    })
}

/// A pending review on the main line (branch reviews happen in the branch).
fn main_review(g: &FlowGraph, steps: &[Step], done: &[String]) -> Option<Review> {
    pending_review(g, steps, done).filter(|r| g.branch_of(&r.reviewer).is_none() && g.branch_of(&r.subject).is_none())
}

const NO_EVIDENCE: &str = "You have not checked anything yet, so you cannot know. Check the actual result first \
(open the page with the browser tool and read its console errors, run the program or its tests, read the key files), \
then call goal_verdict again with what you saw as evidence.";

const NO_EVIDENCE_REVIEW: &str = "You approved without checking anything. Look at the work first (read the changed files, \
run the tests, or open the page with the browser tool), then decide with review again.";

const NO_LOOK_FINISH: &str = "You have not looked at any image in this piece of work, so you cannot report what anything looks \
like. Take the screenshots your task asks for (for a page in this repo use url \"index.html\"; do not guess localhost ports), \
look at them, then finish with what you actually saw. If the page cannot be opened, finish saying exactly that and why.";

/// Phrases that report having seen something.
const SEEING_CLAIMS: &[&str] = &[
    "visually", "i can see", "i see ", "i saw", "looks good", "looks correct", "looks fine", "looks great", "look good",
    "is visible", "are visible", "visible in the", "the screenshot shows", "screenshot shows", "in the screenshot",
    "the frames show", "görsel olarak", "görüyorum", "gördüm", "görünüyor", "ekran görüntüsünde",
];

/// Phrases that admit a check could not be done.
const ADMITS_NOT_CHECKED: &[&str] = &[
    "could not", "couldn't", "cannot", "can't", "unable", "failed to", "did not load", "not checked", "not verified",
    "no image", "yapamadım", "açılamadı", "kontrol edilmedi", "doğrulanmadı",
];

/// The first phrase in `text` that claims something was seen.
fn claims_sight(text: &str) -> Option<&'static str> {
    let t = text.to_lowercase();
    SEEING_CLAIMS.iter().copied().find(|p| t.contains(p))
}

/// Claims of having looked oneself (a report of what a teammate saw is fine).
fn first_person_sight(text: &str) -> bool {
    let t = text.to_lowercase();
    ["i can see", "i see ", "i saw", "i looked", "verified visually", "visually verified", "visually confirmed", "görüyorum", "gördüm"]
        .iter()
        .any(|p| t.contains(p))
}

fn admits_not_checked(text: &str) -> bool {
    let t = text.to_lowercase();
    ADMITS_NOT_CHECKED.iter().any(|p| t.contains(p))
}

/// Where `id`'s current piece of work began: after its last accepted finish.
fn work_start(steps: &[Step], id: &str) -> i64 {
    steps.iter().rev().find(|s| is_finish(s, id) && s.meta["ok"] != false).map_or(-1, |s| s.seq)
}

/// Whether `id` was shown an image after step `since` (it needs a model
/// that can see: `sees`).
fn saw_image(steps: &[Step], id: &str, since: i64, sees: bool) -> bool {
    sees && steps.iter().any(|s| {
        s.seq > since
            && s.kind == StepKind::Act
            && s.meta["agent"] == id
            && s.meta["ok"] != false
            && s.meta["image"].is_string()
            && super::shows_images(s.meta["tool"].as_str().unwrap_or(""), true)
    })
}

/// Whether `id` asked a teammate something after `since` (a report may
/// then pass on what the teammate saw).
fn consulted(steps: &[Step], id: &str, since: i64) -> bool {
    steps.iter().any(|s| s.seq > since && s.meta["agent"] == id && s.meta["tool"] == "consult" && s.meta["ok"] != false)
}

/// A node whose job is looking at images.
fn visual_task(n: &FlowNode) -> bool {
    let t = format!("{} {}", n.role, n.task).to_lowercase();
    ["screenshot", "image", "visual", "vision", "look at", "görsel", "ekran görüntü"].iter().any(|w| t.contains(w))
}

/// Why a report claims more than the agent could know (the
/// `no-false-claims` skill), or `None` when it is honest enough.
fn false_claim(text: &str, saw: bool, sees: bool, asked: bool) -> Option<String> {
    if saw || asked || admits_not_checked(text) {
        return None;
    }
    let phrase = claims_sight(text)?;
    Some(format!(
        "Rejected by no-false-claims: your report says {phrase:?}, but you were not shown any image in this piece of work{}. \
         Report only what your tool results showed. For how something looks, either look at it now{}, or say plainly that it was not checked.",
        if sees { "" } else { " (and your model cannot see images)" },
        if sees { " (screenshot_page / screenshot_sequence)" } else { " through a teammate who can see (consult with images)" }
    ))
}

/// Verdict arguments with a JSON object that a model put inside a string
/// field (e.g. `{"text": "{\"achieved\": false, \"checklist\": [..]}"}`) merged in.
fn flatten_verdict(mut args: serde_json::Map<String, Value>) -> serde_json::Map<String, Value> {
    let nested: Vec<serde_json::Map<String, Value>> = args
        .values()
        .filter_map(Value::as_str)
        .flat_map(protocol::json_objects)
        .filter(|o| o.contains_key("checklist") || o.contains_key("achieved"))
        .collect();
    for o in nested {
        for (k, v) in o {
            args.entry(k).or_insert(v);
        }
    }
    args
}

/// Whether a verdict's arguments are a "yes".
fn positive(args: &serde_json::Map<String, Value>) -> bool {
    ["achieved", "accept", "passed"].iter().any(|k| args.get(*k).and_then(Value::as_bool) == Some(true))
}

/// What opening the run's index.html in a real browser showed.
struct PageCheck {
    /// Uncaught exceptions and failed loads: the page is broken.
    errors: Vec<String>,
    console: Vec<String>,
    /// Requests that failed or got an HTTP error (404s).
    failed: Vec<String>,
    /// Canvases, WebGL availability.
    notes: Vec<String>,
    text: String,
    /// Screenshot of the page (repo-relative), shown to a judge who can see.
    shot: Option<String>,
}

impl PageCheck {
    fn note(&self) -> String {
        let mut s = String::from("\n# The engine opened index.html in a headless browser\n");
        if self.console.is_empty() {
            s.push_str("Console: (no messages)\n");
        } else {
            s.push_str("Console:\n");
            for l in &self.console {
                s.push_str(&format!("  {l}\n"));
            }
        }
        if !self.failed.is_empty() {
            s.push_str("Failed requests:\n");
            for l in &self.failed {
                s.push_str(&format!("  {l}\n"));
            }
        }
        for n in &self.notes {
            s.push_str(&format!("Note: {n}\n"));
        }
        s.push_str(&format!("Rendered text: {}\n", if self.text.trim().is_empty() { "(none)".into() } else { clip(self.text.trim(), 1_500) }));
        s
    }
}

impl Inner {
    /// Opens index.html (when the run has one and a browser is available)
    /// the way a person would check a web page.
    async fn page_check(&self, cfg: &Config, run: &Run) -> Option<PageCheck> {
        let root = crate::fsutil::canonical(Path::new(&run.worktree)).ok()?;
        if !cfg.limits.allow_network || !root.join("index.html").is_file() || super::browser::find_browser().is_none() {
            return None;
        }
        let server = super::browser::LocalServer::start(root).await.ok()?;
        let url = reqwest::Url::parse(&format!("http://127.0.0.1:{}/index.html", server.port)).ok()?;
        self.log(&run.id, "opening index.html in a browser for the goal check");
        let rel = format!("{}/goal-check.png", super::tools::SHOTS_DIR);
        let shot_path = Path::new(&run.worktree).join(&rel);
        if let Some(parent) = shot_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::remove_file(&shot_path);
        let mut opts = super::browser::RenderOpts::new(4_000, (1280, 800));
        opts.shots = vec![shot_path];
        let r = match super::browser::render(&url, &opts).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(run = run.id, "page check failed: {e}");
                return None;
            }
        };
        let mut errors = r.errors();
        if let Some(e) = &r.load_error {
            errors.insert(0, format!("index.html did not load: {e}"));
        }
        Some(PageCheck { errors, console: r.console, failed: r.failed, notes: r.notes, text: r.text, shot: r.screenshots.first().map(|_| rel) })
    }
}

/// A side session's outcome.
pub(super) struct Side {
    /// The final tool's arguments (e.g. {"text": ..} or {"achieved": ..}).
    pub(super) args: serde_json::Map<String, Value>,
    pub(super) turns: Vec<Value>,
    pub(super) cost: f64,
}

impl Inner {
    fn node_model(&self, cfg: &Config, spec: &crate::storage::RunSpec, g: &FlowGraph, n: &FlowNode) -> Result<ModelRef> {
        let dynamic = n.dynamic.then(|| g.lead_settings().dynamic_model).flatten();
        let worker = (n.kind != NodeKind::Lead).then(|| spec.worker_model.clone()).flatten();
        n.model
            .clone()
            .or(dynamic)
            .or(worker)
            .or_else(|| spec.model.clone())
            .or_else(|| cfg.routing.actor.clone())
            .ok_or_else(|| Error::InvalidRequest(format!("no model for {}", n.name)))
    }

    pub(super) async fn flow_turn(&self, cfg: &Config, run: &Run, steps: &[Step], mut state: RunState) -> Result<Option<Stop>> {
        let g = current_graph(run, &state).ok_or_else(|| {
            Error::InvalidRequest(
                "this run uses the removed hierarchical orchestra and is read-only; start a new orchestra".into(),
            )
        })?;
        let lead = g.lead().map(|n| n.id.clone()).ok_or_else(|| Error::InvalidRequest("the flow has no Lead".into()))?;
        if state.phase == Phase::Plan {
            state.phase = Phase::Act;
        }
        reopen_for_messages(&g, steps, &mut state, &lead);
        if let Some(p) = state.flow.parallel.as_ref().map(|p| p.node.clone()) {
            return self.parallel_turn(cfg, run, state, &g, &p).await;
        }
        if let Some(r) = main_review(&g, steps, &state.flow.done) {
            let reviewer = r.reviewer.clone();
            return self.node_turn(cfg, run, steps, state, &g, &reviewer, Some(r)).await;
        }
        match next_main(&g, &state.flow.done) {
            None => Ok(Some(Stop::Done)),
            Some(id) if g.node(&id).is_some_and(|n| n.kind == NodeKind::Goal) => {
                self.goal_check(cfg, run, steps, state, &g, &lead).await
            }
            Some(id) if g.node(&id).is_some_and(|n| n.kind == NodeKind::Parallel) => {
                self.parallel_turn(cfg, run, state, &g, &id).await
            }
            Some(id) => self.node_turn(cfg, run, steps, state, &g, &id, None).await,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn node_turn(
        &self,
        cfg: &Config,
        run: &Run,
        steps: &[Step],
        mut state: RunState,
        g: &FlowGraph,
        id: &str,
        review: Option<Review>,
    ) -> Result<Option<Stop>> {
        let me = g.node(id).cloned().ok_or_else(|| Error::InvalidRequest(format!("unknown node {id}")))?;
        let lead = g.lead().map(|n| n.id.clone()).unwrap_or_default();
        if review.is_none()
            && let Some(stop) = self.hand_back_if_stuck(cfg, run, steps, &state, g, &me, &lead).await?
        {
            return Ok(stop);
        }
        let last_actor = steps.iter().rev().find(|s| s.kind == StepKind::Act).and_then(|s| s.meta["agent"].as_str());
        if last_actor != Some(me.id.as_str()) {
            state.failures = 0;
            state.escalate = false;
        }
        let base_model = self.node_model(cfg, &run.spec, g, &me)?;
        let (model, escalated) = match (&cfg.routing.escalation, state.escalate) {
            (Some(e), true) if base_model.provider != ProviderId::Acp => (e.clone(), true),
            _ => (base_model, false),
        };
        let wt = GitRepo::new(&run.worktree);
        let files = wt.ls_files_at(&run.base_commit).await?;
        let delegates = review.is_none() && lead_delegates(g, &me);
        let net = cfg.limits.net_for(me.network);
        let tail = flow_tail(g, &me, &state, steps, review.as_ref(), &wt.changed_names(&run.base_commit).await?, delegates);
        if model.provider == ProviderId::Acp {
            return self.acp_node_turn(cfg, run, steps, state, g, &me, &model, review, &files, &tail, delegates).await;
        }

        let role = me.role_key();
        let skills = self.ws.skills.compose_for("", Audience::Node(&role))?;
        let my_vision = self.ws.gateway.supports_vision(&model).await;
        let mut toolbox =
            ToolBox::new(Path::new(&run.worktree), &cfg.limits, skills.tools.clone())?.with_vision(my_vision).with_network(me.network);
        if review.is_some() {
            toolbox = toolbox.restricted(if my_vision { &VIEWING } else { TESTING });
        } else if delegates {
            toolbox = toolbox.restricted(&LEADING);
        }
        let dynamic_left = g.lead_settings().max_dynamic.saturating_sub(state.flow.patch.added_nodes.len() as u32);
        let mut sees = HashSet::new();
        for (_, n) in g.links(&me.id) {
            if n.is_worker()
                && let Ok(m) = self.node_model(cfg, &run.spec, g, n)
                && self.ws.gateway.supports_vision(&m).await
            {
                sees.insert(n.id.clone());
            }
        }
        let (extra_text, extra_defs) = flow_tools(g, &me, review.as_ref(), dynamic_left, &sees);
        let mut schemas = toolbox.schemas();
        schemas.extend(extra_defs);
        let system = format!("{}\n\n{}", prompts::actor_system(&format!("{}{extra_text}", toolbox.describe())), skills.system)
            .trim_end()
            .to_string();
        let pinned = flow_pinned(run, g, &me, &files, &sees, net, delegates);
        let groups = flow_groups(g, &me, &lead, steps, cfg.limits.tool_output_chars);
        let (messages, memory_cost) =
            self.fit(cfg, run, &me.id, &model, pinned, groups, cfg.context_chars(&model), &mut state).await;
        self.log(&run.id, match &review {
            Some(r) => format!("{} is reviewing {}", me.name, name_of(g, &r.subject)),
            None => format!("{} is working", me.name),
        });
        let messages = attach_latest_image(messages, steps, Some(&me.id), &toolbox);
        let req = CompletionRequest {
            system,
            messages,
            tail: Some(tail),
            cache_long: true,
            tools: schemas,
            sink: Some(self.live_sink(&run.id, Some(&me.id))),
            ..Default::default()
        };
        let out = self.call_req(cfg, run, &model, req).await?;

        let mut meta = call_meta(&out);
        meta["agent"] = json!(me.id);
        meta["escalated"] = json!(escalated);
        if let Some(r) = &review {
            meta["reviewing"] = json!({ "subject": r.subject, "edge": r.edge.id });
        }
        let mut next = state.clone();
        let mut extra_cost = memory_cost;
        let mut stop = None;
        let violations = skills.check_output(&out.text);
        let (observation, ok) = if !violations.is_empty() {
            let list: Vec<String> = violations.iter().map(|v| format!("{} (reply {})", v.skill, v.detail)).collect();
            (format!("Your reply was rejected by validator skills: {}. Redo it.", list.join("; ")), Some(false))
        } else {
            match protocol::parse_action(&out.text) {
                Err(e) => (
                    format!("Could not parse an action: {e}. Reply with a short note and exactly one tool call, e.g. {{\"tool\": \"read_file\", \"args\": {{\"path\": \"README.md\"}}}}."),
                    Some(false),
                ),
                Ok(a) => {
                    meta["tool"] = json!(a.tool);
                    let r = self
                        .flow_action(cfg, run, steps, g, &me, review.as_ref(), &toolbox, &a, &mut next, &mut meta, &out.text)
                        .await?;
                    extra_cost += r.cost;
                    stop = r.stop;
                    (r.observation, r.ok)
                }
            }
        };
        super::count_outcome(cfg, &mut meta, &mut next, ok);
        self.append(run, StepKind::Act, &out.text, Some(&observation), meta, &next, out.cost_usd + extra_cost).await?;
        if stop.is_some() {
            return Ok(stop);
        }
        if next.failures >= cfg.limits.max_failures {
            return Ok(Some(Stop::Paused(Some(format!(
                "{} failed {} turns in a row; message them with guidance or change their model, then resume",
                me.name, next.failures
            )))));
        }
        Ok(None)
    }

    /// Carries out one parsed action of a flow node.
    #[allow(clippy::too_many_arguments)]
    async fn flow_action(
        &self,
        cfg: &Config,
        run: &Run,
        steps: &[Step],
        g: &FlowGraph,
        me: &FlowNode,
        review: Option<&Review>,
        toolbox: &ToolBox,
        a: &Action,
        next: &mut RunState,
        meta: &mut Value,
        raw: &str,
    ) -> Result<Outcome> {
        let arg = |k: &str| a.str_arg(k).unwrap_or("").trim().to_string();
        match (a.tool.as_str(), review) {
            ("review", Some(r)) => {
                let accept = a.args.get("accept").and_then(Value::as_bool).unwrap_or(false);
                let feedback = arg("feedback");
                if accept {
                    // Approving without having looked at anything is a rubber
                    // stamp: send the reviewer to check once.
                    let fin = last_finish(steps, &r.subject).unwrap_or(-1);
                    let this_review = |s: &&Step| s.seq > fin && s.meta["agent"] == me.id.as_str() && s.meta["reviewing"]["edge"] == r.edge.id.as_str();
                    let looked = steps.iter().filter(this_review).any(|s| {
                        let t = s.meta["tool"].as_str().unwrap_or("");
                        !matches!(t, "review" | "") && s.meta["ok"] != false
                    });
                    let nudged = steps.iter().filter(this_review).any(|s| s.meta["evidence_nudge"] == true);
                    if !looked && !nudged {
                        meta["evidence_nudge"] = json!(true);
                        return Ok(Outcome {
                            observation: NO_EVIDENCE_REVIEW.into(),
                            ok: None,
                            cost: 0.0,
                            stop: None,
                        });
                    }
                    if self.ws.skills.compose("").is_ok_and(|c| c.honest) {
                        let saw = saw_image(steps, &me.id, fin, toolbox.sees());
                        let pushed = steps.iter().filter(this_review).any(|s| s.meta["false_claim"] == true);
                        if let Some(why) = false_claim(&feedback, saw, toolbox.sees(), consulted(steps, &me.id, fin))
                            && !pushed
                        {
                            meta["false_claim"] = json!(true);
                            return Ok(Outcome::fail(why));
                        }
                    }
                }
                if !accept && feedback.is_empty() {
                    return Ok(Outcome::fail("to send the work back, give precise feedback"));
                }
                meta["review"] = json!({ "kind": r.kind, "subject": r.subject, "edge": r.edge.id, "accept": accept, "feedback": feedback });
                if accept {
                    return Ok(Outcome::ok(format!("Accepted {}'s work.", name_of(g, &r.subject))));
                }
                reopen(g, next, &r.subject);
                let rejects = steps
                    .iter()
                    .filter(|s| s.meta["review"]["edge"] == r.edge.id.as_str() && s.meta["review"]["accept"] == false)
                    .count() as u32
                    + 1;
                let mut o = Outcome::ok(format!("Sent {}'s work back.", name_of(g, &r.subject)));
                if rejects >= r.edge.loop_limit.max(1) {
                    o.stop = Some(Stop::Paused(Some(format!(
                        "{} sent {}'s work back {rejects} times (the edge's limit); look at the disagreement, message them, then resume",
                        me.name,
                        name_of(g, &r.subject)
                    ))));
                }
                Ok(o)
            }
            ("finish", Some(_)) => Ok(Outcome::fail("you are reviewing: decide with the review tool instead of finish")),
            ("finish", None) => {
                let l = links(g, &me.id);
                let notes = arg("notes");
                if !l.notebooks.is_empty() {
                    let asked = steps.iter().any(|s| s.meta["agent"] == me.id.as_str() && s.meta["notes_missing"] == true);
                    if notes.is_empty() && !asked {
                        meta["notes_missing"] = json!(true);
                        return Ok(Outcome::fail(format!(
                            "finish needs \"notes\" for the shared notebook ({}): what the other agents connected to it should know. Call finish again with summary and notes.",
                            l.notebooks.join(", ")
                        )));
                    }
                    if !notes.is_empty() && !matches!(notes.to_lowercase().as_str(), "none" | "-" | "n/a") {
                        meta["notes"] = json!(l.notebooks.iter().map(|nb| json!({ "notebook": nb, "entry": clip(&notes, NOTE_CHARS) })).collect::<Vec<_>>());
                    }
                }
                let mut summary = a.str_arg("summary").or_else(|| a.str_arg("report")).unwrap_or("").trim().to_string();
                let report = format!("{summary}\n{notes}");
                let since = work_start(steps, &me.id);
                let sees = toolbox.sees();
                let saw = saw_image(steps, &me.id, since, sees);
                let pushed = |key: &str| steps.iter().filter(|s| s.seq > since && s.meta["agent"] == me.id.as_str() && s.meta[key] == true).count();
                // An agent whose job is looking must have looked. It may say
                // honestly that it could not; after two refusals its report
                // goes on, marked as unverified, so the flow never deadlocks.
                if sees && visual_task(me) && !saw && !admits_not_checked(&report) {
                    if pushed("no_look") < 2 {
                        meta["no_look"] = json!(true);
                        return Ok(Outcome::fail(NO_LOOK_FINISH));
                    }
                    summary = format!("[UNVERIFIED: {} did not look at any image] {summary}", me.name);
                }
                if self.ws.skills.compose("").is_ok_and(|c| c.honest) {
                    if let Some(why) = false_claim(&report, saw, sees, consulted(steps, &me.id, since))
                        && pushed("false_claim") < 2
                    {
                        meta["false_claim"] = json!(true);
                        return Ok(Outcome::fail(why));
                    }
                    let copied = steps.iter().any(|s| {
                        s.seq <= since && is_finish(s, &me.id) && s.meta["ok"] != false && s.meta["summary"].as_str() == Some(summary.as_str())
                    });
                    if copied && summary.len() > 40 && pushed("copied_report") < 1 {
                        meta["copied_report"] = json!(true);
                        return Ok(Outcome::fail(
                            "Rejected by no-false-claims: this report is word for word one you gave before. Say what you checked and found this time.",
                        ));
                    }
                }
                meta["summary"] = json!(if summary.is_empty() { raw.to_string() } else { summary });
                if let Some(c) = node_changes(run, &me.id, steps, cfg.limits.tool_output_chars).await {
                    meta["changes"] = c;
                }
                if !next.flow.done.contains(&me.id) {
                    next.flow.done.push(me.id.clone());
                }
                let to: Vec<String> = g.succs(&me.id).iter().map(|s| name_of(g, s)).collect();
                if to.is_empty() {
                    return Ok(Outcome::ok(match g.branch_of(&me.id) {
                        Some((d, _)) => format!("Report handed back to {}.", name_of(g, &d)),
                        None => "Report recorded.".to_string(),
                    }));
                }
                Ok(Outcome::ok(format!("Report handed to {}.", to.join(", "))))
            }
            ("consult" | "ask_suggestions" | "request_test", None) => self.communicate(cfg, run, steps, g, me, a, meta).await,
            ("notebook_write", None) => {
                let l = links(g, &me.id);
                let nb = if arg("notebook").is_empty() { l.notebooks.first().cloned().unwrap_or_default() } else { sanitize_id(&arg("notebook")) };
                if !l.notebooks.contains(&nb) {
                    return Ok(Outcome::fail(format!("no notebook {nb:?} is connected to you; yours: {}", l.notebooks.join(", "))));
                }
                let entry = arg("entry");
                if entry.is_empty() {
                    return Ok(Outcome::fail("notebook_write needs a non-empty \"entry\""));
                }
                meta["notes"] = json!([{ "notebook": nb, "entry": clip(&entry, NOTE_CHARS) }]);
                Ok(Outcome::ok(format!("Noted in {}.", name_of(g, &nb))))
            }
            ("notebook_read", None) => {
                let l = links(g, &me.id);
                let want = sanitize_id(&arg("notebook"));
                let q = arg("query").to_lowercase();
                let mut out = String::new();
                for s in steps {
                    for (nb, entry) in notes_in(s) {
                        if !l.notebooks.contains(&nb) || (!want.is_empty() && nb != want) || (!q.is_empty() && !entry.to_lowercase().contains(&q)) {
                            continue;
                        }
                        out.push_str(&format!("- [{} · {}] {entry}\n", name_of(g, &nb), name_of(g, agent_of(s, ""))));
                    }
                }
                meta["notebook_read"] = json!({ "notebook": if want.is_empty() { json!(l.notebooks) } else { json!(want) }, "query": q });
                Ok(Outcome::ok(if out.is_empty() { "No notes yet.".into() } else { out }))
            }
            ("rerun", None) if me.kind == NodeKind::Lead => {
                let target = sanitize_id(&arg("agent"));
                let instruction = arg("instruction");
                match g.node(&target) {
                    Some(n) if n.kind == NodeKind::Agent && !instruction.is_empty() => {
                        // A branch runs again as a whole, from its Dispatcher.
                        let from = g.branch_of(&target).map(|(d, _)| d).unwrap_or_else(|| target.clone());
                        reopen(g, next, &from);
                        meta["rerun"] = json!({ "agent": target, "instruction": instruction });
                        Ok(Outcome::ok(format!("{} will run again after you finish, then everything after it.", n.name)))
                    }
                    Some(_) if instruction.is_empty() => Ok(Outcome::fail("rerun needs an \"instruction\"")),
                    _ => Ok(Outcome::fail(format!("no agent {target:?}; agents: {}", g.nodes.iter().filter(|n| n.kind == NodeKind::Agent).map(|n| n.id.as_str()).collect::<Vec<_>>().join(", ")))),
                }
            }
            ("spawn_agent" | "retire_agent", None) if me.kind == NodeKind::Lead && g.lead_settings().dynamic_agents => {
                self.change_team(cfg, run, g, me, a, next, meta).await
            }
            (_, _) => {
                if let Some(why) = repeated_call(steps, Some(&me.id), a) {
                    meta["repeat"] = json!(true);
                    return Ok(Outcome::fail(why));
                }
                let (obs, ok) = self.gated(run, Some(&me.id), toolbox, a).await;
                if ok.is_none() {
                    meta["declined"] = json!(true);
                }
                note_images(meta, image_of(a, &obs, ok));
                note_lines(meta, &obs, ok);
                // The same call failing the same way more than three times is a
                // dead end the whole team should know about.
                if ok == Some(false) {
                    let before = identical_failures(steps, Some(&me.id), a, &obs);
                    let nbs: Vec<String> = g.notebooks_of(&me.id).into_iter().map(|n| n.id.clone()).collect();
                    if before == 3 && !nbs.is_empty() {
                        let entry = format!(
                            "Dead end (tried {} times by {}, same result every time): {} {} -> {}. Do not repeat it; find another way.",
                            before + 1,
                            me.name,
                            a.tool,
                            clip(&serde_json::to_string(&a.args).unwrap_or_default(), 300),
                            clip(obs.lines().find(|l| !l.trim().is_empty() && !l.starts_with('$')).unwrap_or(&obs), 300)
                        );
                        meta["notes"] = json!(nbs.iter().map(|nb| json!({ "notebook": nb, "entry": entry })).collect::<Vec<_>>());
                        meta["dead_end"] = json!(true);
                    }
                }
                let helpers: Vec<String> = links(g, &me.id).consult.iter().map(|(_, id)| name_of(g, id)).collect();
                let help = if helpers.is_empty() {
                    "web_search the exact error message".to_string()
                } else {
                    format!("consult {} (see Who can help you), or web_search the exact error message", helpers.join(" or "))
                };
                let obs = format!("{obs}{}", stuck_hint(steps, Some(&me.id), &obs, ok, &help));
                Ok(Outcome { observation: obs, ok, cost: 0.0, stop: None })
            }
        }
    }

    /// An agent that used up its turns on this assignment, or keeps hitting
    /// the same wall, stops and hands the problem to the Lead with what it
    /// tried; the Lead decides (rerun with an instruction, another agent, a
    /// different approach). A stuck Lead pauses the run for the user.
    #[allow(clippy::too_many_arguments)]
    async fn hand_back_if_stuck(
        &self,
        cfg: &Config,
        run: &Run,
        steps: &[Step],
        state: &RunState,
        g: &FlowGraph,
        me: &FlowNode,
        lead: &str,
    ) -> Result<Option<Option<Stop>>> {
        let start = assignment_boundary(steps, &me.id);
        let mine: Vec<&Step> =
            steps.iter().filter(|s| s.seq > start && s.kind == StepKind::Act && s.meta["agent"] == me.id.as_str()).collect();
        let stuck = mine.iter().filter(|s| s.observation.as_deref().is_some_and(|o| o.contains("\n\nSTUCK:"))).count();
        let repeats = mine.iter().filter(|s| s.meta["repeat"] == true).count();
        let limit = cfg.limits.node_turns.max(3) as usize;
        if mine.len() < limit && stuck < 3 && repeats < 3 {
            return Ok(None);
        }
        let why = if repeats >= 3 {
            format!("kept repeating a call that cannot give a new result ({repeats} times)")
        } else if stuck >= 3 {
            format!("kept hitting the same error ({stuck} warnings)")
        } else {
            format!("used {} turns without finishing", mine.len())
        };
        let mut errors: Vec<String> = Vec::new();
        for s in mine.iter().rev() {
            if let Some(e) = super::error_sig(s.observation.as_deref().unwrap_or(""), s.meta["ok"].as_bool())
                && !errors.contains(&e)
            {
                errors.push(e);
            }
            if errors.len() == 4 {
                break;
            }
        }
        let tried: Vec<String> = mine
            .iter()
            .rev()
            .take(8)
            .rev()
            .map(|s| format!("{}{}", s.meta["tool"].as_str().unwrap_or("?"), if s.meta["ok"] == false { " (failed)" } else { "" }))
            .collect();
        let text = format!(
            "{} {why} and hands the problem back.\nRecent errors:\n{}\nLast actions: {}",
            me.name,
            if errors.is_empty() { "(none recorded)".to_string() } else { errors.iter().map(|e| format!("- {e}")).collect::<Vec<_>>().join("\n") },
            tried.join(", ")
        );
        let mut next = RunState { failures: 0, escalate: false, ..state.clone() };
        let meta = json!({ "agent": me.id, "tool": "handback", "ok": true, "handback": { "why": why, "errors": errors, "turns": mine.len() } });
        // In a parallel branch there is no Lead to take it: the branch
        // ends, and its Dispatcher decides without it.
        if super::in_branch() {
            self.append(run, StepKind::Act, &text, None, meta, &next, 0.0).await?;
            return Ok(Some(Some(Stop::Paused(Some(format!("{} is stuck ({why})", me.name))))));
        }
        if me.id == lead {
            self.append(run, StepKind::Act, &text, None, meta, &next, 0.0).await?;
            return Ok(Some(Some(Stop::Paused(Some(format!(
                "the Lead is stuck ({why}); give it guidance or change its model, then resume"
            ))))));
        }
        next.flow.done.retain(|d| d != lead);
        self.log(&run.id, format!("{} is stuck and hands back to the Lead", me.name));
        self.append(run, StepKind::Act, &text, None, meta, &next, 0.0).await?;
        let _ = g;
        Ok(Some(None))
    }

    /// consult / ask_suggestions / request_test: a side session with the
    /// other agent inside this turn.
    #[allow(clippy::too_many_arguments)]
    async fn communicate(&self, cfg: &Config, run: &Run, steps: &[Step], g: &FlowGraph, me: &FlowNode, a: &Action, meta: &mut Value) -> Result<Outcome> {
        let l = links(g, &me.id);
        let (cap, list, key, tools): (&str, &Vec<(FlowEdge, String)>, &str, &[&'static str]) = match a.tool.as_str() {
            "consult" => ("consult", &l.consult, "question", READ_ONLY),
            "ask_suggestions" => ("suggest", &l.suggest, "focus", READ_ONLY),
            _ => ("test", &l.test, "what", TESTING),
        };
        let wanted = sanitize_id(a.str_arg("agent").or_else(|| a.str_arg("to")).unwrap_or(""));
        let Some((edge, other)) = list.iter().find(|(_, id)| *id == wanted || sanitize_id(&name_of(g, id)) == wanted) else {
            let ids: Vec<&str> = list.iter().map(|(_, id)| id.as_str()).collect();
            return Ok(Outcome::fail(format!("you cannot {cap} {wanted:?}; allowed: {}", ids.join(", "))));
        };
        let request = a.str_arg(key).or_else(|| a.str_arg("question")).or_else(|| a.str_arg("text")).unwrap_or("").trim().to_string();
        if request.is_empty() && cap != "suggest" {
            return Ok(Outcome::fail(format!("{} needs \"{key}\"", a.tool)));
        }
        // Images go only to agents whose model can see them.
        let paths: Vec<String> = match a.args.get("images") {
            Some(Value::Array(v)) => v.iter().filter_map(Value::as_str).map(str::to_string).collect(),
            Some(Value::String(p)) if !p.trim().is_empty() => vec![p.clone()],
            _ => vec![],
        };
        let mut images = Vec::new();
        if !paths.is_empty() {
            let callee = g.node(other).expect("linked");
            let sees = match self.node_model(cfg, &run.spec, g, callee) {
                Ok(m) => self.ws.gateway.supports_vision(&m).await,
                Err(_) => false,
            };
            if !sees {
                return Ok(Outcome::fail(format!(
                    "{} cannot see images (its model is text-only); describe what you need in words, or ask an agent that can see images",
                    callee.name
                )));
            }
            let tb = ToolBox::new(Path::new(&run.worktree), &cfg.limits, vec![])?;
            for p in &paths {
                match tb.load_image(p) {
                    Ok(img) => images.push(img),
                    Err(e) => return Ok(Outcome::fail(e)),
                }
            }
        }
        let used = steps.iter().filter(|s| s.meta["comm"]["edge"] == edge.id.as_str() && s.meta["comm"]["cap"] == cap).count() as u32;
        if used >= edge.call_limit {
            return Ok(Outcome::fail(format!("the {cap} limit on this link is used up ({used}/{}); decide yourself", edge.call_limit)));
        }
        let callee = g.node(other).cloned().expect("linked");
        let (ask, finish) = match cap {
            "consult" => (
                format!("{} ({}) asks you:\n{request}\n\nAnswer from your role, checking the code if needed. Call answer when ready.", me.name, me.id),
                tool("answer", "Your answer.", json!({ "text": { "type": "string" } }), &["text"]),
            ),
            "suggest" => (
                format!(
                    "{} ({}) asks for improvement suggestions on the current changes{}.\n\nLook at the changes (files listed below) and call answer with a short, prioritized list.",
                    me.name,
                    me.id,
                    if request.is_empty() { String::new() } else { format!(", focusing on: {request}") }
                ),
                tool("answer", "Your prioritized suggestions.", json!({ "text": { "type": "string" } }), &["text"]),
            ),
            _ => (
                format!("{} ({}) asks you to test:\n{request}\n\nRun what is needed, then call answer with passed=true/false and a short report of what you ran and saw.", me.name, me.id),
                tool("answer", "The test result.", json!({ "passed": { "type": "boolean" }, "text": { "type": "string" } }), &["passed", "text"]),
            ),
        };
        self.log(&run.id, format!("{} → {} ({cap})", me.name, callee.name));
        let ask = if images.is_empty() { ask } else { format!("{ask}\n\nThe images they refer to are attached: {}.", paths.join(", ")) };
        let side = self.side_session(cfg, run, g, &callee, &ask, tools, finish, None, images).await?;
        let text = side.args.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string();
        let passed = side.args.get("passed").and_then(Value::as_bool);
        meta["comm"] = json!({
            "edge": edge.id, "cap": cap, "with": callee.id, "request": request, "response": text, "images": paths,
            "passed": passed, "turns": side.turns, "cost": side.cost,
        });
        let head = match passed {
            Some(true) => format!("{} tested it: PASSED\n", callee.name),
            Some(false) => format!("{} tested it: FAILED\n", callee.name),
            None => format!("{} answers:\n", callee.name),
        };
        Ok(Outcome { observation: format!("{head}{text}"), ok: Some(true), cost: side.cost, stop: None })
    }

    /// A short, self-contained session of `callee` that ends with `finish`
    /// (whose arguments are returned). Tool calls go through approvals.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn side_session(
        &self,
        cfg: &Config,
        run: &Run,
        g: &FlowGraph,
        callee: &FlowNode,
        ask: &str,
        tools: &[&'static str],
        finish: ToolDef,
        evidence: Option<&str>,
        images: Vec<Image>,
    ) -> Result<Side> {
        self.side_session_n(cfg, run, g, callee, ask, tools, finish, evidence, images, SIDE_TURNS).await
    }

    /// `side_session` with up to `max_turns` turns.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn side_session_n(
        &self,
        cfg: &Config,
        run: &Run,
        g: &FlowGraph,
        callee: &FlowNode,
        ask: &str,
        tools: &[&'static str],
        finish: ToolDef,
        evidence: Option<&str>,
        images: Vec<Image>,
        max_turns: usize,
    ) -> Result<Side> {
        let model = self.node_model(cfg, &run.spec, g, callee)?;
        let sees = self.ws.gateway.supports_vision(&model).await;
        let wt = GitRepo::new(&run.worktree);
        let changed = wt.changed_names(&run.base_commit).await.unwrap_or_default();
        let who = format!(
            "# Mission\n{}\n\n# You\n{} (id {}). Role: {}\nYour task in the flow: {}\n\n{}\n\n# Request\n{ask}",
            run.goal.trim(),
            callee.name,
            callee.id,
            if callee.role.is_empty() { "-" } else { &callee.role },
            clip(&callee.task, 1_500),
            prompts::changes_note(&changed),
        );
        if model.provider == ProviderId::Acp {
            let prompt = format!(
                "{who}\n\nDo not change any files. End your reply with a JSON object: {}",
                if finish.parameters["properties"].get("passed").is_some() { r#"{"passed": true|false, "text": "..."}"# } else { r#"{"text": "..."}"# }
            );
            let out = self.external_turn(cfg, run, &model, &callee.name, prompt).await?;
            let mut args = protocol::json_objects(&out.text).into_iter().rev().find(|o| o.contains_key("text") || o.contains_key("passed")).unwrap_or_default();
            if !args.contains_key("text") {
                args.insert("text".into(), json!(out.text));
            }
            return Ok(Side { args, turns: vec![json!({ "external": super::external::tool_lines(&out) })], cost: 0.0 });
        }
        let mut allowed = tools.to_vec();
        if sees {
            allowed.push("view_image");
        }
        let toolbox =
            ToolBox::new(Path::new(&run.worktree), &cfg.limits, vec![])?.with_vision(sees).with_network(callee.network).restricted(&allowed);
        let mut schemas = toolbox.schemas();
        let finish_name = finish.name.clone();
        let finish_line = format!("- {} {}: {}\n", finish.name, finish.parameters["properties"], finish.description);
        schemas.push(finish.clone());
        let mut system = prompts::actor_system(&format!("{}{finish_line}", toolbox.describe()));
        if callee.kind == NodeKind::Parallel
            && let Ok(playbook) = self.ws.skills.compose_role("dispatcher")
            && !playbook.system.is_empty()
        {
            system = format!("{system}

{}", playbook.system);
        }
        let mut messages = vec![Message::user(who).with_images(if sees { images } else { vec![] })];
        let mut turns: Vec<Value> = Vec::new();
        let mut cost = 0.0;
        let pause = self.pause_flag(&run.id);
        let honest = evidence.is_some() && self.ws.skills.compose("").is_ok_and(|c| c.honest);
        for turn in 0..max_turns {
            if pause.load(Ordering::SeqCst) {
                return Err(Error::Provider(format!("{}'s side session was interrupted by a pause", callee.name)));
            }
            let last = turn + 1 == max_turns;
            if last {
                messages.push(Message::user(format!("This is your last turn: call {finish_name} now.")));
            }
            let req = CompletionRequest {
                system: system.clone(),
                messages: messages.clone(),
                tools: schemas.clone(),
                sink: Some(self.live_sink(&run.id, Some(&callee.id))),
                ..Default::default()
            };
            let out = self.call_req(cfg, run, &model, req).await?;
            cost += out.cost_usd;
            match protocol::parse_action(&out.text) {
                Ok(a) if (a.tool == finish_name || (a.tool == "finish" && last))
                    && evidence.is_some()
                    && !last
                    && positive(&a.args)
                    && !turns.iter().any(|t: &Value| t["tool"] != finish_name.as_str() && t["tool"] != "rejected") =>
                {
                    // A "yes" with nothing checked is not a judgement.
                    turns.push(json!({ "tool": "rejected", "why": "no evidence" }));
                    messages.push(Message::assistant(out.text.clone()));
                    messages.push(Message::user(evidence.unwrap_or_default().to_string()));
                }
                Ok(a) if a.tool == finish_name
                    && honest
                    && !sees
                    && !last
                    && !turns.iter().any(|t: &Value| t["why"] == "claims sight")
                    && first_person_sight(&serde_json::to_string(&a.args).unwrap_or_default()) =>
                {
                    // A judge that cannot see wrote as if it had looked.
                    turns.push(json!({ "tool": "rejected", "why": "claims sight" }));
                    messages.push(Message::assistant(out.text.clone()));
                    messages.push(Message::user(
                        "Rejected by no-false-claims: your model cannot see images, yet your verdict says you saw something. \
                         Base anything about looks only on what the request says someone who can see reported, or say it was not seen; \
                         then call the verdict again.",
                    ));
                }
                Ok(a) if a.tool == finish_name || (a.tool == "finish" && last) => {
                    let mut args = a.args.clone();
                    if !args.contains_key("text")
                        && let Some(s) = a.str_arg("summary").or_else(|| a.str_arg("answer"))
                    {
                        args.insert("text".into(), json!(s));
                    }
                    turns.push(json!({ "tool": finish_name }));
                    return Ok(Side { args, turns, cost });
                }
                Ok(a) => {
                    let (obs, ok) = self.gated(run, Some(&callee.id), &toolbox, &a).await;
                    turns.push(json!({ "tool": a.tool, "ok": ok, "args": clip(&serde_json::to_string(&a.args).unwrap_or_default(), 300) }));
                    messages.push(Message::assistant(out.text.clone()));
                    let mut result = Message::user(format!("Result of {}:\n{}", a.tool, clip(&obs, cfg.limits.tool_output_chars)));
                    if super::shows_images(&a.tool, sees) {
                        for p in image_of(&a, &obs, ok).into_iter().take(6) {
                            if let Ok(img) = toolbox.load_image(&p) {
                                result.images.push(img);
                            }
                        }
                    }
                    messages.push(result);
                }
                Err(e) => {
                    messages.push(Message::assistant(out.text.clone()));
                    messages.push(Message::user(format!("Could not parse a tool call ({e}). Call a tool, or {finish_name} to reply.")));
                    if last {
                        let mut args = serde_json::Map::new();
                        args.insert("text".into(), json!(out.text));
                        return Ok(Side { args, turns, cost });
                    }
                }
            }
        }
        let mut args = serde_json::Map::new();
        args.insert("text".into(), json!("(no answer within the turn limit)"));
        Ok(Side { args, turns, cost })
    }

    /// spawn_agent / retire_agent (the Lead, when dynamic agents are on).
    #[allow(clippy::too_many_arguments)]
    async fn change_team(&self, _cfg: &Config, run: &Run, g: &FlowGraph, me: &FlowNode, a: &Action, next: &mut RunState, meta: &mut Value) -> Result<Outcome> {
        let arg = |k: &str| a.str_arg(k).unwrap_or("").trim().to_string();
        // A new or removed agent is a team change: in semi-auto and manual
        // mode the user sees the draft and approves it first.
        let autonomy = self.ws.store.get_run(&run.id).await.map(|r| r.spec.autonomy).unwrap_or(run.spec.autonomy);
        if self.approvals.needed(&run.id, &a.tool, ToolClass::Exec, autonomy) {
            self.log(&run.id, format!("waiting for your approval: {}", a.tool));
            let flag = self.pause_flag(&run.id);
            let d = self
                .approvals
                .ask(&run.id, Some(&me.id), &a.tool, Value::Object(a.args.clone()), ToolClass::Exec, a.tool.clone(), &flag)
                .await;
            if !matches!(d, Decision::Allow) {
                meta["declined"] = json!(true);
                return Ok(Outcome { observation: approvals::refused_observation(&a.tool, &d), ok: None, cost: 0.0, stop: None });
            }
        }
        let mut patch = next.flow.patch.clone();
        let base = next.flow.graph.clone().or_else(|| run.spec.graph.clone()).unwrap_or_default();
        // Connects `from` → `to`. An edge with that id removed earlier is
        // un-removed instead: the base graph's original comes back, and a
        // new one is only added when there is no original (otherwise the
        // removal would swallow the new edge, or two would exist).
        let connect = |patch: &mut crate::storage::GraphPatch, from: &str, to: &str, branch: bool| {
            let e = if branch { FlowEdge::branch(from, to) } else { FlowEdge::new(from, to) };
            let was_removed = patch.removed_edges.contains(&e.id);
            patch.removed_edges.retain(|id| *id != e.id);
            if !(was_removed && base.edge_by_id(&e.id).is_some()) && !patch.added_edges.iter().any(|x| x.id == e.id) {
                patch.added_edges.push(e);
            }
        };
        let label;
        if a.tool == "spawn_agent" {
            let settings = g.lead_settings();
            if patch.added_nodes.len() as u32 >= settings.max_dynamic {
                return Ok(Outcome::fail(format!("the limit of {} added agents is reached", settings.max_dynamic)));
            }
            let (name, task) = (arg("name"), arg("task"));
            if name.is_empty() || task.is_empty() {
                return Ok(Outcome::fail("spawn_agent needs \"name\" and \"task\""));
            }
            let mut id = sanitize_id(&name);
            if id.is_empty() {
                id = "agent".into();
            }
            let stem = id.clone();
            let mut i = 2;
            // Ids of removed nodes stay taken: the patch keeps removing them.
            while g.node(&id).is_some() || patch.removed_nodes.contains(&id) || base.node(&id).is_some() {
                id = format!("{stem}_{i}");
                i += 1;
            }
            let after = if arg("after").is_empty() { me.id.clone() } else { sanitize_id(&arg("after")) };
            // One link per socket: the new agent goes into an existing link.
            let before = if arg("before").is_empty() {
                g.flow_edges().find(|e| e.from == after && !e.branch).map(|e| e.to.clone()).unwrap_or_default()
            } else {
                sanitize_id(&arg("before"))
            };
            let into_branch = g.edge(&after, &before).is_some_and(|e| e.branch);
            if !before.is_empty() && g.edge(&after, &before).is_none() {
                return Ok(Outcome::fail(format!("{after} does not link to {before}: a new agent goes between two linked nodes")));
            }
            let mut node = FlowNode::new(&id, NodeKind::Agent, &name, &task);
            node.role = arg("role");
            let arch = arg("archetype").to_lowercase();
            if crate::storage::ROLE_KEYS.contains(&arch.as_str()) && arch != "lead" && arch != "dispatcher" {
                node.archetype = arch;
            }
            node.network = serde_json::from_value(json!(arg("network"))).ok();
            node.dynamic = true;
            if let (Some(p), Some(q)) = (g.node(&after).and_then(|n| n.pos), g.node(&before).and_then(|n| n.pos)) {
                node.pos = Some([(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0 + 90.0]);
            }
            patch.added_nodes.push(node);
            if let Some(e) = g.edge(&after, &before) {
                patch.added_edges.retain(|x| x.id != e.id);
                if !patch.removed_edges.contains(&e.id) {
                    patch.removed_edges.push(e.id.clone());
                }
            }
            if into_branch {
                connect(&mut patch, &after, &id, true);
                connect(&mut patch, &id, &before, false);
            } else {
                connect(&mut patch, &after, &id, false);
                if !before.is_empty() {
                    connect(&mut patch, &id, &before, false);
                }
            }
            label = format!("added {name} ({id}) between {after} and {before}");
            meta["team_change"] = json!({ "action": "spawn", "agent": id, "after": after, "before": before });
        } else {
            let target = sanitize_id(&arg("agent"));
            match g.node(&target) {
                Some(n) if n.kind == NodeKind::Agent => {}
                _ => return Ok(Outcome::fail(format!("no agent {target:?} to retire"))),
            }
            let preds: Vec<String> = g.preds(&target).iter().map(|s| s.to_string()).collect();
            let succs: Vec<String> = g.succs(&target).iter().map(|s| s.to_string()).collect();
            patch.removed_nodes.push(target.clone());
            patch.added_nodes.retain(|n| n.id != target);
            patch.added_edges.retain(|e| e.from != target && e.to != target);
            for p in &preds {
                for s in &succs {
                    if g.edge(p, s).is_none() {
                        connect(&mut patch, p, s, g.edge(p, &target).is_some_and(|e| e.branch));
                    }
                }
            }
            label = format!("retired {target}");
            meta["team_change"] = json!({ "action": "retire", "agent": target });
        }
        let trial = base.patched(&patch);
        if let Err(e) = trial.validate() {
            return Ok(Outcome::fail(format!("that change would break the flow: {e}")));
        }
        next.flow.patch = patch;
        if a.tool == "spawn_agent" {
            let id = meta["team_change"]["agent"].as_str().unwrap_or("").to_string();
            reopen(&trial, next, &id);
        }
        next.flow.done.retain(|d| trial.node(d).is_some());
        self.log(&run.id, format!("Lead {label}"));
        Ok(Outcome::ok(format!("Team changed: {label}.")))
    }

    /// An external (ACP) agent as a node: one turn is the whole job.
    #[allow(clippy::too_many_arguments)]
    async fn acp_node_turn(
        &self,
        cfg: &Config,
        run: &Run,
        steps: &[Step],
        mut state: RunState,
        g: &FlowGraph,
        me: &FlowNode,
        model: &ModelRef,
        review: Option<Review>,
        files: &[String],
        tail: &str,
        delegates: bool,
    ) -> Result<Option<Stop>> {
        let lead = g.lead().map(|n| n.id.clone()).unwrap_or_default();
        let net = cfg.limits.net_for(me.network);
        // An external Lead has no rerun tool, so it delegates in its first
        // assignment only; after a failed Goal check it may fix things itself.
        let delegates = delegates && last_finish(steps, &me.id).is_none();
        let pinned = flow_pinned(run, g, me, files, &HashSet::new(), net, delegates);
        let groups = flow_groups(g, me, &lead, steps, cfg.limits.tool_output_chars).into_iter().map(|(_, m)| m).collect();
        let messages = prompts::windowed(pinned, groups, cfg.limits.cloud_context_chars);
        let suffix = match &review {
            Some(r) => format!(
                "\n\n---\nYou are reviewing {}'s work. Do not change files. End with a JSON object: {{\"accept\": true|false, \"feedback\": \"...\"}}.",
                name_of(g, &r.subject)
            ),
            None => "\n\n---\nWork directly in the current directory (the shared git working tree). Do only your task. End with a concise report for the next agents: what you did, how you verified it, and anything left open. The diff of your changes is attached automatically.".into(),
        };
        // External agents bring their own tools, so the role playbook and
        // the limits travel in the prompt (and in refused permission requests).
        let role = me.role_key();
        let playbook = self.ws.skills.compose_for("", Audience::Node(&role))?.system;
        let (mut deny, limits_note) = acp_restrictions(net);
        if delegates {
            deny.extend(["edit", "delete", "move", "execute"].map(String::from));
        }
        let prompt = format!(
            "{}{}\n\n{tail}{suffix}{limits_note}",
            if playbook.is_empty() { String::new() } else { format!("{playbook}\n\n") },
            super::external::as_prompt(&messages)
        );
        let out = self.external_turn_denying(cfg, run, model, &me.name, prompt, deny).await?;
        let mut meta = super::external::meta(model, &out);
        meta["agent"] = json!(me.id);
        let mut next = RunState { failures: 0, escalate: false, ..state.clone() };
        let text = if out.text.is_empty() { format!("({} finished without a report)", me.name) } else { out.text.clone() };
        let mut stop = out.cancelled.then_some(Stop::Paused(None));
        if !out.cancelled {
            match &review {
                Some(r) => {
                    let v = protocol::json_objects(&out.text).into_iter().rev().find(|o| o.contains_key("accept"));
                    let accept = v.as_ref().and_then(|o| o.get("accept")).and_then(Value::as_bool).unwrap_or(true);
                    let feedback = v.as_ref().and_then(|o| o.get("feedback")).and_then(Value::as_str).unwrap_or(&text).to_string();
                    meta["tool"] = json!("review");
                    meta["reviewing"] = json!({ "subject": r.subject, "edge": r.edge.id });
                    meta["review"] = json!({ "kind": r.kind, "subject": r.subject, "edge": r.edge.id, "accept": accept, "feedback": feedback });
                    if !accept {
                        reopen(g, &mut next, &r.subject);
                    }
                }
                None => {
                    meta["summary"] = json!(clip(&text, 6_000));
                    if let Some(c) = node_changes(run, &me.id, steps, cfg.limits.tool_output_chars).await {
                        meta["changes"] = c;
                    }
                    if !next.flow.done.contains(&me.id) {
                        next.flow.done.push(me.id.clone());
                    }
                }
            }
        } else {
            stop = Some(Stop::Paused(None));
        }
        state = next;
        let obs = super::external::tool_lines(&out);
        self.append(run, StepKind::Act, &text, Some(&obs), meta, &state, 0.0).await?;
        Ok(stop)
    }

    /// Runs the Goal check once every node before the Goal finished.
    async fn goal_check(&self, cfg: &Config, run: &Run, steps: &[Step], state: RunState, g: &FlowGraph, lead: &str) -> Result<Option<Stop>> {
        let goal = g.goal().cloned().expect("validated");
        let gs = g.goal_settings();
        self.log(&run.id, "checking the goal");
        // Commands first: the goal's own, the run's, and validator skills'.
        let skills = self.ws.skills.compose("")?;
        let mut commands = verify_commands(run, &skills);
        if let Some(c) = gs.verify_command.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
            commands.insert(0, ("goal".into(), c.to_string()));
        }
        let (cmd_ok, cmd_summary, cmd_details) = self.run_commands(cfg, run, &commands).await;
        let mut meta = json!({ "goal": true, "agent": goal.id, "judge": gs.judge, "commands": cmd_summary });
        let mut cost = 0.0;
        // A web page is checked the way a person would: opened in a browser.
        let page = if cmd_ok { self.page_check(cfg, run).await } else { None };
        if let Some(p) = &page {
            meta["page"] = json!({ "errors": p.errors, "console": p.console, "shot": p.shot });
        }
        let page_note = page.as_ref().map(|p| p.note()).unwrap_or_default();
        let (achieved, missing, judge_text) = if !cmd_ok {
            (false, format!("verification failed:\n{cmd_summary}"), String::new())
        } else if let Some(p) = page.as_ref().filter(|p| !p.errors.is_empty()) {
            (false, format!("index.html fails in a real browser:\n{}", p.errors.join("\n")), String::new())
        } else {
            match gs.judge {
                GoalJudge::Command => (true, String::new(), String::new()),
                GoalJudge::Agent | GoalJudge::Judge => {
                    let judge_node = match gs.judge {
                        GoalJudge::Agent => {
                            // The agent that finished last before the Goal.
                            let preds = g.preds(&goal.id);
                            let who = steps
                                .iter()
                                .rev()
                                .find(|s| s.meta["tool"] == "finish" && preds.iter().any(|p| s.meta["agent"] == *p))
                                .and_then(|s| s.meta["agent"].as_str())
                                .unwrap_or(preds.first().copied().unwrap_or(lead))
                                .to_string();
                            meta["judged"] = json!(who);
                            g.node(&who).cloned().expect("pred exists")
                        }
                        _ => {
                            let mut j = FlowNode::new("judge", NodeKind::Agent, "Judge", "Decide independently whether the mission's success criteria are met.");
                            j.role = "independent reviewer".into();
                            j.model = gs.judge_model.clone().or_else(|| run.spec.model.clone());
                            j
                        }
                    };
                    let shot = page.as_ref().and_then(|p| p.shot.clone());
                    let (sight_note, sight_cost) = self.sight_for_blind_judge(cfg, run, g, &judge_node, &gs.criteria, shot.as_deref()).await?;
                    cost += sight_cost;
                    let (stat, patch) = GitRepo::new(&run.worktree).diff_from(&run.base_commit).await.unwrap_or_default();
                    let ask = format!(
                        "Has the goal been reached? Judge the actual result, not the reports: agents often claim success that is not there.\n\n# Success criteria\n{}\n\n# Verification commands\n{}\n{page_note}{sight_note}\n# All changes so far\n{}\n```diff\n{}\n```\n\n\
                         Check the result yourself before deciding: open web pages with the browser tool (and read their console errors), \
                         run the program or its tests (start servers with start_process and stop them afterwards), read the key files, \
                         and check each criterion; you cannot change files. Be strict: achieved=true only if every criterion is met, and say in \
                         `evidence` what you checked and saw. Then call goal_verdict.",
                        if gs.criteria.trim().is_empty() { run.goal.trim() } else { gs.criteria.trim() },
                        if cmd_summary.is_empty() { "(none)".into() } else { cmd_summary.clone() },
                        stat.trim(),
                        clip(&patch, cfg.limits.tool_output_chars * 2)
                    );
                    let finish = tool(
                        "goal_verdict",
                        "Your verdict after checking: every success criterion with whether it is met and the evidence, then overall.",
                        json!({
                            "checklist": { "type": "array", "description": "one entry per success criterion", "items": { "type": "object", "properties": {
                                "criterion": { "type": "string" }, "met": { "type": "boolean" }, "evidence": { "type": "string", "description": "what you checked and saw" }
                            }, "required": ["criterion", "met", "evidence"] } },
                            "achieved": { "type": "boolean", "description": "true only if every criterion is met" },
                            "evidence": { "type": "string", "description": "overall: what you checked and what you saw" },
                            "missing": { "type": "string", "description": "what is missing or wrong (empty when achieved)" }
                        }),
                        &["checklist", "achieved", "evidence"],
                    );
                    let images: Vec<Image> = shot
                        .iter()
                        .filter_map(|p| ToolBox::new(Path::new(&run.worktree), &cfg.limits, vec![]).ok()?.load_image(p).ok())
                        .collect();
                    let side = self
                        .side_session(cfg, run, g, &judge_node, &ask, TESTING, finish, Some(NO_EVIDENCE), images)
                        .await?;
                    cost += side.cost;
                    let args = flatten_verdict(side.args.clone());
                    let checklist: Vec<(String, bool, String)> = args
                        .get("checklist")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .map(|c| {
                            (
                                c["criterion"].as_str().unwrap_or("").trim().to_string(),
                                c["met"].as_bool().unwrap_or(false),
                                c["evidence"].as_str().unwrap_or("").trim().to_string(),
                            )
                        })
                        .filter(|(c, _, _)| !c.is_empty())
                        .collect();
                    let unmet: Vec<&(String, bool, String)> = checklist.iter().filter(|c| !c.1).collect();
                    // The checklist decides when there is one (models often fill the
                    // overall flag inconsistently); every criterion must hold.
                    let achieved = if checklist.is_empty() {
                        args.get("achieved").and_then(Value::as_bool).unwrap_or(false)
                    } else {
                        unmet.is_empty()
                    };
                    let mut missing = if achieved {
                        String::new()
                    } else {
                        args.get("missing")
                            .and_then(Value::as_str)
                            .or_else(|| args.get("text").and_then(Value::as_str).filter(|_| checklist.is_empty()))
                            .unwrap_or("")
                            .trim()
                            .to_string()
                    };
                    for (c, _, e) in &unmet {
                        missing.push_str(&format!("\n- {c}{}", if e.is_empty() { String::new() } else { format!(" ({e})") }));
                    }
                    let missing = missing.trim().to_string();
                    let mut evidence = args.get("evidence").and_then(Value::as_str).unwrap_or("").trim().to_string();
                    for (c, met, e) in &checklist {
                        evidence.push_str(&format!("\n{} {c}: {e}", if *met { "✓" } else { "✗" }));
                    }
                    let evidence = evidence.trim().to_string();
                    meta["checklist"] = json!(checklist.iter().map(|(c, m, e)| json!({ "criterion": c, "met": m, "evidence": e })).collect::<Vec<_>>());
                    let checked: Vec<&str> =
                        side.turns.iter().filter_map(|t| t["tool"].as_str()).filter(|t| *t != "goal_verdict" && *t != "rejected").collect();
                    meta["turns"] = json!(side.turns);
                    meta["evidence"] = json!(evidence);
                    let how = if checked.is_empty() { "without checking anything".to_string() } else { format!("after {}", checked.join(", ")) };
                    let evidence_line = if evidence.is_empty() { String::new() } else { format!("\nEvidence: {evidence}") };
                    (
                        achieved,
                        missing.clone(),
                        format!("{} says: {} ({how}).{evidence_line}", judge_node.name, if achieved { "achieved" } else { "not achieved" }),
                    )
                }
            }
        };
        meta["ok"] = json!(achieved);
        meta["missing"] = json!(missing);
        let mut next = RunState { failures: 0, escalate: false, ..state };
        let summary = if achieved {
            next.phase = Phase::Done;
            next.flow.done.push(goal.id.clone());
            format!("Goal reached. {judge_text}").trim().to_string()
        } else {
            next.flow.goal_rounds += 1;
            next.flow.done.retain(|d| d != lead && *d != goal.id);
            format!("Goal NOT reached (check {}/{}). {judge_text}\nMissing:\n{missing}", next.flow.goal_rounds, gs.max_rounds.max(1))
        };
        let details = format!(
            "{cmd_details}\n\n{page_note}\n{}",
            if missing.is_empty() { String::new() } else { format!("Missing:\n{missing}") }
        )
        .trim()
        .to_string();
        self.append(run, StepKind::Verify, &summary, Some(&details), meta, &next, cost).await?;
        if !achieved && next.flow.goal_rounds >= gs.max_rounds.max(1) {
            return Ok(Some(Stop::Paused(Some(format!(
                "the goal was not reached after {} checks; the Lead has the reasons. Message the Lead or adjust the flow, then resume",
                next.flow.goal_rounds
            )))));
        }
        Ok(None)
    }

    /// When the goal is judged by looking but the judge cannot see: an agent
    /// of the flow that can see describes the engine's screenshot, and the
    /// judge is told to rely on that (or, with nobody who can see, not to
    /// claim anything about looks). Empty when no such help is needed.
    async fn sight_for_blind_judge(
        &self,
        cfg: &Config,
        run: &Run,
        g: &FlowGraph,
        judge: &FlowNode,
        criteria: &str,
        shot: Option<&str>,
    ) -> Result<(String, f64)> {
        let criteria = if criteria.trim().is_empty() { run.goal.trim() } else { criteria.trim() };
        let lower = criteria.to_lowercase();
        let Some(shot) = shot else { return Ok((String::new(), 0.0)) };
        let judge_sees = match self.node_model(cfg, &run.spec, g, judge) {
            Ok(m) => self.ws.gateway.supports_vision(&m).await,
            Err(_) => false,
        };
        if judge_sees || !VISUAL.iter().any(|w| lower.contains(w)) {
            return Ok((String::new(), 0.0));
        }
        let mut seer = None;
        for n in g.nodes.iter().filter(|n| n.is_worker() && n.id != judge.id) {
            if let Ok(m) = self.node_model(cfg, &run.spec, g, n)
                && m.provider != ProviderId::Acp
                && self.ws.gateway.supports_vision(&m).await
            {
                seer = Some(n.clone());
                break;
            }
        }
        let Some(seer) = seer else {
            return Ok((
                "\n# Nobody who can see looked at the page\nThe engine took a screenshot, but neither you nor any agent in this flow \
                 can see images. Do not claim anything about how the page looks: a visual criterion is met only with other hard \
                 evidence, and its evidence must say that nobody saw it.\n"
                    .into(),
                0.0,
            ));
        };
        let Ok(img) = ToolBox::new(Path::new(&run.worktree), &cfg.limits, vec![]).and_then(|t| t.load_image(shot).map_err(Error::InvalidRequest))
        else {
            return Ok((String::new(), 0.0));
        };
        self.log(&run.id, format!("{} is describing the goal screenshot for {}", seer.name, judge.name));
        let ask = format!(
            "The engine opened index.html in a headless browser; the screenshot is attached to this message. Answer only from what the \
             image shows. For each criterion below: met / not met / cannot tell, with one sentence of what is visible. Also say whether \
             the page (or a large part of it) is blank or one flat colour.\n\n# Criteria\n{criteria}\n\nThen call describe with your answers."
        );
        let describe = tool("describe", "What the screenshot shows, criterion by criterion.", json!({ "text": { "type": "string" } }), &["text"]);
        let side = self.side_session(cfg, run, g, &seer, &ask, &[], describe, None, vec![img]).await?;
        let text = side.args.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string();
        if text.is_empty() {
            return Ok((String::new(), side.cost));
        }
        Ok((
            format!(
                "\n# What {} (who can see) saw in the engine's screenshot of index.html\n{}\nYou cannot see images: for how the page \
                 looks rely only on this description, and do not add visual details of your own.\n",
                seer.name,
                clip(&text, 2_000)
            ),
            side.cost,
        ))
    }

    /// Runs `commands` in the worktree: (all passed, summary, details).
    pub(super) async fn run_commands(&self, cfg: &Config, run: &Run, commands: &[(String, String)]) -> (bool, String, String) {
        let timeout = std::time::Duration::from_secs(cfg.limits.command_timeout_secs.max(1));
        let per = super::tools::MAX_STORED_OUTPUT / commands.len().max(1);
        let (mut ok, mut summary, mut details) = (true, Vec::new(), String::new());
        for (label, cmd) in commands {
            let o = super::tools::run_shell(Path::new(&run.worktree), cmd, &[], timeout).await;
            let passed = o.success();
            ok &= passed;
            let status = match (o.timed_out, o.exit_code) {
                (true, _) => "timed out".to_string(),
                (_, Some(c)) => format!("exit {c}"),
                _ => "killed".to_string(),
            };
            summary.push(format!("{} [{label}] {cmd} ({status})", if passed { "PASS" } else { "FAIL" }));
            details.push_str(&format!("$ {cmd}\n[{status}]\n{}\n\n", clip(&o.output, per)));
        }
        (ok, summary.join("\n"), details.trim_end().to_string())
    }
}

struct Outcome {
    observation: String,
    ok: Option<bool>,
    cost: f64,
    stop: Option<Stop>,
}

impl Outcome {
    fn ok(s: impl Into<String>) -> Self {
        Self { observation: s.into(), ok: Some(true), cost: 0.0, stop: None }
    }
    fn fail(s: impl Into<String>) -> Self {
        Self { observation: s.into(), ok: Some(false), cost: 0.0, stop: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn designer_output_is_parsed_and_repaired() {
        let t = r#"Plan: {"goal": {"criteria": "- tests pass"}, "nodes": [
            {"id": "Lead", "kind": "lead", "name": "Lead", "task": "plan"},
            {"id": "Back-End", "kind": "agent", "name": "Backend", "task": "api"},
            {"id": "front", "kind": "agent", "name": "Frontend", "task": "ui"},
            {"id": "dup", "kind": "lead", "name": "Second lead", "task": "x"},
            {"id": "empty", "kind": "agent", "name": "No task", "task": ""},
            {"id": "mem", "kind": "notebook", "name": "API notes"}
        ], "edges": [
            {"from": "lead", "to": "back-end"}, {"from": "back_end", "to": "front", "caps": {"consult": true}},
            {"from": "front", "to": "back_end"}, {"from": "front", "to": "goal", "caps": {"approve": true}},
            {"from": "mem", "to": "back_end"}, {"from": "mem", "to": "front"}, {"from": "ghost", "to": "front"}
        ]}"#;
        let g = parse_flow(t, "build it");
        g.validate().unwrap();
        assert_eq!(g.lead().unwrap().id, "lead");
        assert_eq!(g.node("dup").unwrap().kind, NodeKind::Agent, "a second lead becomes an agent");
        assert!(g.node("empty").is_none());
        assert!(g.edge("back_end", "front").unwrap().caps.consult);
        assert!(g.edge("front", "back_end").is_none(), "the cycle-making edge is dropped");
        assert_eq!(g.goal_settings().criteria, "- tests pass");
        assert_eq!(g.notebooks_of("front")[0].id, "mem");
        // A second notebook for the same agent is dropped: one per agent.
        let two = r#"{"nodes": [{"id": "lead", "kind": "lead", "task": "t"}, {"id": "a", "kind": "agent", "task": "x"},
            {"id": "n1", "kind": "notebook", "name": "One"}, {"id": "n2", "kind": "notebook", "name": "Two"}],
            "edges": [{"from": "lead", "to": "a"}, {"from": "a", "to": "goal"}, {"from": "n1", "to": "a"}, {"from": "n2", "to": "a"}]}"#;
        let g2 = parse_flow(two, "x");
        assert_eq!(g2.notebooks_of("a").iter().map(|n| n.id.as_str()).collect::<Vec<_>>(), ["n1"]);
        // `dup` had no edges: repaired into the one line, before the Goal.
        assert_eq!(g.preds("dup"), ["front"]);
        assert_eq!(g.succs("dup"), ["goal"]);
        assert!(g.nodes.iter().all(|n| n.pos.is_some()));
    }

    #[test]
    fn designed_fan_outs_become_one_line_and_dispatchers_keep_their_branches() {
        let fan = r#"{"nodes": [{"id": "lead", "kind": "lead", "task": "t"}, {"id": "a", "kind": "agent", "task": "x"},
            {"id": "b", "kind": "agent", "task": "y"}],
            "edges": [{"from": "lead", "to": "a"}, {"from": "lead", "to": "b"}, {"from": "a", "to": "goal"}, {"from": "b", "to": "goal"}]}"#;
        let g = parse_flow(fan, "x");
        g.validate().unwrap();
        assert_eq!(g.order(), ["lead", "a", "b", "goal"]);
        let par = r#"{"nodes": [{"id": "lead", "kind": "lead", "task": "t"},
            {"id": "par", "kind": "parallel", "name": "Dispatcher", "task": "merge", "policy": "first_success", "work": "research"},
            {"id": "a", "kind": "agent", "task": "x"}, {"id": "b", "kind": "agent", "task": "y"}, {"id": "b2", "kind": "agent", "task": "z"}],
            "edges": [{"from": "lead", "to": "par"}, {"from": "par", "to": "a", "branch": true}, {"from": "par", "to": "b", "branch": true},
                      {"from": "b", "to": "b2"}, {"from": "b2", "to": "goal"}, {"from": "par", "to": "goal"}]}"#;
        let g = parse_flow(par, "x");
        g.validate().unwrap();
        let ps = g.node("par").unwrap().parallel_settings();
        assert_eq!((ps.policy, ps.work), (crate::storage::ParallelPolicy::FirstSuccess, crate::storage::ParallelWork::Research));
        assert_eq!(g.branch_heads("par").len(), 2);
        assert_eq!(g.branch_chain("b"), ["b", "b2"]);
        assert!(g.succs("b2").is_empty(), "a branch ends at its last agent");
        assert_eq!(g.succs("par"), ["a", "b", "goal"]);
    }

    #[test]
    fn claims_of_sight_need_an_image() {
        let lie = "No tests present; verified visually.";
        assert!(false_claim(lie, false, false, false).is_some_and(|w| w.contains("\"visually\"") && w.contains("cannot see")));
        assert!(false_claim(lie, true, true, false).is_none(), "an agent that looked may say so");
        assert!(false_claim(lie, false, false, true).is_none(), "a teammate who can see may have reported it");
        assert!(false_claim("Could not open the page, so the look is not verified visually.", false, false, false).is_none());
        assert!(false_claim("Static checks pass.", false, false, false).is_none());
        assert!(false_claim("Okyanus görünüyor.", false, true, false).is_some());
        assert!(first_person_sight(r#"{"evidence": "I can see waves"}"#));
        assert!(!first_person_sight(r#"{"evidence": "Eyes reported waves"}"#));
    }

    #[test]
    fn unusable_designs_fall_back_to_the_lead_alone() {
        let g = parse_flow("no json at all", "fix the typo");
        assert_eq!(g.order(), ["lead", "goal"]);
        assert_eq!(g.goal_settings().criteria, "fix the typo");
    }
}
