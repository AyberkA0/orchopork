//! The orchestra's flow graph: nodes (Lead → agents → Goal, plus
//! Notebooks) and the edges between them.
//!
//! Work flows left to right along edges, one link per socket: every node
//! has at most one link in and one link out, so the flow is a single line.
//! Parallel work goes through a Dispatcher node: besides its one link in
//! and out it has any number of branch sockets, each starting a branch (a
//! line of agents) that runs at the same time as the others, in its own
//! worktree, and reports back to the Dispatcher. Every edge between two workers
//! carries the finished node's report (and diff) to the next one; its
//! capabilities add ways for the two agents to talk (consult, suggest,
//! test) and review gates (report back, approve). Notebook edges are a
//! fixed kind: shared memory, not flow.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::config::{ModelRef, NetAccess};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// The single entry point: receives the goal, plans and hands off.
    Lead,
    #[default]
    Agent,
    /// The single exit: checks whether the goal is reached.
    Goal,
    /// Shared memory for the agents connected to it, for this run only.
    Notebook,
    /// The Dispatcher: runs its branches in parallel, collects their
    /// reports, merges their work and moves the flow on.
    Parallel,
}

/// When the Dispatcher moves the flow on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelPolicy {
    /// Once every branch has ended.
    #[default]
    WaitAll,
    /// As soon as one branch finishes; the others are cancelled.
    FirstSuccess,
    /// The Dispatcher's model reads each report and decides whether to go
    /// on (cancelling the branches still running) or to wait.
    Manager,
}

/// What the branches produce.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelWork {
    /// Changes to the project: finished branches are merged into the run's
    /// worktree (the Dispatcher resolves conflicts).
    #[default]
    Code,
    /// Research: findings go to reports and notebooks; file changes in the
    /// branches are discarded.
    Research,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParallelSettings {
    pub policy: ParallelPolicy,
    pub work: ParallelWork,
}

/// Who decides at the Goal node whether the goal is reached. Verification
/// commands, when there are any, run first and must pass in every mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalJudge {
    /// The agent that finished last before the Goal asks itself.
    #[default]
    Agent,
    /// An independent model (`judge_model`, else the lead's model).
    Judge,
    /// Only the verification commands decide.
    Command,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GoalSettings {
    /// Success criteria, ideally a checklist.
    pub criteria: String,
    pub judge: GoalJudge,
    pub judge_model: Option<ModelRef>,
    /// Must exit 0 (in addition to the run's own verify command).
    pub verify_command: Option<String>,
    /// Failed checks the Lead may try to fix before the run pauses.
    pub max_rounds: u32,
}

impl Default for GoalSettings {
    fn default() -> Self {
        Self { criteria: String::new(), judge: GoalJudge::Agent, judge_model: None, verify_command: None, max_rounds: 3 }
    }
}

/// Lead options: may it add and remove agents while the run executes?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LeadSettings {
    pub dynamic_agents: bool,
    pub max_dynamic: u32,
    /// Model of agents the Lead adds (else the team's worker model).
    pub dynamic_model: Option<ModelRef>,
    /// May the Lead edit files and run commands itself even though the
    /// flow has other agents? Off by default: the Lead investigates, plans
    /// and coordinates, and the agents do the work. A Lead with no agents
    /// after it always has its hands free.
    pub hands_on: bool,
}

impl Default for LeadSettings {
    fn default() -> Self {
        Self { dynamic_agents: false, max_dynamic: 3, dynamic_model: None, hands_on: false }
    }
}

/// Role keys the bundled role skills are written for. A node's role skill
/// is chosen by `FlowNode::role_key`.
pub const ROLE_KEYS: &[&str] =
    &["lead", "coder", "tester", "visual-tester", "reviewer", "researcher", "docs", "refactorer", "dispatcher"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowNode {
    pub id: String,
    #[serde(default)]
    pub kind: NodeKind,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub role: String,
    /// Which role playbook (skills with `roles: [..]`) guides this agent:
    /// one of `ROLE_KEYS` or a key of your own skills. Empty: guessed from
    /// the name and role text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub archetype: String,
    #[serde(default)]
    pub task: String,
    /// This agent's internet level; it can only narrow the workspace's.
    /// `None`: whatever the workspace allows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<NetAccess>,
    /// Overrides the team's model for this node.
    #[serde(default)]
    pub model: Option<ModelRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal: Option<GoalSettings>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lead: Option<LeadSettings>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel: Option<ParallelSettings>,
    /// Added by the Lead while the run executed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dynamic: bool,
    /// Canvas position (x, y), set by the editor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pos: Option<[f64; 2]>,
}

impl FlowNode {
    pub fn new(id: &str, kind: NodeKind, name: &str, task: &str) -> Self {
        Self {
            id: id.into(),
            kind,
            name: name.into(),
            role: String::new(),
            archetype: String::new(),
            task: task.into(),
            network: None,
            model: None,
            goal: (kind == NodeKind::Goal).then(GoalSettings::default),
            lead: (kind == NodeKind::Lead).then(LeadSettings::default),
            parallel: (kind == NodeKind::Parallel).then(ParallelSettings::default),
            dynamic: false,
            pos: None,
        }
    }

    /// Nodes with a model of their own (the Dispatcher's decides and merges).
    pub fn is_worker(&self) -> bool {
        matches!(self.kind, NodeKind::Lead | NodeKind::Agent | NodeKind::Parallel)
    }

    pub fn parallel_settings(&self) -> ParallelSettings {
        self.parallel.clone().unwrap_or_default()
    }

    /// The role playbook key for this node: the set archetype, else the
    /// node's kind, else a guess from its name and role text (the task is
    /// left out: any task mentions tests or docs in passing). Empty when
    /// nothing fits; such an agent gets the general team rules only.
    pub fn role_key(&self) -> String {
        match self.kind {
            NodeKind::Lead => return "lead".into(),
            NodeKind::Parallel => return "dispatcher".into(),
            NodeKind::Goal | NodeKind::Notebook => return String::new(),
            NodeKind::Agent => {}
        }
        let set = self.archetype.trim().to_lowercase();
        if !set.is_empty() {
            return set;
        }
        let t = format!("{} {}", self.name, self.role).to_lowercase();
        const GUESS: &[(&str, &[&str])] = &[
            ("visual-tester", &["visual", "screenshot", "görsel", "ui test", "ux test"]),
            ("tester", &["test", "qa", "quality", "verif"]),
            ("reviewer", &["review", "audit", "critic"]),
            ("researcher", &["research", "investigat", "explor", "scout"]),
            ("docs", &["doc", "readme", "writer", "changelog"]),
            ("refactorer", &["refactor", "clean", "restructur", "tidy"]),
            ("coder", &["code", "coder", "implement", "develop", "engineer"]),
        ];
        GUESS.iter().find(|(_, words)| words.iter().any(|w| t.contains(w))).map_or(String::new(), |(k, _)| (*k).to_string())
    }
}

/// What an edge lets its two agents do besides carrying the work forward.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EdgeCaps {
    /// Either side may ask the other a question.
    pub consult: bool,
    /// Either side may ask the other for improvement suggestions.
    pub suggest: bool,
    /// When `to` finishes, its report goes back to `from`, who accepts it or
    /// sends `to` back with an instruction.
    pub report: bool,
    /// `from` may ask `to` to test something.
    pub test: bool,
    /// Before `to` starts, it reviews `from`'s work and approves it or sends
    /// it back with reasons.
    pub approve: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeContext {
    /// Only the report.
    Report,
    /// The report and the diff of the node's changes.
    #[default]
    ReportDiff,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlowEdge {
    #[serde(default)]
    pub id: String,
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub caps: EdgeCaps,
    #[serde(default)]
    pub context: EdgeContext,
    /// Uses of each of consult/suggest/test per run.
    #[serde(default = "default_call_limit")]
    pub call_limit: u32,
    /// Report / approve rounds before the run pauses.
    #[serde(default = "default_loop_limit")]
    pub loop_limit: u32,
    /// From one of a Dispatcher's branch sockets to the branch's first agent.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub branch: bool,
}

fn default_call_limit() -> u32 {
    3
}
fn default_loop_limit() -> u32 {
    3
}

impl FlowEdge {
    pub fn new(from: &str, to: &str) -> Self {
        Self {
            id: format!("{from}->{to}"),
            from: from.into(),
            to: to.into(),
            caps: EdgeCaps::default(),
            context: EdgeContext::default(),
            call_limit: default_call_limit(),
            loop_limit: default_loop_limit(),
            branch: false,
        }
    }

    /// A branch edge from a Dispatcher.
    pub fn branch(from: &str, to: &str) -> Self {
        Self { branch: true, ..Self::new(from, to) }
    }

    /// The other endpoint, if `node` is one of them.
    pub fn other(&self, node: &str) -> Option<&str> {
        if self.from == node {
            Some(&self.to)
        } else if self.to == node {
            Some(&self.from)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlowGraph {
    pub nodes: Vec<FlowNode>,
    pub edges: Vec<FlowEdge>,
    /// Teams: agents the editor shows as one node. Purely visual; the run
    /// treats their members like any other node.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<FlowGroup>,
}

/// A team of agents drawn as one node (like a node group in Blender).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FlowGroup {
    pub id: String,
    pub name: String,
    pub members: Vec<String>,
    pub pos: Option<[f64; 2]>,
    /// Shown expanded (its members drawn) instead of as one node.
    pub open: bool,
}

/// Changes the Lead made while the run executed. Lives in the checkpointed
/// run state, so a rewind undoes them too.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GraphPatch {
    pub added_nodes: Vec<FlowNode>,
    pub added_edges: Vec<FlowEdge>,
    pub removed_nodes: Vec<String>,
    pub removed_edges: Vec<String>,
}

impl GraphPatch {
    pub fn is_empty(&self) -> bool {
        self.added_nodes.is_empty()
            && self.added_edges.is_empty()
            && self.removed_nodes.is_empty()
            && self.removed_edges.is_empty()
    }
}

impl FlowGraph {
    pub fn node(&self, id: &str) -> Option<&FlowNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn lead(&self) -> Option<&FlowNode> {
        self.nodes.iter().find(|n| n.kind == NodeKind::Lead)
    }

    pub fn goal(&self) -> Option<&FlowNode> {
        self.nodes.iter().find(|n| n.kind == NodeKind::Goal)
    }

    pub fn goal_settings(&self) -> GoalSettings {
        self.goal().and_then(|g| g.goal.clone()).unwrap_or_default()
    }

    pub fn lead_settings(&self) -> LeadSettings {
        self.lead().and_then(|l| l.lead.clone()).unwrap_or_default()
    }

    fn kind_of(&self, id: &str) -> Option<NodeKind> {
        self.node(id).map(|n| n.kind)
    }

    /// An edge that carries work (neither end is a notebook).
    pub fn is_flow(&self, e: &FlowEdge) -> bool {
        self.kind_of(&e.from) != Some(NodeKind::Notebook) && self.kind_of(&e.to) != Some(NodeKind::Notebook)
    }

    pub fn flow_edges(&self) -> impl Iterator<Item = &FlowEdge> {
        self.edges.iter().filter(|e| self.is_flow(e))
    }

    pub fn preds(&self, id: &str) -> Vec<&str> {
        self.flow_edges().filter(|e| e.to == id).map(|e| e.from.as_str()).collect()
    }

    pub fn succs(&self, id: &str) -> Vec<&str> {
        self.flow_edges().filter(|e| e.from == id).map(|e| e.to.as_str()).collect()
    }

    pub fn edge(&self, from: &str, to: &str) -> Option<&FlowEdge> {
        self.edges.iter().find(|e| e.from == from && e.to == to)
    }

    pub fn edge_by_id(&self, id: &str) -> Option<&FlowEdge> {
        self.edges.iter().find(|e| e.id == id)
    }

    /// Flow edges touching `id`, with the neighbour.
    pub fn links(&self, id: &str) -> Vec<(&FlowEdge, &FlowNode)> {
        self.flow_edges().filter_map(|e| e.other(id).and_then(|o| self.node(o)).map(|n| (e, n))).collect()
    }

    /// Notebooks connected to `id`.
    pub fn notebooks_of(&self, id: &str) -> Vec<&FlowNode> {
        self.edges
            .iter()
            .filter(|e| !self.is_flow(e))
            .filter_map(|e| e.other(id))
            .filter_map(|o| self.node(o))
            .filter(|n| n.kind == NodeKind::Notebook)
            .collect()
    }

    /// Nodes reachable from `id` along flow edges, `id` included.
    pub fn downstream(&self, id: &str) -> Vec<String> {
        let mut seen = vec![id.to_string()];
        let mut q = VecDeque::from([id.to_string()]);
        while let Some(n) = q.pop_front() {
            for s in self.succs(&n) {
                if !seen.iter().any(|x| x == s) {
                    seen.push(s.to_string());
                    q.push_back(s.to_string());
                }
            }
        }
        seen
    }

    /// The Dispatcher's branches: the first agent of each, top to bottom on
    /// the canvas.
    pub fn branch_heads(&self, dispatcher: &str) -> Vec<String> {
        let mut heads: Vec<&FlowNode> =
            self.edges.iter().filter(|e| e.branch && e.from == dispatcher).filter_map(|e| self.node(&e.to)).collect();
        heads.sort_by(|a, b| {
            let y = |n: &FlowNode| n.pos.map_or(0.0, |p| p[1]);
            y(a).total_cmp(&y(b)).then_with(|| a.id.cmp(&b.id))
        });
        heads.into_iter().map(|n| n.id.clone()).collect()
    }

    /// The agents of the branch starting at `head`, in order: along the
    /// one link out of each, until an agent without one (it reports back
    /// to the Dispatcher). Stops before anything that is not a plain agent
    /// with a single link in, which `validate` rejects.
    pub fn branch_chain(&self, head: &str) -> Vec<String> {
        let mut chain = vec![head.to_string()];
        let mut cur = head.to_string();
        loop {
            let next = self.flow_edges().filter(|e| e.from == cur && !e.branch).map(|e| e.to.clone()).collect::<Vec<_>>();
            let [n] = next.as_slice() else { break };
            let ok = self.kind_of(n) == Some(NodeKind::Agent) && self.preds(n).len() == 1 && !chain.contains(n);
            if !ok {
                break;
            }
            chain.push(n.clone());
            cur = n.clone();
        }
        chain
    }

    /// The Dispatcher and branch head of the branch `id` belongs to.
    pub fn branch_of(&self, id: &str) -> Option<(String, String)> {
        self.edges.iter().filter(|e| e.branch).find_map(|e| {
            self.branch_chain(&e.to).iter().any(|x| x == id).then(|| (e.from.clone(), e.to.clone()))
        })
    }

    /// Workers and the Goal in execution order: a topological order of the
    /// flow edges where ties go to the node higher on the canvas (then by
    /// id), so the order of parallel branches is visible and stable.
    pub fn order(&self) -> Vec<String> {
        let flow: Vec<&FlowNode> = self.nodes.iter().filter(|n| n.kind != NodeKind::Notebook).collect();
        let mut indeg: HashMap<&str, usize> = flow.iter().map(|n| (n.id.as_str(), 0)).collect();
        for e in self.flow_edges() {
            if let Some(d) = indeg.get_mut(e.to.as_str()) {
                *d += 1;
            }
        }
        let key = |id: &str| {
            let n = self.node(id);
            let y = n.and_then(|n| n.pos).map_or(0.0, |p| p[1]);
            (n.map(|n| n.kind != NodeKind::Lead).unwrap_or(true), (y * 100.0) as i64, id.to_string())
        };
        let mut ready: Vec<&str> = indeg.iter().filter(|(_, d)| **d == 0).map(|(id, _)| *id).collect();
        let mut out = Vec::new();
        while !ready.is_empty() {
            ready.sort_by_key(|id| key(id));
            let n = ready.remove(0);
            out.push(n.to_string());
            for s in self.succs(n) {
                if let Some(d) = indeg.get_mut(s) {
                    *d -= 1;
                    if *d == 0 {
                        ready.push(s);
                    }
                }
            }
        }
        out
    }

    /// The graph with the Lead's runtime changes applied.
    pub fn patched(&self, p: &GraphPatch) -> FlowGraph {
        if p.is_empty() {
            return self.clone();
        }
        let mut g = self.clone();
        g.nodes.extend(p.added_nodes.iter().cloned());
        g.edges.extend(p.added_edges.iter().cloned());
        g.nodes.retain(|n| !p.removed_nodes.contains(&n.id));
        for gr in &mut g.groups {
            gr.members.retain(|m| !p.removed_nodes.contains(m));
        }
        g.edges.retain(|e| {
            !p.removed_edges.contains(&e.id) && !p.removed_nodes.contains(&e.from) && !p.removed_nodes.contains(&e.to)
        });
        g
    }

    /// Fills in edge ids and node defaults; call before `validate`.
    pub fn normalize(&mut self) {
        for n in &mut self.nodes {
            n.id = n.id.trim().to_string();
            if n.name.trim().is_empty() {
                n.name = match n.kind {
                    NodeKind::Lead => "Lead".into(),
                    NodeKind::Goal => "Goal".into(),
                    NodeKind::Parallel => "Dispatcher".into(),
                    _ => n.id.clone(),
                };
            }
            match n.kind {
                NodeKind::Goal if n.goal.is_none() => n.goal = Some(GoalSettings::default()),
                NodeKind::Lead if n.lead.is_none() => n.lead = Some(LeadSettings::default()),
                NodeKind::Parallel if n.parallel.is_none() => n.parallel = Some(ParallelSettings::default()),
                _ => {}
            }
        }
        let mut seen = HashSet::new();
        for e in &mut self.edges {
            if e.id.trim().is_empty() || seen.contains(&e.id) {
                e.id = format!("{}->{}", e.from, e.to);
            }
            seen.insert(e.id.clone());
        }
    }

    /// Structural rules: one Lead, at most one Goal (without one the run
    /// ends when every node finished), well-formed edges, no cycles, every
    /// worker reachable from the Lead and on a path to the Goal if any.
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(Error::InvalidRequest(m));
        let count = |k: NodeKind| self.nodes.iter().filter(|n| n.kind == k).count();
        if count(NodeKind::Lead) != 1 {
            return bad(format!("the flow needs exactly one Lead node (found {})", count(NodeKind::Lead)));
        }
        if count(NodeKind::Goal) > 1 {
            return bad(format!("the flow can have at most one Goal node (found {})", count(NodeKind::Goal)));
        }
        let mut ids = HashSet::new();
        for n in &self.nodes {
            if n.id.is_empty() || !ids.insert(n.id.as_str()) {
                return bad(format!("duplicate or empty node id {:?}", n.id));
            }
            if n.kind == NodeKind::Agent && n.task.trim().is_empty() {
                return bad(format!("agent {:?} needs a task", n.name));
            }
        }
        let mut pairs = HashSet::new();
        for e in &self.edges {
            let (Some(a), Some(b)) = (self.kind_of(&e.from), self.kind_of(&e.to)) else {
                return bad(format!("edge {} connects a node that does not exist", e.id));
            };
            if e.from == e.to {
                return bad(format!("edge {} connects {} to itself", e.id, e.from));
            }
            if !pairs.insert((e.from.as_str(), e.to.as_str())) {
                return bad(format!("{} and {} are connected twice", e.from, e.to));
            }
            if e.branch && (a != NodeKind::Parallel || b != NodeKind::Agent) {
                return bad(format!("a branch link goes from a Dispatcher to an agent ({} → {})", e.from, e.to));
            }
            match (a, b) {
                (NodeKind::Notebook, NodeKind::Notebook) => return bad("notebooks cannot be connected to each other".into()),
                (NodeKind::Notebook, NodeKind::Parallel) | (NodeKind::Parallel, NodeKind::Notebook) => {
                    return bad("notebooks connect to agents, not to a Dispatcher".into());
                }
                (NodeKind::Notebook, NodeKind::Goal) | (NodeKind::Goal, NodeKind::Notebook) => {
                    return bad("notebooks connect to agents, not to the Goal".into());
                }
                (NodeKind::Notebook, _) | (_, NodeKind::Notebook) => {}
                (NodeKind::Goal, _) => return bad("the Goal is the end of the flow; nothing can follow it".into()),
                (_, NodeKind::Lead) => return bad("the Lead is the start of the flow; nothing can lead into it".into()),
                _ => {}
            }
        }
        // One link per socket: parallel work goes through a Dispatcher.
        for n in self.nodes.iter().filter(|n| n.kind != NodeKind::Notebook) {
            if self.preds(&n.id).len() > 1 {
                return bad(format!("{} has more than one link coming in; a socket takes one link (use a Dispatcher for parallel work)", n.name));
            }
            if self.flow_edges().filter(|e| e.from == n.id && !e.branch).count() > 1 {
                return bad(format!("{} has more than one link going out; a socket takes one link (use a Dispatcher for parallel work)", n.name));
            }
        }
        let order = self.order();
        let flow_nodes = self.nodes.iter().filter(|n| n.kind != NodeKind::Notebook).count();
        if order.len() != flow_nodes {
            return bad("the flow has a cycle; send work back with report/approve edges or the Goal instead".into());
        }
        // Branches: lines of agents that end by reporting to their
        // Dispatcher; no Dispatcher inside a branch.
        let mut in_branch: HashSet<String> = HashSet::new();
        for d in self.nodes.iter().filter(|n| n.kind == NodeKind::Parallel) {
            let heads = self.branch_heads(&d.id);
            if heads.is_empty() {
                return bad(format!("{} has no branches: connect agents or teams to its branch sockets", d.name));
            }
            for h in heads {
                let chain = self.branch_chain(&h);
                let last = chain.last().expect("non-empty");
                if let Some(e) = self.flow_edges().find(|e| e.from == *last && !e.branch) {
                    return bad(format!(
                        "{}'s branch must end with its last agent (it reports back to {}), but {} links on to {}",
                        d.name,
                        d.name,
                        self.node(last).map_or(last.as_str(), |n| n.name.as_str()),
                        self.node(&e.to).map_or(e.to.as_str(), |n| n.name.as_str())
                    ));
                }
                in_branch.extend(chain);
            }
        }
        let lead = self.lead().expect("counted").id.clone();
        // A team is one line of agents: one first, one last, all linked.
        for gr in &self.groups {
            let m: Vec<&String> = gr.members.iter().filter(|id| self.node(id).is_some()).collect();
            let inside = |id: &str| m.iter().any(|x| *x == id);
            let firsts: Vec<&&String> = m.iter().filter(|id| !self.preds(id).iter().any(|p| inside(p))).collect();
            let lasts = m.iter().filter(|id| !self.succs(id).iter().any(|s| inside(s))).count();
            let linked = firsts.first().is_some_and(|f| {
                let mut seen = vec![(**f).clone()];
                let mut cur = (**f).clone();
                while let Some(n) = self.succs(&cur).into_iter().find(|s| inside(s) && !seen.iter().any(|x| x == s)) {
                    cur = n.to_string();
                    seen.push(cur.clone());
                }
                seen.len() == m.len()
            });
            if !m.is_empty() && (firsts.len() != 1 || lasts != 1 || !linked) {
                return bad(format!("team {} must be one line of agents: one first agent, one last agent, linked in between", gr.name));
            }
        }
        let goal = self.goal().map(|n| n.id.clone());
        let from_lead: HashSet<String> = self.downstream(&lead).into_iter().collect();
        for n in self.nodes.iter().filter(|n| n.kind != NodeKind::Notebook) {
            if !from_lead.contains(&n.id) {
                return bad(format!("{} cannot be reached from the Lead", n.name));
            }
            if let Some(goal) = &goal
                && !in_branch.contains(&n.id)
                && !self.downstream(&n.id).contains(goal)
            {
                return bad(format!("{} has no path to the Goal", n.name));
            }
        }
        Ok(())
    }

    /// A linear graph for a legacy team (lead + members): Lead → each
    /// member in turn → Goal, with the Lead reviewing reports.
    pub fn from_team(names: &[(String, String, String)], goal_text: &str) -> FlowGraph {
        let mut g = FlowGraph::default();
        g.nodes.push(FlowNode::new("lead", NodeKind::Lead, "Lead", goal_text));
        let mut prev = "lead".to_string();
        for (i, (name, role, task)) in names.iter().enumerate() {
            let id = format!("agent_{}", i + 1);
            let mut n = FlowNode::new(&id, NodeKind::Agent, name, task);
            n.role = role.clone();
            g.nodes.push(n);
            g.edges.push(FlowEdge::new(&prev, &id));
            prev = id;
        }
        g.nodes.push(FlowNode::new("goal", NodeKind::Goal, "Goal", ""));
        g.edges.push(FlowEdge::new(&prev, "goal"));
        g
    }
}

/// Canvas positions for nodes without one: columns by longest path from
/// the Lead, rows in order; notebooks in a row below.
pub fn auto_layout(g: &mut FlowGraph) {
    let order = g.order();
    let mut col: BTreeMap<String, usize> = BTreeMap::new();
    // What follows a Dispatcher goes after its longest branch: a second
    // pass, once the branches have their columns.
    let after: HashMap<String, Vec<String>> = order
        .iter()
        .map(|id| {
            let branches = g
                .flow_edges()
                .filter(|e| e.to == *id && !e.branch && g.kind_of(&e.from) == Some(NodeKind::Parallel))
                .flat_map(|e| g.branch_heads(&e.from).into_iter().flat_map(|h| g.branch_chain(&h)))
                .collect();
            (id.clone(), branches)
        })
        .collect();
    for _ in 0..2 {
        for id in &order {
            let c = g
                .preds(id)
                .into_iter()
                .chain(after[id].iter().map(String::as_str))
                .filter_map(|p| col.get(p))
                .map(|c| c + 1)
                .max()
                .unwrap_or(0);
            col.insert(id.clone(), c);
        }
    }
    let mut rows: HashMap<usize, usize> = HashMap::new();
    let mut nb = 0;
    let max_rows = {
        let mut per: HashMap<usize, usize> = HashMap::new();
        for c in col.values() {
            *per.entry(*c).or_default() += 1;
        }
        per.values().copied().max().unwrap_or(1)
    };
    for id in &order {
        let c = col[id];
        let r = rows.entry(c).or_default();
        if let Some(n) = g.nodes.iter_mut().find(|n| &n.id == id)
            && n.pos.is_none()
        {
            n.pos = Some([40.0 + c as f64 * 300.0, 60.0 + *r as f64 * 170.0]);
        }
        *r += 1;
    }
    for n in g.nodes.iter_mut().filter(|n| n.kind == NodeKind::Notebook) {
        if n.pos.is_none() {
            n.pos = Some([340.0 + nb as f64 * 300.0, 80.0 + max_rows as f64 * 170.0]);
        }
        nb += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(nodes: &[(&str, NodeKind)], edges: &[(&str, &str)]) -> FlowGraph {
        let mut g = FlowGraph {
            nodes: nodes.iter().map(|(id, k)| FlowNode::new(id, *k, id, "task")).collect(),
            edges: edges.iter().map(|(a, b)| FlowEdge::new(a, b)).collect(),
            ..Default::default()
        };
        g.normalize();
        g
    }
    use NodeKind::*;

    #[test]
    fn a_simple_flow_is_valid_and_ordered() {
        let f = g(&[("lead", Lead), ("dev", Agent), ("test", Agent), ("goal", Goal)], &[
            ("lead", "dev"),
            ("dev", "test"),
            ("test", "goal"),
        ]);
        f.validate().unwrap();
        assert_eq!(f.order(), ["lead", "dev", "test", "goal"]);
        assert_eq!(f.downstream("dev"), ["dev", "test", "goal"]);
    }

    #[test]
    fn structural_rules_are_enforced() {
        let cases: Vec<(FlowGraph, &str)> = vec![
            (g(&[("lead", Lead), ("goal", Goal), ("l2", Lead)], &[("lead", "goal"), ("l2", "goal")]), "exactly one Lead"),
            (g(&[("lead", Lead), ("a", Agent), ("g1", Goal), ("g2", Goal)], &[("lead", "a"), ("a", "g1"), ("a", "g2")]), "at most one Goal"),
            (g(&[("lead", Lead), ("a", Agent), ("b", Agent), ("goal", Goal)], &[("lead", "goal"), ("a", "b"), ("b", "a")]), "cycle"),
            (g(&[("lead", Lead), ("a", Agent), ("goal", Goal)], &[("lead", "goal")]), "cannot be reached"),
            (g(&[("lead", Lead), ("a", Agent), ("goal", Goal)], &[("lead", "a"), ("lead", "goal")]), "more than one link going out"),
            (g(&[("lead", Lead), ("a", Agent), ("b", Agent), ("goal", Goal)], &[("lead", "a"), ("a", "goal"), ("b", "goal")]), "more than one link coming in"),
            (g(&[("lead", Lead), ("par", Parallel), ("goal", Goal)], &[("lead", "par"), ("par", "goal")]), "has no branches"),
            (g(&[("lead", Lead), ("goal", Goal), ("n1", Notebook), ("n2", Notebook)], &[("lead", "goal"), ("n1", "n2")]), "notebooks cannot"),
            (g(&[("lead", Lead), ("goal", Goal), ("n", Notebook)], &[("lead", "goal"), ("n", "goal")]), "not to the Goal"),
            (g(&[("lead", Lead), ("a", Agent), ("goal", Goal)], &[("lead", "a"), ("a", "goal"), ("goal", "a")]), "end of the flow"),
            (g(&[("lead", Lead), ("a", Agent), ("goal", Goal)], &[("lead", "a"), ("a", "goal"), ("a", "lead")]), "start of the flow"),
        ];
        for (f, want) in cases {
            let e = f.validate().unwrap_err().to_string();
            assert!(e.contains(want), "{want}: got {e}");
        }
    }

    #[test]
    fn teams_survive_a_round_trip_and_lose_retired_members() {
        let mut f = g(&[("lead", Lead), ("a", Agent), ("b", Agent)], &[("lead", "a"), ("a", "b")]);
        f.groups.push(FlowGroup { id: "team_1".into(), name: "Team".into(), members: vec!["a".into(), "b".into()], pos: Some([1.0, 2.0]), open: false });
        let back: FlowGraph = serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
        assert_eq!(back, f);
        back.validate().unwrap();
        let mut p = GraphPatch::default();
        p.removed_nodes.push("b".into());
        assert_eq!(back.patched(&p).groups[0].members, ["a"]);
    }

    #[test]
    fn a_team_is_one_line_of_agents() {
        let mut f = g(&[("lead", Lead), ("a", Agent), ("b", Agent), ("c", Agent)], &[("lead", "a"), ("a", "b"), ("b", "c")]);
        f.groups.push(FlowGroup { id: "t".into(), name: "T".into(), members: vec!["a".into(), "b".into()], pos: None, open: false });
        f.validate().unwrap();
        f.groups[0].members = vec!["a".into(), "c".into()];
        assert!(f.validate().unwrap_err().to_string().contains("one line of agents"), "a and c are not linked");
    }

    #[test]
    fn the_goal_is_optional() {
        let f = g(&[("lead", Lead), ("a", Agent), ("b", Agent)], &[("lead", "a"), ("a", "b")]);
        f.validate().unwrap();
        assert_eq!(f.order().len(), 3);
    }

    #[test]
    fn notebooks_are_not_flow_and_need_no_path_to_the_goal() {
        let f = g(&[("lead", Lead), ("a", Agent), ("goal", Goal), ("mem", Notebook)], &[
            ("lead", "a"),
            ("a", "goal"),
            ("mem", "a"),
            ("lead", "mem"),
        ]);
        f.validate().unwrap();
        assert_eq!(f.order(), ["lead", "a", "goal"]);
        assert_eq!(f.notebooks_of("a")[0].id, "mem");
        assert!(f.preds("a") == ["lead"], "notebook edges carry no work");
    }

    #[test]
    fn parallel_branches_hang_off_a_dispatcher() {
        let mut f = g(&[("lead", Lead), ("par", Parallel), ("front", Agent), ("back", Agent), ("api", Agent), ("goal", Goal)], &[
            ("lead", "par"),
            ("back", "api"),
            ("par", "goal"),
        ]);
        f.edges.push(FlowEdge::branch("par", "front"));
        f.edges.push(FlowEdge::branch("par", "back"));
        f.normalize();
        f.nodes[2].pos = Some([300.0, 400.0]);
        f.nodes[3].pos = Some([300.0, 100.0]);
        f.validate().unwrap();
        assert_eq!(f.branch_heads("par"), ["back", "front"], "top to bottom");
        assert_eq!(f.branch_chain("back"), ["back", "api"]);
        assert_eq!(f.branch_of("api"), Some(("par".into(), "back".into())));
        assert_eq!(f.branch_of("goal"), None);
        // Branch agents need no path to the Goal: they report to the Dispatcher.
        // But a branch must end with its last agent.
        let mut bad = f.clone();
        bad.edges.retain(|e| e.id != "par->goal");
        bad.edges.push(FlowEdge::new("api", "goal"));
        assert!(bad.validate().unwrap_err().to_string().contains("must end with its last agent"));
        // No Dispatcher inside a branch, and branch links start at a Dispatcher.
        let mut nested = f.clone();
        nested.edges.push(FlowEdge::branch("lead", "x"));
        assert!(nested.validate().is_err());
    }

    #[test]
    fn patches_add_and_remove_nodes() {
        let f = g(&[("lead", Lead), ("a", Agent), ("goal", Goal)], &[("lead", "a"), ("a", "goal")]);
        let mut p = GraphPatch::default();
        p.added_nodes.push(FlowNode::new("x", Agent, "X", "extra"));
        p.removed_edges.push("a->goal".into());
        p.added_edges.push(FlowEdge::new("a", "x"));
        p.added_edges.push(FlowEdge::new("x", "goal"));
        let h = f.patched(&p);
        h.validate().unwrap();
        assert_eq!(h.order(), ["lead", "a", "x", "goal"]);
    }

    #[test]
    fn layout_places_columns_left_to_right() {
        let mut f = g(&[("lead", Lead), ("a", Agent), ("goal", Goal), ("mem", Notebook)], &[("lead", "a"), ("a", "goal"), ("mem", "a")]);
        auto_layout(&mut f);
        let x = |id: &str| f.node(id).unwrap().pos.unwrap()[0];
        assert!(x("lead") < x("a") && x("a") < x("goal"));
    }
}
