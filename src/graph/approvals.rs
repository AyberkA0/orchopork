//! Human approval of tool calls (semi-autonomous and manual runs).
//!
//! A gated call parks its run's loop here until the user allows or denies
//! it from the dashboard (or the CLI), switches the run to a mode that does
//! not need approval, or pauses the run. Pending requests live in memory
//! only: if orchopork exits, the run is marked interrupted and the turn is
//! redone on resume.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use tokio::sync::{broadcast, oneshot};

use super::Event;
use super::tools::ToolClass;
use crate::error::{Error, Result};
use crate::storage::{Autonomy, now_unix};

#[derive(Debug, Clone, Serialize)]
pub struct Approval {
    pub id: String,
    pub run_id: String,
    /// Orchestra agent id, if any.
    pub agent: Option<String>,
    pub tool: String,
    pub args: Value,
    /// "write" or "exec".
    pub class: &'static str,
    pub created_at: i64,
}

pub enum Decision {
    Allow,
    Deny(String),
    /// The run was paused while the call waited.
    Paused,
}

struct Pending {
    info: Approval,
    class: ToolClass,
    /// Key for "always allow" (tool name, or `acp:<kind>`).
    key: String,
    tx: oneshot::Sender<Decision>,
}

pub struct Approvals {
    pending: Mutex<HashMap<String, Pending>>,
    /// Per run: keys the user chose to always allow.
    always: Mutex<HashMap<String, HashSet<String>>>,
    events: broadcast::Sender<Event>,
}

pub fn class_name(c: ToolClass) -> &'static str {
    match c {
        ToolClass::Read => "read",
        ToolClass::Write => "write",
        ToolClass::Exec => "exec",
        ToolClass::Control => "control",
    }
}

impl Approvals {
    pub fn new(events: broadcast::Sender<Event>) -> Self {
        Self { pending: Mutex::new(HashMap::new()), always: Mutex::new(HashMap::new()), events }
    }

    /// Whether `key` needs asking under `autonomy` (and was not "always
    /// allowed" for this run).
    pub fn needed(&self, run_id: &str, key: &str, class: ToolClass, autonomy: Autonomy) -> bool {
        class.needs_approval(autonomy) && !self.always.lock().unwrap().get(run_id).is_some_and(|s| s.contains(key))
    }

    /// Waits for the user's decision on one call.
    #[allow(clippy::too_many_arguments)]
    pub async fn ask(
        &self,
        run_id: &str,
        agent: Option<&str>,
        tool: &str,
        args: Value,
        class: ToolClass,
        key: String,
        pause: &AtomicBool,
    ) -> Decision {
        let (tx, mut rx) = oneshot::channel();
        let info = Approval {
            id: uuid::Uuid::new_v4().simple().to_string()[..10].to_string(),
            run_id: run_id.to_string(),
            agent: agent.map(str::to_string),
            tool: tool.to_string(),
            args,
            class: class_name(class),
            created_at: now_unix(),
        };
        {
            let mut p = self.pending.lock().unwrap();
            // One loop per run, so at most one pending call per run.
            p.insert(run_id.to_string(), Pending { info: info.clone(), class, key, tx });
        }
        let _ = self.events.send(Event::Approval { run_id: run_id.to_string(), approval: Some(info) });
        let decision = loop {
            tokio::select! {
                d = &mut rx => break d.unwrap_or(Decision::Paused),
                _ = tokio::time::sleep(Duration::from_millis(300)) => {
                    if pause.load(Ordering::SeqCst) {
                        break Decision::Paused;
                    }
                }
            }
        };
        self.pending.lock().unwrap().remove(run_id);
        let _ = self.events.send(Event::Approval { run_id: run_id.to_string(), approval: None });
        decision
    }

    pub fn pending(&self, run_id: &str) -> Option<Approval> {
        self.pending.lock().unwrap().get(run_id).map(|p| p.info.clone())
    }

    /// Resolves the pending call `id` of `run_id`.
    pub fn decide(&self, run_id: &str, id: &str, allow: bool, always: bool, note: &str) -> Result<()> {
        let p = {
            let mut map = self.pending.lock().unwrap();
            match map.get(run_id) {
                Some(p) if p.info.id == id => map.remove(run_id).expect("present"),
                _ => return Err(Error::Conflict("that request is no longer pending".into())),
            }
        };
        if allow && always {
            self.always.lock().unwrap().entry(run_id.to_string()).or_default().insert(p.key.clone());
        }
        let _ = p.tx.send(if allow { Decision::Allow } else { Decision::Deny(note.trim().to_string()) });
        Ok(())
    }

    /// After a mode change: lets a pending call through if the new mode
    /// would not have asked.
    pub fn reconsider(&self, run_id: &str, autonomy: Autonomy) {
        let mut map = self.pending.lock().unwrap();
        if map.get(run_id).is_some_and(|p| !p.class.needs_approval(autonomy)) {
            let p = map.remove(run_id).expect("present");
            let _ = p.tx.send(Decision::Allow);
        }
    }

    /// Drops a pending request (its run was killed).
    pub fn withdraw(&self, run_id: &str) {
        if self.pending.lock().unwrap().remove(run_id).is_some() {
            let _ = self.events.send(Event::Approval { run_id: run_id.to_string(), approval: None });
        }
    }

    pub fn forget(&self, run_id: &str) {
        self.always.lock().unwrap().remove(run_id);
    }
}

/// Observation fed back to the model for a call that did not run.
pub fn refused_observation(tool: &str, d: &Decision) -> String {
    match d {
        Decision::Deny(note) if !note.is_empty() => format!(
            "The user declined this {tool} call and said: {note}\nIt was not executed. Follow their guidance."
        ),
        Decision::Deny(_) => format!(
            "The user declined this {tool} call; it was not executed. Take a different approach, or finish and explain what you need from them."
        ),
        _ => format!("Not executed: the run was paused while this {tool} call waited for approval."),
    }
}

/// ACP tool-call kinds mapped onto our classes.
pub fn acp_class(kind: &str) -> ToolClass {
    match kind {
        "read" | "search" | "fetch" | "think" => ToolClass::Read,
        "edit" | "move" => ToolClass::Write,
        _ => ToolClass::Exec,
    }
}
