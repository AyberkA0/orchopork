//! Runs and their steps. Every step row points at the git commit that
//! captured the run's worktree right after that step, and carries the full
//! `RunState` the loop needs to continue from it — so resuming, restarting
//! after a crash, and rewinding are all just "load step N".

use serde::{Deserialize, Serialize};
use sqlx::Row;
use sqlx::sqlite::SqliteRow;

use super::{Store, now_unix};
use crate::config::ModelRef;
use crate::error::{Error, Result};

macro_rules! text_enum {
    ($name:ident { $($variant:ident => $s:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }

        impl $name {
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $s),+ }
            }
            pub fn parse(s: &str) -> Result<Self> {
                match s {
                    $($s => Ok($name::$variant),)+
                    _ => Err(Error::InvalidRequest(format!(concat!("unknown ", stringify!($name), " {:?}"), s))),
                }
            }
        }
    };
}

text_enum!(RunStatus { Running => "running", Paused => "paused", Done => "done" });
text_enum!(Phase { Plan => "plan", Act => "act", Verify => "verify", Review => "review", Done => "done" });
text_enum!(StepKind { Plan => "plan", Act => "act", Verify => "verify", Review => "review", Inject => "inject" });

/// How a run is driven.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// plan → act → verify → review, models from the workspace routing.
    #[default]
    Classic,
    /// One agent with tools and a single chosen model (a classic harness).
    Solo,
    /// A hierarchy of agents led by one commander (see `graph::orchestra`).
    Orchestra,
}

/// One member of an orchestra. `parent == None` marks the single root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub role: String,
    pub task: String,
    #[serde(default)]
    pub parent: Option<String>,
    /// Overrides the run's lead model for this agent.
    #[serde(default)]
    pub model: Option<ModelRef>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunSpec {
    pub mode: RunMode,
    /// Solo: the model. Orchestra: the lead's model, and every agent's when
    /// `worker_model` is not set.
    pub model: Option<ModelRef>,
    /// Orchestra: the model of every agent below the lead that does not
    /// pick its own, e.g. a strong lead with cheaper workers.
    pub worker_model: Option<ModelRef>,
    pub agents: Vec<AgentSpec>,
}

impl RunSpec {
    pub fn root(&self) -> Option<&AgentSpec> {
        self.agents.iter().find(|a| a.parent.is_none())
    }
    pub fn agent(&self, id: &str) -> Option<&AgentSpec> {
        self.agents.iter().find(|a| a.id == id)
    }
    pub fn children<'a>(&'a self, id: &'a str) -> impl Iterator<Item = &'a AgentSpec> + 'a {
        self.agents.iter().filter(move |a| a.parent.as_deref() == Some(id))
    }

    /// One root, unique ids, existing parents, no cycles, non-empty tasks.
    pub fn validate_agents(&self) -> Result<()> {
        let bad = |m: String| Err(Error::InvalidRequest(m));
        if self.agents.is_empty() {
            return bad("the orchestra has no agents".into());
        }
        let roots = self.agents.iter().filter(|a| a.parent.is_none()).count();
        if roots != 1 {
            return bad(format!("the orchestra needs exactly one lead agent (found {roots})"));
        }
        let mut seen = std::collections::HashSet::new();
        for a in &self.agents {
            if a.id.trim().is_empty() || !seen.insert(a.id.as_str()) {
                return bad(format!("duplicate or empty agent id {:?}", a.id));
            }
            if a.task.trim().is_empty() || a.name.trim().is_empty() {
                return bad(format!("agent {:?} needs a name and a task", a.id));
            }
        }
        for a in &self.agents {
            let mut cur = a.parent.as_deref();
            let mut hops = 0;
            while let Some(p) = cur {
                let Some(pa) = self.agent(p) else {
                    return bad(format!("{}'s commander {p:?} does not exist", a.name));
                };
                hops += 1;
                if hops > self.agents.len() {
                    return bad(format!("{} is part of a command cycle", a.name));
                }
                cur = pa.parent.as_deref();
            }
        }
        Ok(())
    }
}

/// Loop state after a step. Stored with every step, so the state at any
/// checkpoint is exact rather than re-derived.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    pub phase: Phase,
    /// Consecutive failed actor turns.
    pub failures: u32,
    /// Review rounds that ended in "revise".
    pub reviews: u32,
    /// Route the next actor turn to the escalation model.
    pub escalate: bool,
    /// Orchestra only: active agent chain (last = acting now).
    #[serde(default)]
    pub stack: Vec<String>,
    /// Orchestra only: agents whose task is complete.
    #[serde(default)]
    pub done: Vec<String>,
}

impl Default for RunState {
    fn default() -> Self {
        Self { phase: Phase::Plan, failures: 0, reviews: 0, escalate: false, stack: vec![], done: vec![] }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Run {
    pub id: String,
    pub goal: String,
    pub status: RunStatus,
    pub branch: String,
    pub worktree: String,
    pub base_commit: String,
    pub verify_command: Option<String>,
    /// Why the run last stopped, when it was not a clean finish.
    pub error: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    /// Everything this run has spent, including calls whose output was
    /// later rewound away.
    pub cost_usd: f64,
    pub step_count: i64,
    pub spec: RunSpec,
}

#[derive(Debug, Clone, Serialize)]
pub struct Step {
    pub run_id: String,
    pub seq: i64,
    pub kind: StepKind,
    /// Model reply, command summary, or injected text.
    pub output: String,
    /// Tool result / verification output fed back to the model.
    pub observation: Option<String>,
    /// Free-form details: model, tokens, tool name, ok flag, escalation.
    pub meta: serde_json::Value,
    pub state: RunState,
    pub git_commit: String,
    pub cost_usd: f64,
    pub created_at: i64,
}

const RUN_COLUMNS: &str = "r.id, r.goal, r.status, r.branch, r.worktree, r.base_commit, r.verify_command, r.error,
    r.created_at, r.updated_at, r.spec,
    (SELECT COALESCE(SUM(s.cost_usd), 0.0) FROM spend s WHERE s.run_id = r.id) AS cost_usd,
    (SELECT COUNT(*) FROM steps t WHERE t.run_id = r.id) AS step_count";

fn run_from_row(row: &SqliteRow) -> Result<Run> {
    Ok(Run {
        id: row.try_get("id")?,
        goal: row.try_get("goal")?,
        status: RunStatus::parse(row.try_get("status")?)?,
        branch: row.try_get("branch")?,
        worktree: row.try_get("worktree")?,
        base_commit: row.try_get("base_commit")?,
        verify_command: row.try_get("verify_command")?,
        error: row.try_get("error")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        cost_usd: row.try_get("cost_usd")?,
        step_count: row.try_get("step_count")?,
        spec: serde_json::from_str(row.try_get::<&str, _>("spec")?).unwrap_or_default(),
    })
}

fn step_from_row(row: &SqliteRow) -> Result<Step> {
    let meta: String = row.try_get("meta")?;
    let state: String = row.try_get("state")?;
    Ok(Step {
        run_id: row.try_get("run_id")?,
        seq: row.try_get("seq")?,
        kind: StepKind::parse(row.try_get("kind")?)?,
        output: row.try_get("output")?,
        observation: row.try_get("observation")?,
        meta: serde_json::from_str(&meta)?,
        state: serde_json::from_str(&state)?,
        git_commit: row.try_get("git_commit")?,
        cost_usd: row.try_get("cost_usd")?,
        created_at: row.try_get("created_at")?,
    })
}

/// What a caller provides to append a step; `seq` and timestamps are
/// assigned by the store.
pub struct NewStep<'a> {
    pub kind: StepKind,
    pub output: &'a str,
    pub observation: Option<&'a str>,
    pub meta: serde_json::Value,
    pub state: &'a RunState,
    pub git_commit: &'a str,
    pub cost_usd: f64,
}

impl Store {
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_run(
        &self,
        id: &str,
        goal: &str,
        branch: &str,
        worktree: &str,
        base_commit: &str,
        verify_command: Option<&str>,
        spec: &RunSpec,
    ) -> Result<Run> {
        let now = now_unix();
        sqlx::query(
            "INSERT INTO runs (id, goal, status, branch, worktree, base_commit, verify_command, error, created_at, updated_at, spec)
             VALUES (?, ?, 'paused', ?, ?, ?, ?, NULL, ?, ?, ?)",
        )
        .bind(id)
        .bind(goal)
        .bind(branch)
        .bind(worktree)
        .bind(base_commit)
        .bind(verify_command)
        .bind(now)
        .bind(now)
        .bind(serde_json::to_string(spec)?)
        .execute(&self.pool)
        .await?;
        self.get_run(id).await
    }

    pub async fn get_run(&self, id: &str) -> Result<Run> {
        let row = sqlx::query(&format!("SELECT {RUN_COLUMNS} FROM runs r WHERE r.id = ?"))
            .bind(id)
            .fetch_optional(&self.pool)
            .await?
            .ok_or_else(|| Error::NotFound(format!("run {id}")))?;
        run_from_row(&row)
    }

    pub async fn list_runs(&self) -> Result<Vec<Run>> {
        let rows = sqlx::query(&format!("SELECT {RUN_COLUMNS} FROM runs r ORDER BY r.created_at DESC, r.id"))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(run_from_row).collect()
    }

    pub async fn set_run_status(&self, id: &str, status: RunStatus, error: Option<&str>) -> Result<Run> {
        let n = sqlx::query("UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE id = ?")
            .bind(status.as_str())
            .bind(error)
            .bind(now_unix())
            .bind(id)
            .execute(&self.pool)
            .await?
            .rows_affected();
        if n == 0 {
            return Err(Error::NotFound(format!("run {id}")));
        }
        self.get_run(id).await
    }

    /// Runs left `running` by a process that exited mid-step. Nothing is
    /// lost (state lives in steps + git), they just need a resume.
    pub async fn mark_interrupted_runs(&self) -> Result<u64> {
        Ok(sqlx::query("UPDATE runs SET status = 'paused', error = ?, updated_at = ? WHERE status = 'running'")
            .bind("interrupted: orchopork exited while this run was executing; resume to continue")
            .bind(now_unix())
            .execute(&self.pool)
            .await?
            .rows_affected())
    }

    pub async fn set_run_spec(&self, id: &str, spec: &RunSpec) -> Result<()> {
        sqlx::query("UPDATE runs SET spec = ? WHERE id = ?")
            .bind(serde_json::to_string(spec)?)
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn delete_run(&self, id: &str) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        sqlx::query("DELETE FROM steps WHERE run_id = ?").bind(id).execute(&mut *tx).await?;
        sqlx::query("DELETE FROM runs WHERE id = ?").bind(id).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn append_step(&self, run_id: &str, s: NewStep<'_>) -> Result<Step> {
        let now = now_unix();
        let mut tx = self.pool.begin().await?;
        let (seq,): (i64,) = sqlx::query_as("SELECT COALESCE(MAX(seq), -1) + 1 FROM steps WHERE run_id = ?")
            .bind(run_id)
            .fetch_one(&mut *tx)
            .await?;
        sqlx::query(
            "INSERT INTO steps (run_id, seq, kind, output, observation, meta, state, git_commit, cost_usd, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(run_id)
        .bind(seq)
        .bind(s.kind.as_str())
        .bind(s.output)
        .bind(s.observation)
        .bind(s.meta.to_string())
        .bind(serde_json::to_string(s.state)?)
        .bind(s.git_commit)
        .bind(s.cost_usd)
        .bind(now)
        .execute(&mut *tx)
        .await?;
        sqlx::query("UPDATE runs SET updated_at = ? WHERE id = ?").bind(now).bind(run_id).execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(Step {
            run_id: run_id.to_string(),
            seq,
            kind: s.kind,
            output: s.output.to_string(),
            observation: s.observation.map(str::to_string),
            meta: s.meta,
            state: s.state.clone(),
            git_commit: s.git_commit.to_string(),
            cost_usd: s.cost_usd,
            created_at: now,
        })
    }

    pub async fn steps(&self, run_id: &str) -> Result<Vec<Step>> {
        let rows =
            sqlx::query("SELECT * FROM steps WHERE run_id = ? ORDER BY seq").bind(run_id).fetch_all(&self.pool).await?;
        rows.iter().map(step_from_row).collect()
    }

    /// Steps after `seq`, oldest first (what a viewer that already has the
    /// rest needs).
    pub async fn steps_after(&self, run_id: &str, seq: i64) -> Result<Vec<Step>> {
        let rows = sqlx::query("SELECT * FROM steps WHERE run_id = ? AND seq > ? ORDER BY seq")
            .bind(run_id)
            .bind(seq)
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(step_from_row).collect()
    }

    pub async fn step(&self, run_id: &str, seq: i64) -> Result<Step> {
        let row = sqlx::query("SELECT * FROM steps WHERE run_id = ? AND seq = ?")
            .bind(run_id)
            .bind(seq)
            .fetch_optional(&self.pool)
            .await?
            .ok_or_else(|| Error::NotFound(format!("step {seq} of run {run_id}")))?;
        step_from_row(&row)
    }

    pub async fn head_step(&self, run_id: &str) -> Result<Option<Step>> {
        let row = sqlx::query("SELECT * FROM steps WHERE run_id = ? ORDER BY seq DESC LIMIT 1")
            .bind(run_id)
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(step_from_row).transpose()
    }

    pub async fn truncate_steps_after(&self, run_id: &str, seq: i64) -> Result<u64> {
        Ok(sqlx::query("DELETE FROM steps WHERE run_id = ? AND seq > ?")
            .bind(run_id)
            .bind(seq)
            .execute(&self.pool)
            .await?
            .rows_affected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn steps_append_in_order_and_spend_survives_truncation() {
        let d = tempfile::tempdir().unwrap();
        let store = Store::open(&d.path().join("s.db")).await.unwrap();
        store.insert_run("r1", "goal", "orchopork/r1", "/wt", "abcdef1", None, &RunSpec::default()).await.unwrap();
        let st = RunState::default();
        for i in 0..3 {
            let s = store
                .append_step(
                    "r1",
                    NewStep {
                        kind: StepKind::Act,
                        output: "o",
                        observation: None,
                        meta: serde_json::json!({ "i": i }),
                        state: &st,
                        git_commit: "abcdef1",
                        cost_usd: 0.5,
                    },
                )
                .await
                .unwrap();
            assert_eq!(s.seq, i);
            store.record_spend("r1", "claude", "m", 1, 1, 0.5).await.unwrap();
        }
        assert_eq!(store.truncate_steps_after("r1", 0).await.unwrap(), 2);
        let run = store.get_run("r1").await.unwrap();
        assert_eq!(run.step_count, 1);
        assert!((run.cost_usd - 1.5).abs() < 1e-9, "rewinding must not un-spend money");
        assert_eq!(store.head_step("r1").await.unwrap().unwrap().meta["i"], 0);

        store.set_run_status("r1", RunStatus::Running, None).await.unwrap();
        assert_eq!(store.mark_interrupted_runs().await.unwrap(), 1);
        assert_eq!(store.get_run("r1").await.unwrap().status, RunStatus::Paused);
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    fn a(id: &str, parent: Option<&str>) -> AgentSpec {
        AgentSpec {
            id: id.into(),
            name: id.into(),
            role: String::new(),
            task: "t".into(),
            parent: parent.map(Into::into),
            model: None,
        }
    }

    #[test]
    fn agent_trees_are_validated() {
        let ok = RunSpec {
            worker_model: None,
            mode: RunMode::Orchestra,
            model: None,
            agents: vec![a("r", None), a("x", Some("r")), a("y", Some("x"))],
        };
        assert!(ok.validate_agents().is_ok());
        assert_eq!(ok.children("r").count(), 1);
        let two_roots = RunSpec { agents: vec![a("r", None), a("s", None)], ..ok.clone() };
        assert!(two_roots.validate_agents().is_err());
        let cycle = RunSpec { agents: vec![a("r", None), a("x", Some("y")), a("y", Some("x"))], ..ok.clone() };
        assert!(cycle.validate_agents().is_err());
        let orphan = RunSpec { agents: vec![a("r", None), a("x", Some("nope"))], ..ok };
        assert!(orphan.validate_agents().is_err());
    }
}
