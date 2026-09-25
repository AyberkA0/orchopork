use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use super::{Store, now_unix};
use crate::error::{Error, Result};
use crate::git::GitRepo;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Snapshot {
    pub id: String,
    pub thread_id: String,
    pub parent_id: Option<String>,
    pub seq: i64,
    pub node: String,
    /// JSON-encoded context delta produced by this transition.
    pub context_delta: String,
    pub git_commit: String,
    pub accumulated_cost_usd: f64,
    pub created_at: i64,
}

pub enum RewindTarget {
    Snapshot(String),
    Steps(u32),
}

/// Couples every graph transition to exactly one git commit and one SQLite row.
pub struct Checkpointer {
    store: Store,
    repo: GitRepo,
}

impl Checkpointer {
    pub fn new(store: Store, repo: GitRepo) -> Self {
        Self { store, repo }
    }

    pub async fn head(&self, thread_id: &str) -> Result<Option<Snapshot>> {
        Ok(sqlx::query_as("SELECT * FROM snapshots WHERE thread_id = ? ORDER BY seq DESC LIMIT 1")
            .bind(thread_id)
            .fetch_optional(&self.store.pool)
            .await?)
    }

    pub async fn history(&self, thread_id: &str) -> Result<Vec<Snapshot>> {
        Ok(sqlx::query_as("SELECT * FROM snapshots WHERE thread_id = ? ORDER BY seq")
            .bind(thread_id)
            .fetch_all(&self.store.pool)
            .await?)
    }

    /// Commit the working tree, then record the snapshot + spend atomically.
    ///
    /// Ordering: git first, DB second. A crash in between leaves an orphan
    /// commit (harmless, reachable via reflog) but never a snapshot pointing
    /// at a commit that does not exist.
    pub async fn commit(&self, thread_id: &str, node: &str, delta: &Value, step_cost_usd: f64) -> Result<Snapshot> {
        let prev = self.head(thread_id).await?;
        let seq = prev.as_ref().map_or(0, |p| p.seq + 1);
        let commit = self.repo.commit_all(&format!("orchopork: {thread_id} #{seq} {node}")).await?;
        let now = now_unix();
        let snap = Snapshot {
            id: Uuid::new_v4().to_string(),
            thread_id: thread_id.into(),
            parent_id: prev.as_ref().map(|p| p.id.clone()),
            seq,
            node: node.into(),
            context_delta: serde_json::to_string(delta)?,
            git_commit: commit,
            accumulated_cost_usd: prev.map_or(0.0, |p| p.accumulated_cost_usd) + step_cost_usd,
            created_at: now,
        };

        let mut tx = self.store.pool.begin().await?;
        sqlx::query("INSERT OR IGNORE INTO threads (id, created_at) VALUES (?, ?)")
            .bind(thread_id)
            .bind(now)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "INSERT INTO snapshots (id, thread_id, parent_id, seq, node, context_delta, git_commit, accumulated_cost_usd, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&snap.id)
        .bind(&snap.thread_id)
        .bind(&snap.parent_id)
        .bind(snap.seq)
        .bind(&snap.node)
        .bind(&snap.context_delta)
        .bind(&snap.git_commit)
        .bind(snap.accumulated_cost_usd)
        .bind(snap.created_at)
        .execute(&mut *tx)
        .await?;
        sqlx::query("INSERT INTO spend_ledger (thread_id, node, cost_usd, created_at) VALUES (?, ?, ?, ?)")
            .bind(thread_id)
            .bind(node)
            .bind(step_cost_usd)
            .bind(now)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(snap)
    }

    /// `/rewind`: drop snapshots after the target and `git reset --hard` to its commit.
    ///
    /// The DB transaction stays open across the git reset: if git fails the
    /// deletes roll back, so SQLite and the worktree never disagree. Before
    /// resetting, the current tree (including uncommitted edits) is committed
    /// and pinned under `refs/orchopork/rewound/*`, so a rewind is recoverable.
    pub async fn rewind(&self, thread_id: &str, target: RewindTarget) -> Result<Snapshot> {
        let head = self.head(thread_id).await?.ok_or_else(|| Error::NotFound(format!("thread {thread_id}")))?;
        let dest: Snapshot = match target {
            RewindTarget::Snapshot(id) => sqlx::query_as("SELECT * FROM snapshots WHERE id = ? AND thread_id = ? AND seq <= ?")
                .bind(&id)
                .bind(thread_id)
                .bind(head.seq)
                .fetch_optional(&self.store.pool)
                .await?
                .ok_or_else(|| Error::NotFound(format!("snapshot {id} in thread {thread_id}")))?,
            RewindTarget::Steps(n) => {
                let seq = head.seq - i64::from(n);
                sqlx::query_as("SELECT * FROM snapshots WHERE thread_id = ? AND seq = ?")
                    .bind(thread_id)
                    .bind(seq)
                    .fetch_optional(&self.store.pool)
                    .await?
                    .ok_or_else(|| Error::NotFound(format!("cannot rewind {n} steps from seq {}", head.seq)))?
            }
        };

        let safety = self.repo.commit_all("orchopork: pre-rewind safety").await?;
        self.repo
            .update_ref(&format!("refs/orchopork/rewound/{}-{}", now_unix(), head.seq), &safety)
            .await?;

        let mut tx = self.store.pool.begin().await?;
        sqlx::query("DELETE FROM snapshots WHERE thread_id = ? AND seq > ?")
            .bind(thread_id)
            .bind(dest.seq)
            .execute(&mut *tx)
            .await?;
        self.repo.reset_hard(&dest.git_commit).await?; // Err drops tx => rollback
        tx.commit().await?;
        Ok(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    async fn read(dir: &std::path::Path) -> String {
        tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap()
    }

    #[tokio::test]
    async fn snapshot_commit_then_rewind_restores_db_and_worktree() {
        let ws = tempfile::tempdir().unwrap();
        let repo = GitRepo::new(ws.path());
        repo.init().await.unwrap();
        let store = Store::open(&ws.path().join(".orchopork/state.db")).await.unwrap();
        let cp = Checkpointer::new(store.clone(), repo.clone());

        let mut ids = vec![];
        for (v, cost) in [("v1", 0.10), ("v2", 0.20), ("v3", 0.30)] {
            tokio::fs::write(ws.path().join("a.txt"), v).await.unwrap();
            ids.push(cp.commit("t1", "coder", &json!({ "wrote": v }), cost).await.unwrap());
        }
        assert!((ids[2].accumulated_cost_usd - 0.60).abs() < 1e-9);
        // The state DB itself must not be versioned.
        assert!(repo.run(&["ls-files", ".orchopork"]).await.unwrap().is_empty());

        // Uncommitted junk must survive as a recoverable ref, not vanish.
        tokio::fs::write(ws.path().join("a.txt"), "dirty").await.unwrap();
        let d = cp.rewind("t1", RewindTarget::Steps(1)).await.unwrap();
        assert_eq!((d.seq, read(ws.path()).await.as_str()), (1, "v2"));
        assert_eq!(cp.history("t1").await.unwrap().len(), 2);

        let d = cp.rewind("t1", RewindTarget::Snapshot(ids[0].id.clone())).await.unwrap();
        assert_eq!((d.seq, read(ws.path()).await.as_str()), (0, "v1"));
        assert_eq!(cp.head("t1").await.unwrap().unwrap().id, ids[0].id);

        let refs = repo.run(&["for-each-ref", "refs/orchopork/rewound"]).await.unwrap();
        assert_eq!(refs.lines().count(), 2);

        // Money spent is not un-spent by rewinding.
        assert!((store.spend_since(0).await.unwrap() - 0.60).abs() < 1e-9);
        // Rewinding past the start is an error, not a wrap-around.
        assert!(cp.rewind("t1", RewindTarget::Steps(1)).await.is_err());
    }
}
