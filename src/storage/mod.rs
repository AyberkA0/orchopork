//! SQLite persistence (WAL) for threads, snapshots and the spend ledger.

mod checkpoint;

use std::path::Path;
use std::str::FromStr;

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::SqlitePool;

pub use checkpoint::{Checkpointer, RewindTarget, Snapshot};

use crate::error::Result;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id                    TEXT PRIMARY KEY,
    thread_id             TEXT NOT NULL REFERENCES threads(id),
    parent_id             TEXT,
    seq                   INTEGER NOT NULL,
    node                  TEXT NOT NULL,
    context_delta         TEXT NOT NULL,
    git_commit            TEXT NOT NULL,
    accumulated_cost_usd  REAL NOT NULL,
    created_at            INTEGER NOT NULL,
    UNIQUE (thread_id, seq)
);
-- Append-only. Rewinds delete snapshots but never money already spent, so the
-- circuit breaker must read this table, not snapshots.accumulated_cost_usd.
CREATE TABLE IF NOT EXISTS spend_ledger (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT NOT NULL,
    node       TEXT NOT NULL,
    cost_usd   REAL NOT NULL,
    created_at INTEGER NOT NULL
);
";

#[derive(Clone)]
pub struct Store {
    pub(crate) pool: SqlitePool,
}

impl Store {
    pub async fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            tokio::fs::create_dir_all(dir).await?;
        }
        let opts = SqliteConnectOptions::from_str("sqlite://")?
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .busy_timeout(std::time::Duration::from_secs(5));
        let pool = SqlitePoolOptions::new().max_connections(4).connect_with(opts).await?;
        sqlx::raw_sql(SCHEMA).execute(&pool).await?;
        Ok(Self { pool })
    }

    /// Total spend since `since_unix` (for the daily/monthly circuit breaker).
    pub async fn spend_since(&self, since_unix: i64) -> Result<f64> {
        let (sum,): (Option<f64>,) = sqlx::query_as("SELECT SUM(cost_usd) FROM spend_ledger WHERE created_at >= ?")
            .bind(since_unix)
            .fetch_one(&self.pool)
            .await?;
        Ok(sum.unwrap_or(0.0))
    }
}

pub(crate) fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
