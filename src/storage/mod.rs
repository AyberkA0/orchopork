//! SQLite persistence (WAL): runs, their checkpointed steps, and the
//! append-only spend ledger the budget guard reads.

mod checkpoint;

use std::path::Path;
use std::str::FromStr;

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};

pub use checkpoint::{NewStep, Phase, Run, RunState, RunStatus, Step, StepKind};

use crate::error::Result;

/// v2 replaces the v1 `threads`/`snapshots`/`spend_ledger` tables (a manual
/// step API with no run concept). v1 tables are left in place, unused.
const SCHEMA_VERSION: i64 = 2;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    goal            TEXT NOT NULL,
    status          TEXT NOT NULL,
    branch          TEXT NOT NULL,
    worktree        TEXT NOT NULL,
    base_commit     TEXT NOT NULL,
    verify_command  TEXT,
    error           TEXT,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    output       TEXT NOT NULL,
    observation  TEXT,
    meta         TEXT NOT NULL,
    state        TEXT NOT NULL,
    git_commit   TEXT NOT NULL,
    cost_usd     REAL NOT NULL,
    created_at   INTEGER NOT NULL,
    PRIMARY KEY (run_id, seq)
);
-- Append-only, and deliberately not tied to steps: a rewind deletes steps
-- but never un-spends money, and a paid call whose reply was rejected
-- (unparseable, validator) still counts against the cap.
CREATE TABLE IF NOT EXISTS spend (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT NOT NULL,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    prompt_tokens      INTEGER NOT NULL,
    completion_tokens  INTEGER NOT NULL,
    cost_usd           REAL NOT NULL,
    created_at         INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS spend_created_at ON spend(created_at);
CREATE INDEX IF NOT EXISTS spend_run ON spend(run_id);
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
        sqlx::raw_sql(&format!("PRAGMA user_version = {SCHEMA_VERSION}")).execute(&pool).await?;
        Ok(Self { pool })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn record_spend(
        &self,
        run_id: &str,
        provider: &str,
        model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
        cost_usd: f64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO spend (run_id, provider, model, prompt_tokens, completion_tokens, cost_usd, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(run_id)
        .bind(provider)
        .bind(model)
        .bind(prompt_tokens as i64)
        .bind(completion_tokens as i64)
        .bind(cost_usd)
        .bind(now_unix())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Total spend since `since_unix` (the monthly circuit breaker).
    pub async fn spend_since(&self, since_unix: i64) -> Result<f64> {
        let (sum,): (Option<f64>,) = sqlx::query_as("SELECT SUM(cost_usd) FROM spend WHERE created_at >= ?")
            .bind(since_unix)
            .fetch_one(&self.pool)
            .await?;
        Ok(sum.unwrap_or(0.0))
    }
}

pub(crate) fn now_unix() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// UTC unix timestamp for the first instant of the month containing `now`.
/// No chrono dependency: Howard Hinnant's `civil_from_days`/`days_from_civil`.
pub fn month_start_unix(now: i64) -> i64 {
    let days = now.div_euclid(86_400);
    let (y, m, _) = civil_from_days(days);
    days_from_civil(y, m, 1) * 86_400
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 } as u64; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as u64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe as i64 - 719_468
}

#[cfg(test)]
mod time_tests {
    use super::*;

    #[test]
    fn month_start_lands_on_the_first_at_midnight_utc() {
        // 2026-03-17 12:34:56 UTC
        let now = 1_773_924_896;
        let start = month_start_unix(now);
        let (y, m, d) = civil_from_days(start.div_euclid(86_400));
        assert_eq!((y, m, d), (2026, 3, 1));
        assert_eq!(start % 86_400, 0);
    }
}
