//! SQLite persistence (WAL) for threads, snapshots and the spend ledger.

mod checkpoint;

use std::path::Path;
use std::str::FromStr;

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};

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
