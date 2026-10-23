//! What a run produced, for the UI: the files it changed (with a preview
//! of each), and its measurements.
//!
//! Files are served from the run's worktree only: the path must stay inside
//! it (no `..`, no absolute paths, no `.git`, no symlink pointing out). The
//! page renders them in sandboxed frames, and every response carries a
//! sandboxing CSP, so a generated HTML or SVG file never runs with this
//! origin's rights even when opened directly.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use axum::Json;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiError, ApiResult, Shared};
use crate::error::Error;
use crate::git::GitRepo;
use crate::storage::{Run, Step, StepKind};

/// Largest file sent for preview.
const MAX_PREVIEW: u64 = 20 * 1024 * 1024;

/// What the UI should render a file as.
fn kind_of(path: &str) -> &'static str {
    let ext = Path::new(path).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "svg" | "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "ico" | "avif" => "image",
        "html" | "htm" => "html",
        "md" | "markdown" => "markdown",
        "pdf" => "pdf",
        "mp3" | "wav" | "ogg" | "m4a" => "audio",
        "mp4" | "webm" | "mov" => "video",
        "zip" | "gz" | "tar" | "7z" | "exe" | "dll" | "so" | "bin" | "wasm" | "woff" | "woff2" | "ttf" | "otf" => {
            "binary"
        }
        _ => "text",
    }
}

fn content_type(path: &str) -> &'static str {
    let ext = Path::new(path).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "ico" => "image/x-icon",
        "avif" => "image/avif",
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" => "application/json",
        "pdf" => "application/pdf",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "wasm" => "application/wasm",
        _ => match kind_of(path) {
            "binary" => "application/octet-stream",
            _ => "text/plain; charset=utf-8",
        },
    }
}

/// `raw` inside `root`, or why not.
fn confined(root: &Path, raw: &str) -> Result<PathBuf, Error> {
    let bad = || Error::InvalidRequest(format!("{raw:?} is not a file of this run"));
    let mut out = root.to_path_buf();
    for c in Path::new(raw.trim()).components() {
        match c {
            Component::Normal(p) if p != ".git" => out.push(p),
            Component::CurDir => {}
            _ => return Err(bad()),
        }
    }
    let real = crate::fsutil::canonical(&out).map_err(|_| Error::NotFound(raw.to_string()))?;
    let root = crate::fsutil::canonical(root).map_err(|_| bad())?;
    if !real.starts_with(&root) || !real.is_file() {
        return Err(bad());
    }
    Ok(real)
}

/// Lines added and removed per changed file (`None` for binary files).
/// Files not committed yet count as all added.
async fn line_changes(run: &Run) -> BTreeMap<String, (Option<u64>, Option<u64>)> {
    let wt = GitRepo::new(&run.worktree);
    let mut out = wt.numstat(&run.base_commit).await.unwrap_or_default();
    for p in wt.untracked().await.unwrap_or_default() {
        let lines = std::fs::read(Path::new(&run.worktree).join(&p))
            .ok()
            .filter(|b| b.len() < 4 << 20 && !b.contains(&0))
            .map(|b| b.iter().filter(|&&c| c == b'\n').count() as u64);
        out.entry(p).or_insert((lines, lines.map(|_| 0)));
    }
    out
}

/// The run's changed files against its base, uncommitted ones included:
/// `[{path, status, kind, size, added, removed}]`, newest first by mtime.
async fn changed(run: &Run) -> crate::error::Result<Vec<Value>> {
    let wt = GitRepo::new(&run.worktree);
    let mut status: BTreeMap<String, String> = BTreeMap::new();
    for line in wt.changed_names(&run.base_commit).await?.lines() {
        if let Some((s, p)) = line.split_once('\t') {
            status.insert(p.to_string(), s.to_string());
        }
    }
    for p in wt.untracked().await? {
        status.entry(p).or_insert_with(|| "A".into());
    }
    let numstat = line_changes(run).await;
    let mut out: Vec<(std::time::SystemTime, Value)> = Vec::new();
    for (path, st) in status {
        let full = Path::new(&run.worktree).join(&path);
        let meta = std::fs::metadata(&full).ok();
        let (added, removed) = numstat.get(&path).copied().unwrap_or((None, None));
        let mtime = meta.as_ref().and_then(|m| m.modified().ok()).unwrap_or(std::time::UNIX_EPOCH);
        out.push((
            mtime,
            json!({
                "path": path,
                "status": match st.as_str() { "A" => "added", "D" => "deleted", _ => "modified" },
                "kind": kind_of(&path),
                "size": meta.map(|m| m.len()),
                "added": added,
                "removed": removed,
            }),
        ));
    }
    out.sort_by_key(|o| std::cmp::Reverse(o.0));
    Ok(out.into_iter().map(|(_, v)| v).collect())
}

pub async fn files(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    let run = s.engine()?.workspace().store.get_run(&id).await?;
    Ok(Json(json!({ "files": changed(&run).await? })))
}

#[derive(Deserialize)]
pub struct FileQuery {
    path: String,
    /// `1`: send as a download.
    #[serde(default)]
    download: Option<String>,
}

/// The same file at a path-style URL (`/api/runs/<id>/raw/<path>`), so a
/// generated page's relative links (`<img src="pelican.svg">`) resolve to
/// the run's other files.
pub async fn raw(
    State(s): State<Shared>,
    UrlPath((id, path)): UrlPath<(String, String)>,
    Query(q): Query<RawQuery>,
) -> Result<Response, ApiError> {
    file(State(s), UrlPath(id), Query(FileQuery { path, download: q.download })).await
}

#[derive(Deserialize)]
pub struct RawQuery {
    #[serde(default)]
    download: Option<String>,
}

pub async fn file(
    State(s): State<Shared>,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<FileQuery>,
) -> Result<Response, ApiError> {
    let run = s.engine()?.workspace().store.get_run(&id).await?;
    let path = confined(Path::new(&run.worktree), &q.path)?;
    let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    if len > MAX_PREVIEW {
        return Err(Error::InvalidRequest(format!("{} is too large to preview ({len} bytes)", q.path)).into());
    }
    let bytes = tokio::fs::read(&path).await.map_err(Error::from)?;
    let name = path.file_name().map(|n| n.to_string_lossy().replace('"', "")).unwrap_or_default();
    let disposition = if q.download.is_some() {
        format!("attachment; filename=\"{name}\"")
    } else {
        format!("inline; filename=\"{name}\"")
    };
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type(&q.path).to_string()),
            (header::CONTENT_DISPOSITION, disposition),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff".into()),
            // Generated pages may run scripts, but only in an opaque origin:
            // they can never act as this server's page.
            (header::CONTENT_SECURITY_POLICY, "sandbox allow-scripts allow-modals allow-popups".into()),
            (header::CACHE_CONTROL, "no-store".into()),
        ],
        bytes,
    )
        .into_response())
}

// ---- measurements -------------------------------------------------------

#[derive(Default)]
struct Tally {
    turns: u64,
    tools: BTreeMap<String, u64>,
    failures: u64,
    escalations: u64,
    questions: u64,
    prompt: u64,
    cached: u64,
    cache_write: u64,
    completion: u64,
    cost: f64,
    reported_cost: f64,
    ms: u64,
    tool_ms: u64,
    models: BTreeMap<String, u64>,
}

impl Tally {
    fn add(&mut self, s: &Step) {
        let m = &s.meta;
        let n = |k: &str| m[k].as_u64().unwrap_or(0);
        self.prompt += n("prompt_tokens");
        self.cached += n("cache_read_tokens");
        self.cache_write += n("cache_write_tokens");
        self.completion += n("completion_tokens");
        self.cost += s.cost_usd;
        self.reported_cost += m["external"]["reported_cost_usd"].as_f64().unwrap_or(0.0);
        self.ms += n("ms");
        self.tool_ms += n("tool_ms");
        if let Some(model) = m["model"].as_str() {
            *self.models.entry(model.to_string()).or_default() += 1;
        }
        if s.kind == StepKind::Act {
            self.turns += 1;
            let tool = m["tool"].as_str().unwrap_or("?").to_string();
            if tool == "ask" {
                self.questions += 1;
            }
            *self.tools.entry(tool).or_default() += 1;
            if m["ok"] == false {
                self.failures += 1;
            }
            if m["escalated"] == true {
                self.escalations += 1;
            }
        }
    }

    fn json(&self) -> Value {
        json!({
            "turns": self.turns,
            "tools": self.tools,
            "failures": self.failures,
            "escalations": self.escalations,
            "questions": self.questions,
            "tokens": { "input": self.prompt, "cached": self.cached, "cache_write": self.cache_write,
                        "output": self.completion },
            "cache_hit": if self.prompt > 0 { self.cached as f64 / self.prompt as f64 } else { 0.0 },
            "cost_usd": self.cost,
            "reported_cost_usd": self.reported_cost,
            "model_ms": self.ms,
            "tool_ms": self.tool_ms,
            "models": self.models,
        })
    }
}

/// Everything measured about a run: totals, per agent, per model, and a
/// timeline of its steps.
pub async fn measure(run: &Run, steps: &[Step]) -> Value {
    let mut total = Tally::default();
    let mut agents: BTreeMap<String, Tally> = BTreeMap::new();
    let mut by_model: BTreeMap<String, Tally> = BTreeMap::new();
    let root = run.spec.root().map(|r| r.id.clone());
    let (mut verify_pass, mut verify_fail, mut reviews_revise) = (0u64, 0u64, 0u64);
    let mut timeline = Vec::new();
    for s in steps {
        total.add(s);
        // Who spent it: the acting agent, or the commander who answered.
        let who = s.meta["from"].as_str().or(s.meta["agent"].as_str()).map(str::to_string).or_else(|| root.clone());
        if let Some(a) = &who
            && (s.kind == StepKind::Act || s.meta.get("from").is_some())
        {
            agents.entry(a.clone()).or_default().add(s);
        }
        if let Some(m) = s.meta["model"].as_str() {
            by_model.entry(m.to_string()).or_default().add(s);
        }
        match s.kind {
            StepKind::Verify if s.meta["ok"] == true => verify_pass += 1,
            StepKind::Verify => verify_fail += 1,
            StepKind::Review if s.meta["verdict"] == "revise" => reviews_revise += 1,
            _ => {}
        }
        timeline.push(json!({
            "seq": s.seq, "kind": s.kind, "agent": who, "tool": s.meta["tool"], "at": s.created_at,
            "ms": s.meta["ms"], "ok": s.meta["ok"], "cost": s.cost_usd,
            "tokens": s.meta["prompt_tokens"].as_u64().unwrap_or(0) + s.meta["completion_tokens"].as_u64().unwrap_or(0),
        }));
    }
    let (mut added, mut removed, mut files) = (0u64, 0u64, 0u64);
    for (a, r) in line_changes(run).await.values() {
        files += 1;
        added += a.unwrap_or(0);
        removed += r.unwrap_or(0);
    }
    let started = run.created_at;
    let ended = match run.status {
        crate::storage::RunStatus::Running => crate::storage::now_unix(),
        _ => steps.last().map(|s| s.created_at).unwrap_or(run.updated_at),
    };
    json!({
        "run": run.id,
        "goal": run.goal,
        "mode": run.spec.mode,
        "status": run.status,
        "model": run.spec.model,
        "worker_model": run.spec.worker_model,
        "agents_planned": run.spec.agents.len(),
        "retry_of": run.spec.retry_of,
        "wall_secs": (ended - started).max(0),
        "steps": steps.len(),
        "verify": { "passed": verify_pass, "failed": verify_fail },
        "review_rounds": reviews_revise,
        "changes": { "files": files, "added": added, "removed": removed },
        "total": total.json(),
        // The run's real spend (the ledger also counts calls whose output
        // was rewound away), next to what its current steps account for.
        "spent_usd": run.cost_usd,
        "agents": agents.iter().map(|(k, t)| (k.clone(), t.json())).collect::<BTreeMap<_, _>>(),
        "by_model": by_model.iter().map(|(k, t)| (k.clone(), t.json())).collect::<BTreeMap<_, _>>(),
        "timeline": timeline,
    })
}

pub async fn metrics(State(s): State<Shared>, UrlPath(id): UrlPath<String>) -> ApiResult {
    let (run, steps) = s.engine()?.run_detail(&id).await?;
    Ok(Json(measure(&run, &steps).await))
}

#[derive(Deserialize)]
pub struct CompareQuery {
    /// Comma-separated run ids; empty: every run.
    #[serde(default)]
    ids: String,
}

/// Measurements of many runs side by side.
pub async fn compare(State(s): State<Shared>, Query(q): Query<CompareQuery>) -> ApiResult {
    let engine = s.engine()?;
    let wanted: Vec<&str> = q.ids.split(',').map(str::trim).filter(|x| !x.is_empty()).collect();
    let runs = engine.list_runs().await?;
    let mut out = Vec::new();
    for r in runs.iter().filter(|r| wanted.is_empty() || wanted.contains(&r.id.as_str())) {
        let (run, steps) = engine.run_detail(&r.id).await?;
        let mut m = measure(&run, &steps).await;
        if let Some(o) = m.as_object_mut() {
            o.remove("timeline");
        }
        out.push(m);
    }
    Ok(Json(json!({ "runs": out })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_stay_inside_the_worktree() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("art")).unwrap();
        std::fs::write(root.join("art/pelican.svg"), "<svg/>").unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/config"), "x").unwrap();
        assert!(confined(root, "art/pelican.svg").is_ok());
        assert!(confined(root, "./art/pelican.svg").is_ok());
        for bad in ["../x", "/etc/passwd", ".git/config", "art", "art/../../x", "C:\\Windows\\win.ini"] {
            assert!(confined(root, bad).is_err(), "{bad}");
        }
        assert_eq!(kind_of("a/B.SVG"), "image");
        assert_eq!(content_type("p.svg"), "image/svg+xml");
        assert_eq!(kind_of("index.html"), "html");
        assert_eq!(kind_of("src/main.rs"), "text");
    }
}
