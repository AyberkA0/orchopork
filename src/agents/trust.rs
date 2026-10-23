//! Workspace trust. Several agent CLIs hold back in a folder you have not
//! trusted: Gemini CLI and Qwen Code drop to "ask before every action",
//! which a run nobody watches can never answer, and Cursor's headless mode
//! stops with a trust prompt. Every orchopork run gets a fresh worktree, so
//! no folder it works in could have been trusted before.
//!
//! orchopork settles this itself, before the agent starts:
//!
//! - with the tool's own flag where it has one (`--skip-trust` for Gemini
//!   CLI, `--trust` for Cursor; see `cli::build_args`), for that run only;
//! - and, for tools that keep a trust list (Gemini CLI, Qwen Code, also
//!   used over ACP), by trusting the folder that holds orchopork's
//!   worktrees, once. Only that folder: never your checkout or home.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::{Dialect, ExternalAgent};

/// The trust list `agent` reads, if it is a tool that keeps one.
fn trust_file(agent: &ExternalAgent) -> Option<PathBuf> {
    let name = Path::new(&agent.command).file_stem()?.to_string_lossy().to_lowercase();
    let dir = match (agent.dialect, name.as_str()) {
        (Dialect::Gemini, _) | (_, "gemini") => ".gemini",
        (Dialect::Qwen, _) | (_, "qwen") => ".qwen",
        _ => return None,
    };
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(dir).join("trustedFolders.json"))
}

/// How these tools write a folder in their list: on Windows lower case
/// with forward slashes.
fn key(p: &Path) -> String {
    let s = crate::fsutil::strip_verbatim(p.to_path_buf()).to_string_lossy().into_owned();
    if cfg!(windows) { s.replace('\\', "/").to_lowercase() } else { s }
}

/// The folder to trust for a run in `cwd`: the worktrees folder when `cwd`
/// is one of orchopork's worktrees, else `cwd` itself.
fn folder(cwd: &Path) -> PathBuf {
    match cwd.parent() {
        Some(p) if p.file_name().is_some_and(|n| n == "worktrees") => p.to_path_buf(),
        _ => cwd.to_path_buf(),
    }
}

/// Whether `list` already trusts `folder` (directly, through a trusted
/// ancestor, or through a `TRUST_PARENT` entry).
fn covered(list: &Map<String, Value>, folder: &str) -> bool {
    let under = |root: &str| folder == root || folder.starts_with(&format!("{}/", root.trim_end_matches('/')));
    list.iter().any(|(k, v)| match v.as_str() {
        Some("TRUST_FOLDER") => under(k),
        Some("TRUST_PARENT") => k.rsplit_once('/').is_some_and(|(parent, _)| !parent.is_empty() && under(parent)),
        _ => false,
    })
}

/// Makes sure `agent` trusts the folder of a run in `cwd`. Returns a note
/// when it added an entry; any failure is only logged, since the agent's
/// own flags still apply.
pub fn ensure(agent: &ExternalAgent, cwd: &Path) -> Option<String> {
    let file = trust_file(agent)?;
    // Only for a tool that is set up here: never create its folder.
    if !file.parent().is_some_and(Path::is_dir) {
        return None;
    }
    let target = key(&folder(cwd));
    let mut list: Map<String, Value> =
        std::fs::read_to_string(&file).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    if covered(&list, &target) {
        return None;
    }
    if list.get(&target).and_then(Value::as_str) == Some("DO_NOT_TRUST") {
        tracing::warn!("{} is marked DO_NOT_TRUST in {}; leaving it alone", target, file.display());
        return None;
    }
    list.insert(target.clone(), Value::String("TRUST_FOLDER".into()));
    let body = serde_json::to_string_pretty(&Value::Object(list)).unwrap_or_default();
    match crate::fsutil::write_atomic(&file, body.as_bytes(), false) {
        Ok(()) => Some(format!("{} now trusts {target} (orchopork's worktrees)", agent.name)),
        Err(e) => {
            tracing::warn!("could not update {}: {e}", file.display());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_ancestors_cover_a_folder() {
        let list: Map<String, Value> = serde_json::from_str(
            r#"{"c:/users/me/work": "TRUST_FOLDER", "c:/users/me/other/repo": "TRUST_PARENT", "c:/x": "DO_NOT_TRUST"}"#,
        )
        .unwrap();
        assert!(covered(&list, "c:/users/me/work"));
        assert!(covered(&list, "c:/users/me/work/p/.orchopork/worktrees"));
        assert!(!covered(&list, "c:/users/me/workshop"));
        assert!(covered(&list, "c:/users/me/other/sibling"), "TRUST_PARENT trusts the parent");
        assert!(!covered(&list, "c:/x/y"));
    }

    #[test]
    fn runs_trust_the_worktrees_folder() {
        let wt = Path::new("/p/.orchopork/worktrees/ab12cd34");
        assert_eq!(folder(wt), Path::new("/p/.orchopork/worktrees"));
        assert_eq!(folder(Path::new("/p")), Path::new("/p"));
        let a = ExternalAgent::new("g", "G", "gemini", &["--acp"]);
        assert!(trust_file(&a).is_some_and(|f| f.ends_with(".gemini/trustedFolders.json")));
        assert!(trust_file(&ExternalAgent::new("c", "C", "claude", &[])).is_none());
    }
}
