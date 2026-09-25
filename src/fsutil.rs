//! Small filesystem helpers shared by every module that persists state.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::Result;

/// Write `bytes` to `path` via a temp file + rename, so a crash mid-write
/// never leaves a truncated config/secrets file behind. With `private`, the
/// file is created `0600` from the start on unix (no world-readable window
/// between create and chmod).
pub fn write_atomic(path: &Path, bytes: &[u8], private: bool) -> Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(
        ".{}.tmp-{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id()
    ));
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        if private {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        #[cfg(not(unix))]
        let _ = private;
        let mut f = opts.open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// `canonicalize` without Windows' verbatim prefix. On Windows the std
/// call returns `\\?\C:\...` (or `\\?\UNC\server\share\...`), which git
/// and many tools reject ("could not create leading directories"). This
/// strips it back to `C:\...` / `\\server\share\...`; elsewhere it is a
/// plain `canonicalize`.
pub fn canonical(path: &Path) -> std::io::Result<PathBuf> {
    Ok(strip_verbatim(std::fs::canonicalize(path)?))
}

pub fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\")
        && rest.as_bytes().get(1) == Some(&b':')
    {
        return PathBuf::from(rest);
    }
    p
}

/// Keep at most `max` chars of `s`, cutting from the middle so both the
/// start (headers, first error) and the end (summary, last error) of long
/// command output survive.
pub fn clip(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let head = max * 2 / 3;
    let tail = max - head;
    let start: String = s.chars().take(head).collect();
    let end: String = s.chars().skip(count - tail).collect();
    format!("{start}\n[... {} chars omitted ...]\n{end}", count - head - tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_keeps_head_and_tail() {
        assert_eq!(clip("short", 10), "short");
        let c = clip(&"a".repeat(50).chars().chain("Z".chars()).collect::<String>(), 12);
        assert!(c.starts_with("aaaaaaaa") && c.ends_with('Z') && c.contains("omitted"));
    }

    #[test]
    fn verbatim_prefixes_are_stripped() {
        assert_eq!(strip_verbatim(PathBuf::from(r"\\?\C:\Users\a\x")), PathBuf::from(r"C:\Users\a\x"));
        assert_eq!(strip_verbatim(PathBuf::from(r"\\?\UNC\srv\share\x")), PathBuf::from(r"\\srv\share\x"));
        assert_eq!(strip_verbatim(PathBuf::from("/home/a")), PathBuf::from("/home/a"));
    }

    #[test]
    fn atomic_write_replaces_content() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("x.yaml");
        write_atomic(&p, b"one", true).unwrap();
        write_atomic(&p, b"two", true).unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "two");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }
}
