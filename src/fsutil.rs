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
    // Unique per call: two writers of the same file in one process must not
    // share (and rename away) each other's temp file.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = dir.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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

/// Standard base64 (RFC 4648, with padding).
pub fn base64(input: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(T[(n >> (18 - 6 * i)) as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Decodes standard base64 (padding optional, whitespace ignored).
pub fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let val = |c: u8| match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' | b'-' => Some(62),
        b'/' | b'_' => Some(63),
        _ => None,
    };
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace() && *b != b'=').collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        if chunk.len() == 1 {
            return None;
        }
        let mut n = 0u32;
        for (i, c) in chunk.iter().enumerate() {
            n |= u32::from(val(*c)?) << (18 - 6 * i);
        }
        let bytes = n.to_be_bytes();
        out.extend_from_slice(&bytes[1..chunk.len()]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips() {
        for s in ["", "f", "fo", "foo", "foob", "fooba", "foobar"] {
            assert_eq!(base64_decode(&base64(s.as_bytes())).unwrap(), s.as_bytes(), "{s}");
        }
        assert_eq!(base64_decode("Zm9v\nYmFy").unwrap(), b"foobar");
        assert!(base64_decode("Zm9v*").is_none());
    }

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
