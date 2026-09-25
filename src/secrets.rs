//! API keys and tokens, kept out of `config.yaml` and out of anything the
//! server serializes to the browser. Stored as YAML under the state dir,
//! created `0600` on unix.
//!
//! This is at-rest separation, not encryption: anyone with filesystem
//! access to `.orchopork/` reads the plaintext file. Swap in an OS keychain
//! behind this same API if that is not an acceptable tradeoff.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::RwLock;

use crate::error::Result;
use crate::providers::ProviderId;

/// Secret name for the GitHub token used to push run branches.
pub const GITHUB: &str = "github";

pub struct SecretStore {
    path: PathBuf,
    keys: RwLock<BTreeMap<String, String>>,
}

impl SecretStore {
    /// Only a missing file means "no secrets": any other read or parse
    /// error is surfaced, because treating an unreadable file as empty
    /// would make the next `set` silently overwrite every stored key.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let keys = match std::fs::read_to_string(&path) {
            Ok(text) if text.trim().is_empty() => BTreeMap::new(),
            Ok(text) => serde_yaml::from_str(&text)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Self { path, keys: RwLock::new(keys) })
    }

    pub fn get(&self, name: &str) -> Option<String> {
        self.keys.read().unwrap().get(name).filter(|v| !v.is_empty()).cloned()
    }

    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn provider_key(&self, p: ProviderId) -> Option<String> {
        self.get(p.as_str())
    }

    /// Stores `value` under `name`; an empty value removes the entry.
    pub fn set(&self, name: &str, value: &str) -> Result<()> {
        {
            let mut g = self.keys.write().unwrap();
            let value = value.trim();
            if value.is_empty() {
                g.remove(name);
            } else {
                g.insert(name.to_string(), value.to_string());
            }
        }
        self.persist()
    }

    fn persist(&self) -> Result<()> {
        let body = serde_yaml::to_string(&*self.keys.read().unwrap())?;
        crate::fsutil::write_atomic(&self.path, body.as_bytes(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_remove_round_trip_and_persist_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.yaml");

        let s = SecretStore::open(&path).unwrap();
        assert!(!s.has("claude"));
        s.set("claude", " sk-test \n").unwrap();
        assert_eq!(s.provider_key(ProviderId::Claude).as_deref(), Some("sk-test"));

        let reopened = SecretStore::open(&path).unwrap();
        assert_eq!(reopened.get("claude").as_deref(), Some("sk-test"));

        reopened.set("claude", "").unwrap();
        assert!(!SecretStore::open(&path).unwrap().has("claude"));
    }

    #[test]
    fn corrupt_file_is_an_error_not_an_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.yaml");
        std::fs::write(&path, "[not: a map").unwrap();
        assert!(SecretStore::open(&path).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn file_is_not_group_or_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.yaml");
        SecretStore::open(&path).unwrap().set("gemini", "k").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
