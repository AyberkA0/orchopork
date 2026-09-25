//! Provider API keys, kept strictly out of `Wizard` (which is serialized to
//! the browser). Stored as YAML under the state dir, `0600` on unix.
//!
//! This is at-rest obfuscation, not encryption: anyone with filesystem
//! access to `.orchopork/` reads the plaintext file. Swap in an OS keychain
//! (or an encrypted-at-rest store) behind this same API if that's not
//! an acceptable tradeoff for a deployment.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::RwLock;

use crate::error::Result;
use crate::providers::ProviderId;

pub struct SecretStore {
    path: PathBuf,
    keys: RwLock<BTreeMap<ProviderId, String>>,
}

impl SecretStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let keys = match std::fs::read_to_string(&path) {
            Ok(text) => serde_yaml::from_str(&text)?,
            Err(_) => BTreeMap::new(),
        };
        Ok(Self { path, keys: RwLock::new(keys) })
    }

    pub fn get(&self, provider: ProviderId) -> Option<String> {
        self.keys.read().unwrap().get(&provider).cloned()
    }

    pub fn has(&self, provider: ProviderId) -> bool {
        self.keys.read().unwrap().contains_key(&provider)
    }

    pub fn configured_providers(&self) -> Vec<ProviderId> {
        self.keys.read().unwrap().keys().copied().collect()
    }

    pub fn set(&self, provider: ProviderId, key: String) -> Result<()> {
        self.keys.write().unwrap().insert(provider, key);
        self.persist()
    }

    pub fn remove(&self, provider: ProviderId) -> Result<()> {
        self.keys.write().unwrap().remove(&provider);
        self.persist()
    }

    fn persist(&self) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let body = serde_yaml::to_string(&*self.keys.read().unwrap())?;
        std::fs::write(&self.path, body)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
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
        assert!(!s.has(ProviderId::Claude));
        s.set(ProviderId::Claude, "sk-test".into()).unwrap();
        assert_eq!(s.get(ProviderId::Claude).as_deref(), Some("sk-test"));

        let reopened = SecretStore::open(&path).unwrap();
        assert_eq!(reopened.get(ProviderId::Claude).as_deref(), Some("sk-test"));
        assert_eq!(reopened.configured_providers(), vec![ProviderId::Claude]);

        reopened.remove(ProviderId::Claude).unwrap();
        assert!(!SecretStore::open(&path).unwrap().has(ProviderId::Claude));
    }

    #[cfg(unix)]
    #[test]
    fn file_is_not_group_or_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.yaml");
        SecretStore::open(&path).unwrap().set(ProviderId::Gemini, "k".into()).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
