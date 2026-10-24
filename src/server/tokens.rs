use std::path::PathBuf;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessToken {
    pub id: String,
    pub name: String,
    pub token: String,
    pub created_at: u64,
}

pub struct TokenStore {
    path: PathBuf,
    tokens: RwLock<Vec<AccessToken>>,
}

impl TokenStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let tokens = if path.is_file() {
            let content = std::fs::read_to_string(&path)?;
            serde_json::from_str(&content).unwrap_or_else(|_| {
                // Keep the unreadable file instead of overwriting it on the next save.
                let _ = std::fs::rename(&path, path.with_extension("json.corrupt"));
                Vec::new()
            })
        } else {
            Vec::new()
        };
        Ok(Self { path, tokens: RwLock::new(tokens) })
    }

    pub fn list(&self) -> Vec<AccessToken> {
        self.tokens.read().unwrap().clone()
    }

    pub fn create(&self, name: &str) -> Result<AccessToken> {
        let id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
        let token = format!("op_{}", uuid::Uuid::new_v4().simple());
        let created_at =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let entry = AccessToken { id, name: name.trim().to_string(), token, created_at };
        {
            let mut g = self.tokens.write().unwrap();
            g.push(entry.clone());
        }
        self.persist()?;
        Ok(entry)
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let removed = {
            let mut g = self.tokens.write().unwrap();
            let before = g.len();
            g.retain(|t| t.id != id);
            g.len() < before
        };
        if removed {
            self.persist()?;
        }
        Ok(removed)
    }

    pub fn validate(&self, token_str: &str) -> bool {
        let token_str = token_str.trim();
        if token_str.is_empty() {
            return false;
        }
        // No early exit and constant-time per comparison, so timing leaks nothing.
        self.tokens.read().unwrap().iter().fold(false, |ok, t| ok | ct_eq(t.token.as_bytes(), token_str.as_bytes()))
    }

    fn persist(&self) -> Result<()> {
        let body = serde_json::to_string_pretty(&*self.tokens.read().unwrap())?;
        crate::fsutil::write_atomic(&self.path, body.as_bytes(), true)
    }
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}
