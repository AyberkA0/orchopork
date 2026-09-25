use std::path::PathBuf;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("sqlite: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("git: {0}")]
    Git(String),
    #[error("skill {}: {msg}", .path.display())]
    SkillFile { path: PathBuf, msg: String },
    #[error("skill: {0}")]
    Skill(String),
    #[error("wizard: {0}")]
    Wizard(String),
    #[error("not found: {0}")]
    NotFound(String),
}
