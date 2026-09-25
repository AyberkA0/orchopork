//! orchopork: hybrid local/cloud agent orchestration engine.

pub mod bootstrap;
pub mod error;
pub mod git;
pub mod graph;
pub mod providers;
pub mod secrets;
pub mod server;
pub mod skills;
pub mod storage;

pub use error::{Error, Result};
