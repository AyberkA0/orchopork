//! orchopork: a lean hybrid local/cloud agent orchestration engine.
//!
//! - [`bootstrap::Workspace`] opens a workspace's state (`.orchopork/`).
//! - [`graph::Engine`] runs goals autonomously in isolated git worktrees.
//! - [`providers::Gateway`] routes LLM calls and enforces the monthly budget.
//! - [`skills`] are declarative prompt modifiers, tools and validators.
//! - [`server`] is the local HTTP API + embedded dashboard.

pub mod acp;
pub mod bootstrap;
pub mod config;
pub mod error;
pub mod fsutil;
pub mod git;
pub mod graph;
pub mod providers;
pub mod secrets;
pub mod server;
pub mod skills;
pub mod storage;

pub use error::{Error, Result};
