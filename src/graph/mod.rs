//! Execution graph. Nodes are agent phases; every transition is checkpointed
//! through `storage::Checkpointer` (SQLite row + git commit). The scheduler,
//! LLM gateway and circuit breaker land here next.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Planner,
    Coder,
    Critic,
    TestRunner,
    ToolExecutor,
}

impl NodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeKind::Planner => "planner",
            NodeKind::Coder => "coder",
            NodeKind::Critic => "critic",
            NodeKind::TestRunner => "test_runner",
            NodeKind::ToolExecutor => "tool_executor",
        }
    }
}
