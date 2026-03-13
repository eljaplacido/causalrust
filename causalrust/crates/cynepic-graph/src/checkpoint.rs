//! Checkpointing for pause/resume of graph execution.
//!
//! A [`Checkpoint`] captures the state of a graph execution at a specific point,
//! enabling human-in-the-loop workflows where execution can be paused, inspected,
//! and resumed.

use crate::node::NodeId;
use serde::{Deserialize, Serialize};

/// A checkpoint captures the state of a graph execution at a specific point.
///
/// This enables pause/resume for human-in-the-loop workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint<S: Serialize> {
    /// The current state.
    pub state: S,
    /// Which node to execute next.
    pub next_node: NodeId,
    /// How many steps have been executed so far.
    pub steps_completed: usize,
    /// When the checkpoint was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Optional reason for checkpoint (e.g., "human_approval_required").
    pub reason: Option<String>,
}

impl<S: Serialize + for<'de> Deserialize<'de>> Checkpoint<S> {
    /// Create a new checkpoint with the given state, next node, and step count.
    pub fn new(state: S, next_node: NodeId, steps_completed: usize) -> Self {
        Self {
            state,
            next_node,
            steps_completed,
            created_at: chrono::Utc::now(),
            reason: None,
        }
    }

    /// Create a checkpoint with a reason string.
    pub fn with_reason(
        state: S,
        next_node: NodeId,
        steps_completed: usize,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            state,
            next_node,
            steps_completed,
            created_at: chrono::Utc::now(),
            reason: Some(reason.into()),
        }
    }

    /// Serialize this checkpoint to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize a checkpoint from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}
