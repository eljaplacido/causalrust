//! Event hooks for observing graph execution.
//!
//! Implement [`GraphHook`] to receive [`GraphEvent`]s during execution.
//! Use [`EventCollector`] for testing or [`TracingHook`] for structured logging.

use crate::node::NodeId;

/// Events emitted during graph execution.
#[derive(Debug, Clone)]
pub enum GraphEvent {
    /// A node is about to execute.
    NodeStarted { node: NodeId, step: usize },
    /// A node completed successfully.
    NodeCompleted {
        node: NodeId,
        step: usize,
        duration_ms: u64,
    },
    /// A node failed.
    NodeFailed {
        node: NodeId,
        step: usize,
        error: String,
    },
    /// Execution completed.
    ExecutionCompleted { total_steps: usize, total_ms: u64 },
    /// A routing decision was made (conditional edge).
    RouteDecision {
        from: NodeId,
        to: NodeId,
        step: usize,
    },
}

/// Trait for receiving graph execution events.
///
/// Implement this to add logging, metrics, tracing, etc.
#[async_trait::async_trait]
pub trait GraphHook: Send + Sync {
    /// Called when a graph event occurs.
    async fn on_event(&self, event: GraphEvent);
}

/// A simple hook that collects all events into a vec.
///
/// Useful for testing and assertions about execution flow.
#[derive(Debug, Default)]
pub struct EventCollector {
    events: std::sync::Mutex<Vec<GraphEvent>>,
}

impl EventCollector {
    /// Create a new empty event collector.
    pub fn new() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Get a snapshot of all collected events.
    pub fn events(&self) -> Vec<GraphEvent> {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Count the number of `NodeCompleted` events.
    pub fn node_count(&self) -> usize {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e, GraphEvent::NodeCompleted { .. }))
            .count()
    }
}

#[async_trait::async_trait]
impl GraphHook for EventCollector {
    async fn on_event(&self, event: GraphEvent) {
        self.events.lock().unwrap_or_else(|e| e.into_inner()).push(event);
    }
}

/// A hook that logs events via the `tracing` crate.
///
/// Emits structured log messages at appropriate levels:
/// - `debug` for NodeStarted and RouteDecision
/// - `info` for NodeCompleted and ExecutionCompleted
/// - `error` for NodeFailed
pub struct TracingHook;

#[async_trait::async_trait]
impl GraphHook for TracingHook {
    async fn on_event(&self, event: GraphEvent) {
        match &event {
            GraphEvent::NodeStarted { node, step } => {
                tracing::debug!(node = %node, step = step, "Node started");
            }
            GraphEvent::NodeCompleted {
                node,
                step,
                duration_ms,
            } => {
                tracing::info!(node = %node, step = step, duration_ms = duration_ms, "Node completed");
            }
            GraphEvent::NodeFailed { node, step, error } => {
                tracing::error!(node = %node, step = step, error = error, "Node failed");
            }
            GraphEvent::ExecutionCompleted {
                total_steps,
                total_ms,
            } => {
                tracing::info!(total_steps = total_steps, total_ms = total_ms, "Execution completed");
            }
            GraphEvent::RouteDecision { from, to, step } => {
                tracing::debug!(from = %from, to = %to, step = step, "Route decision");
            }
        }
    }
}
