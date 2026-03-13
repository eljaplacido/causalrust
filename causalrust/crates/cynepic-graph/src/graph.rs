use crate::checkpoint::Checkpoint;
use crate::hooks::{GraphEvent, GraphHook};
use crate::node::{Node, NodeError, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A stateful workflow graph that routes execution through typed nodes.
///
/// `S` is the state type that flows through the graph. Each node
/// receives the current state and returns the modified state.
pub struct StateGraph<S: Send + Sync + 'static> {
    nodes: HashMap<NodeId, Arc<dyn Node<S>>>,
    edges: HashMap<NodeId, Vec<NodeId>>,
    conditional_edges: HashMap<NodeId, Arc<dyn Fn(&S) -> NodeId + Send + Sync>>,
    entry_node: Option<NodeId>,
    hooks: Vec<Arc<dyn GraphHook>>,
}

impl<S: Send + Sync + 'static> StateGraph<S> {
    /// Create a new empty state graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            conditional_edges: HashMap::new(),
            entry_node: None,
            hooks: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(mut self, node: Arc<dyn Node<S>>) -> Self {
        let id = node.id();
        self.nodes.insert(id, node);
        self
    }

    /// Set the entry node (where execution starts).
    pub fn set_entry(mut self, node_id: NodeId) -> Self {
        self.entry_node = Some(node_id);
        self
    }

    /// Add a fixed edge from one node to another.
    pub fn add_edge(mut self, from: NodeId, to: NodeId) -> Self {
        self.edges.entry(from).or_default().push(to);
        self
    }

    /// Add a conditional edge: a function that examines the state to decide
    /// which node to execute next.
    pub fn add_conditional_edge<F>(mut self, from: NodeId, router: F) -> Self
    where
        F: Fn(&S) -> NodeId + Send + Sync + 'static,
    {
        self.conditional_edges.insert(from, Arc::new(router));
        self
    }

    /// Add an event hook to observe graph execution.
    pub fn add_hook(mut self, hook: Arc<dyn GraphHook>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Emit an event to all registered hooks.
    async fn emit(&self, event: GraphEvent) {
        for hook in &self.hooks {
            hook.on_event(event.clone()).await;
        }
    }

    /// Validate the graph before execution.
    ///
    /// Checks for:
    /// - Entry node is set
    /// - All edge targets reference existing nodes
    /// - No cycles in fixed edges (cycles through conditional edges may be intentional)
    pub fn validate(&self) -> Result<(), GraphError> {
        // Check entry node is set
        let entry = self
            .entry_node
            .as_ref()
            .ok_or(GraphError::NoEntryNode)?;

        // Check entry node exists
        if !self.nodes.contains_key(entry) {
            return Err(GraphError::NodeNotFound(entry.clone()));
        }

        // Check all fixed edge targets exist
        for (from, targets) in &self.edges {
            for to in targets {
                if !self.nodes.contains_key(to) {
                    return Err(GraphError::NodeNotFound(to.clone()));
                }
            }
            if !self.nodes.contains_key(from) {
                return Err(GraphError::NodeNotFound(from.clone()));
            }
        }

        // Cycle detection on fixed edges via DFS
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();
        let mut path = Vec::new();

        // We need to check from all nodes that have fixed edges, not just entry
        for start_node in self.edges.keys() {
            if !visited.contains(start_node) {
                if let Some(cycle_path) = self.dfs_cycle_check(
                    start_node,
                    &mut visited,
                    &mut in_stack,
                    &mut path,
                ) {
                    return Err(GraphError::CycleDetected { path: cycle_path });
                }
            }
        }

        Ok(())
    }

    /// DFS helper for cycle detection on fixed edges only.
    /// Returns Some(cycle_path) if a cycle is found, None otherwise.
    fn dfs_cycle_check(
        &self,
        node: &NodeId,
        visited: &mut HashSet<NodeId>,
        in_stack: &mut HashSet<NodeId>,
        path: &mut Vec<NodeId>,
    ) -> Option<Vec<NodeId>> {
        visited.insert(node.clone());
        in_stack.insert(node.clone());
        path.push(node.clone());

        if let Some(neighbors) = self.edges.get(node) {
            for next in neighbors {
                if !visited.contains(next) {
                    if let Some(cycle) =
                        self.dfs_cycle_check(next, visited, in_stack, path)
                    {
                        return Some(cycle);
                    }
                } else if in_stack.contains(next) {
                    // Found a cycle — extract the cycle portion of the path
                    let cycle_start = path.iter().position(|n| n == next).unwrap();
                    let mut cycle_path: Vec<NodeId> = path[cycle_start..].to_vec();
                    cycle_path.push(next.clone()); // close the cycle
                    return Some(cycle_path);
                }
            }
        }

        path.pop();
        in_stack.remove(node);
        None
    }

    /// Execute the graph starting from the entry node.
    ///
    /// Runs nodes in sequence following edges until no more edges exist
    /// (terminal node) or a maximum step count is reached.
    /// Validates the graph before execution.
    pub async fn execute(&self, initial_state: S, max_steps: usize) -> Result<S, GraphError> {
        self.validate()?;

        let entry = self
            .entry_node
            .clone()
            .ok_or(GraphError::NoEntryNode)?;

        let exec_start = std::time::Instant::now();
        let mut current_id = entry;
        let mut state = initial_state;

        for step in 0..max_steps {
            let node = self
                .nodes
                .get(&current_id)
                .ok_or_else(|| GraphError::NodeNotFound(current_id.clone()))?;

            tracing::debug!(step = step, node = %current_id, "Executing node");
            self.emit(GraphEvent::NodeStarted {
                node: current_id.clone(),
                step,
            })
            .await;

            let node_start = std::time::Instant::now();

            match node.execute(state).await {
                Ok(new_state) => {
                    let duration_ms = node_start.elapsed().as_millis() as u64;
                    self.emit(GraphEvent::NodeCompleted {
                        node: current_id.clone(),
                        step,
                        duration_ms,
                    })
                    .await;
                    state = new_state;
                }
                Err(e) => {
                    self.emit(GraphEvent::NodeFailed {
                        node: current_id.clone(),
                        step,
                        error: e.to_string(),
                    })
                    .await;
                    return Err(GraphError::NodeFailed {
                        node: current_id.clone(),
                        source: e,
                    });
                }
            }

            // Check for conditional edge first
            if let Some(router) = self.conditional_edges.get(&current_id) {
                let next = router(&state);
                self.emit(GraphEvent::RouteDecision {
                    from: current_id.clone(),
                    to: next.clone(),
                    step,
                })
                .await;
                current_id = next;
                continue;
            }

            // Check for fixed edges
            if let Some(next_nodes) = self.edges.get(&current_id) {
                if let Some(next) = next_nodes.first() {
                    current_id = next.clone();
                    continue;
                }
            }

            // No outgoing edges — terminal node
            tracing::info!(node = %current_id, "Reached terminal node");
            let total_ms = exec_start.elapsed().as_millis() as u64;
            self.emit(GraphEvent::ExecutionCompleted {
                total_steps: step + 1,
                total_ms,
            })
            .await;
            return Ok(state);
        }

        Err(GraphError::MaxStepsExceeded(max_steps))
    }

    /// Execute with per-node timeout.
    ///
    /// Each node must complete within `node_timeout`, otherwise a
    /// [`GraphError::NodeTimedOut`] error is returned.
    pub async fn execute_with_timeout(
        &self,
        initial_state: S,
        max_steps: usize,
        node_timeout: std::time::Duration,
    ) -> Result<S, GraphError> {
        self.validate()?;

        let entry = self
            .entry_node
            .clone()
            .ok_or(GraphError::NoEntryNode)?;

        let exec_start = std::time::Instant::now();
        let mut current_id = entry;
        let mut state = initial_state;

        for step in 0..max_steps {
            let node = self
                .nodes
                .get(&current_id)
                .ok_or_else(|| GraphError::NodeNotFound(current_id.clone()))?;

            tracing::debug!(step = step, node = %current_id, "Executing node");
            self.emit(GraphEvent::NodeStarted {
                node: current_id.clone(),
                step,
            })
            .await;

            let node_start = std::time::Instant::now();

            let result = tokio::time::timeout(node_timeout, node.execute(state)).await;

            match result {
                Ok(Ok(new_state)) => {
                    let duration_ms = node_start.elapsed().as_millis() as u64;
                    self.emit(GraphEvent::NodeCompleted {
                        node: current_id.clone(),
                        step,
                        duration_ms,
                    })
                    .await;
                    state = new_state;
                }
                Ok(Err(e)) => {
                    self.emit(GraphEvent::NodeFailed {
                        node: current_id.clone(),
                        step,
                        error: e.to_string(),
                    })
                    .await;
                    return Err(GraphError::NodeFailed {
                        node: current_id.clone(),
                        source: e,
                    });
                }
                Err(_elapsed) => {
                    let timeout_ms = node_timeout.as_millis() as u64;
                    self.emit(GraphEvent::NodeFailed {
                        node: current_id.clone(),
                        step,
                        error: format!("Timed out after {}ms", timeout_ms),
                    })
                    .await;
                    return Err(GraphError::NodeTimedOut {
                        node: current_id.clone(),
                        timeout_ms,
                    });
                }
            }

            // Check for conditional edge first
            if let Some(router) = self.conditional_edges.get(&current_id) {
                let next = router(&state);
                self.emit(GraphEvent::RouteDecision {
                    from: current_id.clone(),
                    to: next.clone(),
                    step,
                })
                .await;
                current_id = next;
                continue;
            }

            // Check for fixed edges
            if let Some(next_nodes) = self.edges.get(&current_id) {
                if let Some(next) = next_nodes.first() {
                    current_id = next.clone();
                    continue;
                }
            }

            // No outgoing edges — terminal node
            tracing::info!(node = %current_id, "Reached terminal node");
            let total_ms = exec_start.elapsed().as_millis() as u64;
            self.emit(GraphEvent::ExecutionCompleted {
                total_steps: step + 1,
                total_ms,
            })
            .await;
            return Ok(state);
        }

        Err(GraphError::MaxStepsExceeded(max_steps))
    }
}

impl<S: Send + Sync + Serialize + for<'de> Deserialize<'de> + 'static> StateGraph<S> {
    /// Resume execution from a checkpoint.
    ///
    /// Starts from `checkpoint.next_node` with `checkpoint.state`, using the
    /// remaining step budget (`max_steps - checkpoint.steps_completed`).
    pub async fn resume(
        &self,
        checkpoint: Checkpoint<S>,
        max_steps: usize,
    ) -> Result<S, GraphError> {
        let remaining_steps = max_steps.saturating_sub(checkpoint.steps_completed);
        if remaining_steps == 0 {
            return Err(GraphError::MaxStepsExceeded(max_steps));
        }

        // Temporarily override the entry node to the checkpoint's next_node
        // We do this by executing the inner loop directly
        let exec_start = std::time::Instant::now();
        let mut current_id = checkpoint.next_node;
        let mut state = checkpoint.state;

        for step in 0..remaining_steps {
            let global_step = checkpoint.steps_completed + step;
            let node = self
                .nodes
                .get(&current_id)
                .ok_or_else(|| GraphError::NodeNotFound(current_id.clone()))?;

            tracing::debug!(step = global_step, node = %current_id, "Executing node (resumed)");
            self.emit(GraphEvent::NodeStarted {
                node: current_id.clone(),
                step: global_step,
            })
            .await;

            let node_start = std::time::Instant::now();

            match node.execute(state).await {
                Ok(new_state) => {
                    let duration_ms = node_start.elapsed().as_millis() as u64;
                    self.emit(GraphEvent::NodeCompleted {
                        node: current_id.clone(),
                        step: global_step,
                        duration_ms,
                    })
                    .await;
                    state = new_state;
                }
                Err(e) => {
                    self.emit(GraphEvent::NodeFailed {
                        node: current_id.clone(),
                        step: global_step,
                        error: e.to_string(),
                    })
                    .await;
                    return Err(GraphError::NodeFailed {
                        node: current_id.clone(),
                        source: e,
                    });
                }
            }

            // Check for conditional edge first
            if let Some(router) = self.conditional_edges.get(&current_id) {
                let next = router(&state);
                self.emit(GraphEvent::RouteDecision {
                    from: current_id.clone(),
                    to: next.clone(),
                    step: global_step,
                })
                .await;
                current_id = next;
                continue;
            }

            // Check for fixed edges
            if let Some(next_nodes) = self.edges.get(&current_id) {
                if let Some(next) = next_nodes.first() {
                    current_id = next.clone();
                    continue;
                }
            }

            // No outgoing edges — terminal node
            tracing::info!(node = %current_id, "Reached terminal node (resumed)");
            let total_ms = exec_start.elapsed().as_millis() as u64;
            self.emit(GraphEvent::ExecutionCompleted {
                total_steps: global_step + 1,
                total_ms,
            })
            .await;
            return Ok(state);
        }

        Err(GraphError::MaxStepsExceeded(max_steps))
    }
}

impl<S: Send + Sync + 'static> Default for StateGraph<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from graph execution.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    /// No entry node has been set on the graph.
    #[error("No entry node set")]
    NoEntryNode,

    /// A referenced node was not found in the graph.
    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),

    /// A node failed during execution.
    #[error("Node {node} failed: {source}")]
    NodeFailed { node: NodeId, source: NodeError },

    /// Execution exceeded the maximum allowed steps.
    #[error("Exceeded maximum steps: {0}")]
    MaxStepsExceeded(usize),

    /// A cycle was detected in the fixed edges of the graph.
    #[error("Cycle detected in fixed edges: {path:?}")]
    CycleDetected { path: Vec<NodeId> },

    /// A node exceeded its execution timeout.
    #[error("Node {node} timed out after {timeout_ms}ms")]
    NodeTimedOut { node: NodeId, timeout_ms: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::EventCollector;
    use crate::node::FnNode;

    #[tokio::test]
    async fn linear_graph_execution() {
        let add_one = Arc::new(FnNode::new("add_one", |x: i32| async move { Ok(x + 1) }));
        let double = Arc::new(FnNode::new("double", |x: i32| async move { Ok(x * 2) }));

        let graph = StateGraph::new()
            .add_node(add_one)
            .add_node(double)
            .set_entry(NodeId::new("add_one"))
            .add_edge(NodeId::new("add_one"), NodeId::new("double"));

        let result = graph.execute(5, 10).await.unwrap();
        // 5 + 1 = 6, 6 * 2 = 12
        assert_eq!(result, 12);
    }

    #[tokio::test]
    async fn conditional_routing() {
        let check = Arc::new(FnNode::new("check", |x: i32| async move { Ok(x) }));
        let positive = Arc::new(FnNode::new("positive", |x: i32| async move { Ok(x * 10) }));
        let negative = Arc::new(FnNode::new("negative", |x: i32| async move { Ok(x * -1) }));

        let graph = StateGraph::new()
            .add_node(check)
            .add_node(positive)
            .add_node(negative)
            .set_entry(NodeId::new("check"))
            .add_conditional_edge(NodeId::new("check"), |x: &i32| {
                if *x > 0 {
                    NodeId::new("positive")
                } else {
                    NodeId::new("negative")
                }
            });

        let result = graph.execute(5, 10).await.unwrap();
        assert_eq!(result, 50); // positive path: 5 * 10

        let result = graph.execute(-3, 10).await.unwrap();
        assert_eq!(result, 3); // negative path: -3 * -1
    }

    #[tokio::test]
    async fn validate_valid_graph() {
        let a = Arc::new(FnNode::new("a", |x: i32| async move { Ok(x + 1) }));
        let b = Arc::new(FnNode::new("b", |x: i32| async move { Ok(x * 2) }));
        let c = Arc::new(FnNode::new("c", |x: i32| async move { Ok(x + 3) }));

        let graph = StateGraph::new()
            .add_node(a)
            .add_node(b)
            .add_node(c)
            .set_entry(NodeId::new("a"))
            .add_edge(NodeId::new("a"), NodeId::new("b"))
            .add_edge(NodeId::new("b"), NodeId::new("c"));

        assert!(graph.validate().is_ok());
    }

    #[tokio::test]
    async fn validate_cycle_detected() {
        let a = Arc::new(FnNode::new("a", |x: i32| async move { Ok(x + 1) }));
        let b = Arc::new(FnNode::new("b", |x: i32| async move { Ok(x * 2) }));
        let c = Arc::new(FnNode::new("c", |x: i32| async move { Ok(x + 3) }));

        let graph = StateGraph::new()
            .add_node(a)
            .add_node(b)
            .add_node(c)
            .set_entry(NodeId::new("a"))
            .add_edge(NodeId::new("a"), NodeId::new("b"))
            .add_edge(NodeId::new("b"), NodeId::new("c"))
            .add_edge(NodeId::new("c"), NodeId::new("a")); // cycle: a -> b -> c -> a

        let result = graph.validate();
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::CycleDetected { path } => {
                // The cycle path should contain the cycle nodes
                assert!(path.len() >= 2, "Cycle path should have at least 2 nodes");
            }
            other => panic!("Expected CycleDetected, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn node_timeout() {
        let slow = Arc::new(FnNode::new("slow", |x: i32| async move {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            Ok(x + 1)
        }));

        let graph = StateGraph::new()
            .add_node(slow)
            .set_entry(NodeId::new("slow"));

        let result = graph
            .execute_with_timeout(5, 10, std::time::Duration::from_millis(50))
            .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::NodeTimedOut { node, timeout_ms } => {
                assert_eq!(node, NodeId::new("slow"));
                assert_eq!(timeout_ms, 50);
            }
            other => panic!("Expected NodeTimedOut, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn event_hooks_collect_events() {
        let collector = Arc::new(EventCollector::new());

        let add_one = Arc::new(FnNode::new("add_one", |x: i32| async move { Ok(x + 1) }));
        let double = Arc::new(FnNode::new("double", |x: i32| async move { Ok(x * 2) }));

        let graph = StateGraph::new()
            .add_node(add_one)
            .add_node(double)
            .set_entry(NodeId::new("add_one"))
            .add_edge(NodeId::new("add_one"), NodeId::new("double"))
            .add_hook(collector.clone());

        let result = graph.execute(5, 10).await.unwrap();
        assert_eq!(result, 12);

        // Should have: NodeStarted(add_one), NodeCompleted(add_one),
        //              NodeStarted(double), NodeCompleted(double),
        //              ExecutionCompleted
        let events = collector.events();
        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], GraphEvent::NodeStarted { node, .. } if node.0 == "add_one"));
        assert!(matches!(&events[1], GraphEvent::NodeCompleted { node, .. } if node.0 == "add_one"));
        assert!(matches!(&events[2], GraphEvent::NodeStarted { node, .. } if node.0 == "double"));
        assert!(matches!(&events[3], GraphEvent::NodeCompleted { node, .. } if node.0 == "double"));
        assert!(matches!(&events[4], GraphEvent::ExecutionCompleted { .. }));
        assert_eq!(collector.node_count(), 2);
    }

    #[tokio::test]
    async fn checkpoint_serialize_deserialize() {
        use crate::checkpoint::Checkpoint;

        let cp = Checkpoint::with_reason(42i32, NodeId::new("double"), 1, "human_approval_required");

        let json = cp.to_json().unwrap();
        let restored: Checkpoint<i32> = Checkpoint::from_json(&json).unwrap();

        assert_eq!(restored.state, 42);
        assert_eq!(restored.next_node, NodeId::new("double"));
        assert_eq!(restored.steps_completed, 1);
        assert_eq!(restored.reason.as_deref(), Some("human_approval_required"));
    }

    #[tokio::test]
    async fn resume_from_checkpoint() {
        use crate::checkpoint::Checkpoint;

        let add_one = Arc::new(FnNode::new("add_one", |x: i32| async move { Ok(x + 1) }));
        let double = Arc::new(FnNode::new("double", |x: i32| async move { Ok(x * 2) }));

        let graph = StateGraph::new()
            .add_node(add_one)
            .add_node(double)
            .set_entry(NodeId::new("add_one"))
            .add_edge(NodeId::new("add_one"), NodeId::new("double"));

        // Simulate: add_one already ran on state 5 -> 6, checkpoint before double
        let checkpoint = Checkpoint::new(6i32, NodeId::new("double"), 1);

        let result = graph.resume(checkpoint, 10).await.unwrap();
        // double(6) = 12
        assert_eq!(result, 12);
    }
}
