use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Unique identifier for a node in the workflow graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl NodeId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A node in the workflow graph.
///
/// Each node is an async function that takes the current state and returns
/// the modified state. Nodes can perform LLM calls, database queries,
/// policy checks, or any other async operation.
#[async_trait]
pub trait Node<S: Send + Sync>: Send + Sync {
    /// Execute this node with the given state.
    async fn execute(&self, state: S) -> Result<S, NodeError>;

    /// Human-readable name of this node.
    fn name(&self) -> &str;

    /// The node's unique identifier.
    fn id(&self) -> NodeId {
        NodeId::new(self.name())
    }
}

/// A simple function-based node.
pub struct FnNode<S> {
    name: String,
    func: Box<dyn Fn(S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S, NodeError>> + Send>> + Send + Sync>,
}

impl<S: Send + Sync + 'static> FnNode<S> {
    /// Create a node from an async function.
    pub fn new<F, Fut>(name: impl Into<String>, func: F) -> Self
    where
        F: Fn(S) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<S, NodeError>> + Send + 'static,
    {
        let name = name.into();
        Self {
            name,
            func: Box::new(move |s| Box::pin(func(s))),
        }
    }
}

#[async_trait]
impl<S: Send + Sync + 'static> Node<S> for FnNode<S> {
    async fn execute(&self, state: S) -> Result<S, NodeError> {
        (self.func)(state).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Errors from node execution.
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("Node execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Node timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    #[error("Human approval required at node {node}")]
    HumanApprovalRequired { node: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fn_node_executes() {
        let node = FnNode::new("double", |x: i32| async move { Ok(x * 2) });
        let result = node.execute(5).await.unwrap();
        assert_eq!(result, 10);
    }

    #[test]
    fn node_id_display() {
        let id = NodeId::new("my_node");
        assert_eq!(id.to_string(), "my_node");
    }
}
