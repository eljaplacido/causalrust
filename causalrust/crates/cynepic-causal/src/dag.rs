use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A Directed Acyclic Graph representing a Structural Causal Model.
///
/// Nodes are named variables; edges represent direct causal relationships.
#[derive(Debug, Clone, Serialize)]
pub struct CausalDag {
    /// The underlying directed graph.
    #[serde(skip)]
    graph: DiGraph<String, ()>,
    /// Map from variable name to node index.
    #[serde(skip)]
    node_map: HashMap<String, NodeIndex>,
    /// Variable names in insertion order.
    variables: Vec<String>,
    /// Edges as (source, target) pairs of variable names.
    edges: Vec<(String, String)>,
}

impl<'de> Deserialize<'de> for CausalDag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct CausalDagData {
            variables: Vec<String>,
            edges: Vec<(String, String)>,
        }
        let data = CausalDagData::deserialize(deserializer)?;
        let mut dag = CausalDag::new();
        for var in &data.variables {
            dag.add_variable(var);
        }
        for (cause, effect) in &data.edges {
            dag.add_edge(cause, effect);
        }
        Ok(dag)
    }
}

impl CausalDag {
    /// Create an empty causal DAG.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            variables: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a variable (node) to the DAG.
    pub fn add_variable(&mut self, name: &str) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(name) {
            return idx;
        }
        let idx = self.graph.add_node(name.to_string());
        self.node_map.insert(name.to_string(), idx);
        self.variables.push(name.to_string());
        idx
    }

    /// Add a causal edge from `cause` to `effect`.
    pub fn add_edge(&mut self, cause: &str, effect: &str) {
        let cause_idx = self.add_variable(cause);
        let effect_idx = self.add_variable(effect);
        self.graph.add_edge(cause_idx, effect_idx, ());
        self.edges.push((cause.to_string(), effect.to_string()));
    }

    /// Get all variable names.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Get number of variables.
    pub fn num_variables(&self) -> usize {
        self.graph.node_count()
    }

    /// Get number of edges.
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get the direct parents (causes) of a variable.
    pub fn parents(&self, name: &str) -> Vec<&str> {
        let Some(&idx) = self.node_map.get(name) else {
            return Vec::new();
        };
        self.graph
            .neighbors_directed(idx, Direction::Incoming)
            .map(|n| self.graph[n].as_str())
            .collect()
    }

    /// Get the direct children (effects) of a variable.
    pub fn children(&self, name: &str) -> Vec<&str> {
        let Some(&idx) = self.node_map.get(name) else {
            return Vec::new();
        };
        self.graph
            .neighbors_directed(idx, Direction::Outgoing)
            .map(|n| self.graph[n].as_str())
            .collect()
    }

    /// Get the underlying petgraph for advanced operations.
    pub fn inner_graph(&self) -> &DiGraph<String, ()> {
        &self.graph
    }

    /// Get the node index for a variable name.
    pub fn node_index(&self, name: &str) -> Option<NodeIndex> {
        self.node_map.get(name).copied()
    }

    /// Check whether the graph is actually acyclic.
    pub fn is_acyclic(&self) -> bool {
        petgraph::algo::is_cyclic_directed(&self.graph) == false
    }
}

impl Default for CausalDag {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smoking_dag() -> CausalDag {
        let mut dag = CausalDag::new();
        dag.add_edge("smoking", "tar");
        dag.add_edge("tar", "cancer");
        dag.add_edge("smoking", "cancer");
        dag
    }

    #[test]
    fn builds_dag_correctly() {
        let dag = smoking_dag();
        assert_eq!(dag.num_variables(), 3);
        assert_eq!(dag.num_edges(), 3);
    }

    #[test]
    fn parents_and_children() {
        let dag = smoking_dag();
        let mut cancer_parents = dag.parents("cancer");
        cancer_parents.sort();
        assert_eq!(cancer_parents, vec!["smoking", "tar"]);
        let mut smoking_children = dag.children("smoking");
        smoking_children.sort();
        assert_eq!(smoking_children, vec!["cancer", "tar"]);
        assert!(dag.parents("smoking").is_empty());
    }

    #[test]
    fn is_acyclic() {
        let dag = smoking_dag();
        assert!(dag.is_acyclic());
    }

    #[test]
    fn duplicate_variable_is_idempotent() {
        let mut dag = CausalDag::new();
        let idx1 = dag.add_variable("X");
        let idx2 = dag.add_variable("X");
        assert_eq!(idx1, idx2);
        assert_eq!(dag.num_variables(), 1);
    }
}
