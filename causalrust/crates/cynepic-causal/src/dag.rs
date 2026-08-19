//! Structural causal models as directed acyclic graphs.
//!
//! # The invariant the name promises
//!
//! `add_edge` previously returned `()` and accepted anything, including cycles
//! and self-loops (finding C11). A type called `CausalDag` holding a cyclic
//! graph invalidates every algorithm built on it: d-separation, backdoor search
//! and the counterfactual engine all assume acyclicity, and on a cyclic graph
//! they either return nonsense or fail to terminate.
//!
//! `add_edge` now returns `Result` and rejects any edge that would close a
//! cycle, naming the cycle it found. The check is a reachability query from the
//! proposed target back to its source, so it costs one traversal per edge and
//! makes the invariant true by construction rather than by convention.
//!
//! # Observed and latent variables
//!
//! Nodes carry a [`VarKind`]. Marking a variable [`VarKind::Latent`] states that
//! it exists in the causal structure but cannot be measured — which is what
//! makes it possible for identification to *fail* rather than return an
//! adjustment set nobody can condition on (finding C2).
//!
//! This is also the first piece of per-node metadata, and the natural place to
//! hang the edge confidence and provenance that `research.md` recommendation 3
//! asks for.

use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::DagError;

/// Whether a variable can be measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VarKind {
    /// Measured, and therefore available to condition on.
    #[default]
    Observed,
    /// Present in the causal structure but not measurable.
    ///
    /// An adjustment set containing a latent variable is not a strategy, it is
    /// a restatement of the problem.
    Latent,
}

/// A directed acyclic graph representing a structural causal model.
///
/// Acyclicity is enforced at insertion, so any `CausalDag` value is acyclic.
#[derive(Debug, Clone, Serialize)]
pub struct CausalDag {
    #[serde(skip)]
    graph: DiGraph<String, ()>,
    #[serde(skip)]
    node_map: HashMap<String, NodeIndex>,
    /// Variable names in insertion order.
    variables: Vec<String>,
    /// Edges as `(source, target)` name pairs.
    edges: Vec<(String, String)>,
    /// Variables marked latent.
    latent: HashSet<String>,
}

impl<'de> Deserialize<'de> for CausalDag {
    /// Rebuilds the petgraph from the serialised name lists.
    ///
    /// Deserialisation goes through `add_edge`, so a serialised cyclic graph is
    /// rejected on load rather than producing a `CausalDag` that violates its
    /// own invariant. This is the reason the impl is hand-written.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        #[derive(Deserialize)]
        struct CausalDagData {
            variables: Vec<String>,
            edges: Vec<(String, String)>,
            #[serde(default)]
            latent: HashSet<String>,
        }

        let data = CausalDagData::deserialize(deserializer)?;
        let mut dag = CausalDag::new();
        for var in &data.variables {
            dag.add_variable(var);
        }
        for (cause, effect) in &data.edges {
            dag.add_edge(cause, effect).map_err(D::Error::custom)?;
        }
        for name in &data.latent {
            dag.mark_latent(name).map_err(D::Error::custom)?;
        }
        Ok(dag)
    }
}

impl CausalDag {
    /// Create an empty DAG.
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            variables: Vec::new(),
            edges: Vec::new(),
            latent: HashSet::new(),
        }
    }

    /// Add a variable, or return the existing index if it is already present.
    pub fn add_variable(&mut self, name: &str) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(name) {
            return idx;
        }
        let idx = self.graph.add_node(name.to_string());
        self.node_map.insert(name.to_string(), idx);
        self.variables.push(name.to_string());
        idx
    }

    /// Add a causal edge `cause -> effect`, creating either variable if needed.
    ///
    /// # Errors
    ///
    /// - [`DagError::SelfLoop`] if `cause == effect`.
    /// - [`DagError::WouldCreateCycle`] if `cause` is already reachable from
    ///   `effect`, naming the path that would close.
    ///
    /// Both were previously accepted silently (finding C11).
    pub fn add_edge(&mut self, cause: &str, effect: &str) -> Result<(), DagError> {
        if cause == effect {
            return Err(DagError::SelfLoop {
                variable: cause.to_string(),
            });
        }

        let cause_idx = self.add_variable(cause);
        let effect_idx = self.add_variable(effect);

        // The edge closes a cycle exactly when `cause` is already reachable
        // from `effect`. Checking before inserting keeps the graph valid at
        // every point rather than inserting and rolling back.
        if let Some(path) = self.path_between(effect_idx, cause_idx) {
            return Err(DagError::WouldCreateCycle {
                cause: cause.to_string(),
                effect: effect.to_string(),
                path,
            });
        }

        self.graph.add_edge(cause_idx, effect_idx, ());
        self.edges.push((cause.to_string(), effect.to_string()));
        Ok(())
    }

    /// Add several edges, stopping at the first rejection.
    ///
    /// # Errors
    ///
    /// As [`Self::add_edge`]. Edges before the failure remain applied — the
    /// graph stays acyclic either way, so a partial application is still valid.
    pub fn add_edges<'a, I>(&mut self, edges: I) -> Result<(), DagError>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        for (cause, effect) in edges {
            self.add_edge(cause, effect)?;
        }
        Ok(())
    }

    /// Mark a variable unobserved.
    ///
    /// # Errors
    ///
    /// [`DagError::UnknownVariable`] if the name is not in the graph. Marking a
    /// variable that does not exist is always a typo, and silently creating it
    /// would produce a graph with an isolated latent node nobody intended.
    pub fn mark_latent(&mut self, name: &str) -> Result<(), DagError> {
        if !self.node_map.contains_key(name) {
            return Err(DagError::UnknownVariable {
                name: name.to_string(),
                known: self.variables.clone(),
            });
        }
        self.latent.insert(name.to_string());
        Ok(())
    }

    /// Whether a variable is marked latent.
    pub fn is_latent(&self, name: &str) -> bool {
        self.latent.contains(name)
    }

    /// The kind of a variable, or `None` if it is not in the graph.
    pub fn var_kind(&self, name: &str) -> Option<VarKind> {
        if !self.node_map.contains_key(name) {
            return None;
        }
        Some(if self.is_latent(name) {
            VarKind::Latent
        } else {
            VarKind::Observed
        })
    }

    /// Names of every latent variable, sorted.
    pub fn latent_variables(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.latent.iter().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    /// Names of every observed variable, in insertion order.
    pub fn observed_variables(&self) -> Vec<&str> {
        self.variables
            .iter()
            .map(String::as_str)
            .filter(|n| !self.latent.contains(*n))
            .collect()
    }

    /// Whether the graph contains a variable.
    pub fn contains(&self, name: &str) -> bool {
        self.node_map.contains_key(name)
    }

    /// All variable names, in insertion order.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Edges as `(cause, effect)` name pairs.
    pub fn edges(&self) -> &[(String, String)] {
        &self.edges
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Direct causes of a variable. Empty if the variable is unknown.
    pub fn parents(&self, name: &str) -> Vec<&str> {
        let Some(&idx) = self.node_map.get(name) else {
            return Vec::new();
        };
        self.graph
            .neighbors_directed(idx, Direction::Incoming)
            .map(|n| self.graph[n].as_str())
            .collect()
    }

    /// Direct effects of a variable. Empty if the variable is unknown.
    pub fn children(&self, name: &str) -> Vec<&str> {
        let Some(&idx) = self.node_map.get(name) else {
            return Vec::new();
        };
        self.graph
            .neighbors_directed(idx, Direction::Outgoing)
            .map(|n| self.graph[n].as_str())
            .collect()
    }

    /// Every variable reachable by following edges forward from `name`,
    /// excluding `name` itself.
    pub fn descendants(&self, name: &str) -> HashSet<String> {
        let mut out = HashSet::new();
        let Some(&start) = self.node_map.get(name) else {
            return out;
        };
        let mut queue = VecDeque::from([start]);
        let mut seen = HashSet::from([start]);
        while let Some(node) = queue.pop_front() {
            for child in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if seen.insert(child) {
                    out.insert(self.graph[child].clone());
                    queue.push_back(child);
                }
            }
        }
        out
    }

    /// The underlying petgraph, for algorithms this type does not wrap.
    pub fn inner_graph(&self) -> &DiGraph<String, ()> {
        &self.graph
    }

    /// Node index for a variable name.
    pub fn node_index(&self, name: &str) -> Option<NodeIndex> {
        self.node_map.get(name).copied()
    }

    /// Always `true`. Retained so existing callers keep compiling.
    ///
    /// Acyclicity is now an invariant enforced by [`Self::add_edge`], so this
    /// can no longer return `false` for a `CausalDag` built through the public
    /// API. It is checked in tests rather than trusted.
    pub fn is_acyclic(&self) -> bool {
        !petgraph::algo::is_cyclic_directed(&self.graph)
    }

    /// A directed path from `from` to `to`, as variable names, if one exists.
    fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![self.graph[from].clone()]);
        }
        let mut prev: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut seen = HashSet::from([from]);
        let mut queue = VecDeque::from([from]);

        while let Some(node) = queue.pop_front() {
            for next in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if !seen.insert(next) {
                    continue;
                }
                prev.insert(next, node);
                if next == to {
                    let mut path = vec![self.graph[to].clone()];
                    let mut cur = to;
                    while let Some(&p) = prev.get(&cur) {
                        path.push(self.graph[p].clone());
                        cur = p;
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(next);
            }
        }
        None
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
        dag.add_edges([("smoking", "tar"), ("tar", "cancer"), ("smoking", "cancer")])
            .expect("acyclic");
        dag
    }

    #[test]
    fn builds_dag_correctly() {
        let dag = smoking_dag();
        assert_eq!(dag.num_variables(), 3);
        assert_eq!(dag.num_edges(), 3);
        assert!(dag.is_acyclic());
    }

    #[test]
    fn parents_and_children() {
        let dag = smoking_dag();
        let mut cancer_parents = dag.parents("cancer");
        cancer_parents.sort_unstable();
        assert_eq!(cancer_parents, vec!["smoking", "tar"]);
        let mut smoking_children = dag.children("smoking");
        smoking_children.sort_unstable();
        assert_eq!(smoking_children, vec!["cancer", "tar"]);
        assert!(dag.parents("smoking").is_empty());
    }

    #[test]
    fn duplicate_variable_is_idempotent() {
        let mut dag = CausalDag::new();
        let a = dag.add_variable("X");
        let b = dag.add_variable("X");
        assert_eq!(a, b);
        assert_eq!(dag.num_variables(), 1);
    }

    #[test]
    fn self_loop_is_rejected() {
        // C11.
        let mut dag = CausalDag::new();
        let err = dag.add_edge("X", "X").unwrap_err();
        assert!(matches!(err, DagError::SelfLoop { .. }));
        assert_eq!(dag.num_edges(), 0);
    }

    #[test]
    fn cycle_is_rejected_and_names_the_path() {
        // C11. X -> Y -> Z, then Z -> X closes a cycle.
        let mut dag = CausalDag::new();
        dag.add_edge("X", "Y").expect("acyclic");
        dag.add_edge("Y", "Z").expect("acyclic");
        let err = dag.add_edge("Z", "X").unwrap_err();
        match err {
            DagError::WouldCreateCycle { path, .. } => {
                assert_eq!(path, vec!["X", "Y", "Z"]);
            }
            other => panic!("expected WouldCreateCycle, got {other:?}"),
        }
        assert!(dag.is_acyclic());
        assert_eq!(dag.num_edges(), 2);
    }

    #[test]
    fn a_rejected_edge_leaves_the_graph_usable() {
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B").expect("acyclic");
        let _ = dag.add_edge("B", "A");
        // The rejected edge must not have been applied, and the graph must
        // still accept legitimate edges afterwards.
        dag.add_edge("B", "C").expect("acyclic");
        assert_eq!(dag.num_edges(), 2);
        assert!(dag.is_acyclic());
    }

    #[test]
    fn diamond_is_not_a_cycle() {
        // A -> B -> D and A -> C -> D. Two paths, no cycle.
        let mut dag = CausalDag::new();
        dag.add_edges([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
            .expect("acyclic");
        assert_eq!(dag.num_edges(), 4);
        assert!(dag.is_acyclic());
    }

    #[test]
    fn latent_marking_round_trips() {
        let mut dag = smoking_dag();
        dag.mark_latent("tar").expect("known");
        assert!(dag.is_latent("tar"));
        assert_eq!(dag.var_kind("tar"), Some(VarKind::Latent));
        assert_eq!(dag.var_kind("smoking"), Some(VarKind::Observed));
        assert_eq!(dag.var_kind("nonexistent"), None);
        assert_eq!(dag.latent_variables(), vec!["tar"]);
        assert_eq!(dag.observed_variables(), vec!["smoking", "cancer"]);
    }

    #[test]
    fn marking_an_unknown_variable_latent_is_an_error() {
        let mut dag = smoking_dag();
        let err = dag.mark_latent("ghost").unwrap_err();
        assert!(matches!(err, DagError::UnknownVariable { .. }));
    }

    #[test]
    fn descendants_follow_edges_forward_only() {
        let dag = smoking_dag();
        let d = dag.descendants("smoking");
        assert_eq!(d.len(), 2);
        assert!(d.contains("tar") && d.contains("cancer"));
        assert!(dag.descendants("cancer").is_empty());
    }

    #[test]
    fn serde_round_trip_preserves_structure_and_latency() {
        let mut dag = smoking_dag();
        dag.mark_latent("tar").expect("known");
        let json = serde_json::to_string(&dag).expect("serialises");
        let back: CausalDag = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(back.num_variables(), 3);
        assert_eq!(back.num_edges(), 3);
        assert!(back.is_latent("tar"));
        assert!(back.is_acyclic());
    }

    #[test]
    fn deserialising_a_cyclic_graph_fails() {
        // The invariant must hold for values that arrive over the wire, not
        // only for ones built through the builder.
        let json = r#"{"variables":["A","B"],"edges":[["A","B"],["B","A"]],"latent":[]}"#;
        let result: Result<CausalDag, _> = serde_json::from_str(json);
        assert!(result.is_err(), "cyclic graph must not deserialise");
    }
}
