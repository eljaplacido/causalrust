//! D-separation testing for causal DAGs.
//!
//! Uses the Bayes-Ball algorithm to determine whether two variables
//! are d-separated given a conditioning set.

use crate::dag::CausalDag;
use petgraph::graph::NodeIndex;
use petgraph::Direction;
use std::collections::{HashSet, VecDeque};

/// Test whether `x` and `y` are d-separated given conditioning set `z`.
///
/// Uses the Bayes-Ball algorithm on the underlying petgraph.
/// A path through a node W is blocked if:
/// - W is in Z and the path is a chain (->W->) or fork (<-W->)
/// - W is NOT in Z (and no descendant of W is in Z) and the path is a collider (->W<-)
///
/// If Y is not reachable via any active path from X, then X and Y are d-separated given Z.
pub fn d_separated(dag: &CausalDag, x: &str, y: &str, z: &HashSet<String>) -> bool {
    let Some(x_idx) = dag.node_index(x) else {
        return true;
    };
    let Some(y_idx) = dag.node_index(y) else {
        return true;
    };

    let graph = dag.inner_graph();

    // Precompute: which nodes are in Z or have a descendant in Z.
    let z_indices: HashSet<NodeIndex> = z
        .iter()
        .filter_map(|name| dag.node_index(name))
        .collect();

    // Find all ancestors of Z nodes (nodes that have a descendant in Z).
    let ancestors_of_z = ancestors_of_set(dag, &z_indices);

    // Bayes-Ball: BFS traversal tracking direction of arrival.
    // State: (node, came_from_parent) where:
    //   true  = the ball arrived at this node traveling DOWN from one of its parents
    //   false = the ball arrived at this node traveling UP from one of its children
    let mut visited: HashSet<(NodeIndex, bool)> = HashSet::new();
    let mut queue: VecDeque<(NodeIndex, bool)> = VecDeque::new();

    // Start from X. We try both directions to find any active path.
    queue.push_back((x_idx, true)); // as if arrived from a parent (going down)
    queue.push_back((x_idx, false)); // as if arrived from a child (going up)

    while let Some((node, came_from_parent)) = queue.pop_front() {
        if !visited.insert((node, came_from_parent)) {
            continue;
        }

        // Check if we reached Y
        if node == y_idx {
            return false; // Not d-separated
        }

        let in_z = z_indices.contains(&node);

        if came_from_parent {
            // Ball arrived going DOWN (from a parent).
            if !in_z {
                // Not conditioned on: continue DOWN to children (chain: ->W->).
                // Arriving from a parent at a non-conditioned node, the ball
                // passes through to children only (not back to parents).
                for child in graph.neighbors_directed(node, Direction::Outgoing) {
                    queue.push_back((child, true)); // child receives from parent
                }
            }
            // If in Z or ancestor of Z: this could be a collider that is activated.
            // But a collider means two parents pointing into it, and we arrived from a parent.
            // If conditioned on (in Z), the path is BLOCKED for chains and forks
            // (the ball cannot pass through a conditioned non-collider).
            // If this is a collider and it (or descendant) is in Z, path is activated.
            if in_z || ancestors_of_z.contains(&node) {
                // Activated collider: go UP to other parents.
                for parent in graph.neighbors_directed(node, Direction::Incoming) {
                    queue.push_back((parent, false)); // parent receives from child
                }
            }
        } else {
            // Ball arrived going UP (from a child).
            if !in_z {
                // Not conditioned on: for a fork (<-W->), pass to parents and children.
                // Going UP to parents:
                for parent in graph.neighbors_directed(node, Direction::Incoming) {
                    queue.push_back((parent, false)); // parent receives from child
                }
                // Going DOWN to children:
                for child in graph.neighbors_directed(node, Direction::Outgoing) {
                    queue.push_back((child, true)); // child receives from parent
                }
            }
            // If in Z: blocks fork and chain, but for colliders we already handle above.
            // When ball arrives from child at a conditioned node, it's blocked
            // (for forks and chains). But this also enables collider paths:
            // we've already handled collider activation in the came_from_parent branch.
        }
    }

    true // Y not reachable => d-separated
}

/// Compute all ancestors of a set of nodes (including the nodes themselves).
fn ancestors_of_set(dag: &CausalDag, nodes: &HashSet<NodeIndex>) -> HashSet<NodeIndex> {
    let graph = dag.inner_graph();
    let mut ancestors = HashSet::new();
    let mut queue: VecDeque<NodeIndex> = nodes.iter().copied().collect();

    while let Some(node) = queue.pop_front() {
        if !ancestors.insert(node) {
            continue;
        }
        for parent in graph.neighbors_directed(node, Direction::Incoming) {
            queue.push_back(parent);
        }
    }

    ancestors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::CausalDag;

    #[test]
    fn chain_blocked_by_middle() {
        // X -> W -> Y
        // Conditioning on W blocks the chain.
        let mut dag = CausalDag::new();
        dag.add_edge("X", "W");
        dag.add_edge("W", "Y");

        let z = HashSet::from(["W".to_string()]);
        assert!(d_separated(&dag, "X", "Y", &z));

        // Without conditioning, not d-separated.
        let empty: HashSet<String> = HashSet::new();
        assert!(!d_separated(&dag, "X", "Y", &empty));
    }

    #[test]
    fn fork_blocked_by_common_cause() {
        // X <- W -> Y
        // Conditioning on W blocks the fork.
        let mut dag = CausalDag::new();
        dag.add_edge("W", "X");
        dag.add_edge("W", "Y");

        let z = HashSet::from(["W".to_string()]);
        assert!(d_separated(&dag, "X", "Y", &z));

        // Without conditioning, not d-separated.
        let empty: HashSet<String> = HashSet::new();
        assert!(!d_separated(&dag, "X", "Y", &empty));
    }

    #[test]
    fn collider_blocked_unless_conditioned() {
        // X -> W <- Y
        // Collider: X and Y are d-separated without conditioning.
        // Conditioning on W opens the path.
        let mut dag = CausalDag::new();
        dag.add_edge("X", "W");
        dag.add_edge("Y", "W");

        let empty: HashSet<String> = HashSet::new();
        assert!(d_separated(&dag, "X", "Y", &empty));

        let z = HashSet::from(["W".to_string()]);
        assert!(!d_separated(&dag, "X", "Y", &z));
    }

    #[test]
    fn complex_case() {
        // A -> B -> C -> D
        // A -> E -> D
        // B -> E (collider at E for paths through B and from external)
        //
        // DAG: A->B, B->C, C->D, A->E, B->E, E->D
        let mut dag = CausalDag::new();
        dag.add_edge("A", "B");
        dag.add_edge("B", "C");
        dag.add_edge("C", "D");
        dag.add_edge("A", "E");
        dag.add_edge("B", "E");
        dag.add_edge("E", "D");

        // Without conditioning: A->B->C->D is active, so A and D are not d-separated.
        let empty: HashSet<String> = HashSet::new();
        assert!(!d_separated(&dag, "A", "D", &empty));

        // Conditioning on {B, E}: blocks A->B chain, blocks A->E chain.
        // Path A->B->C->D: blocked at B (chain, B in Z).
        // Path A->E->D: blocked at E (chain, E in Z).
        // But conditioning on E opens collider B->E<-A... wait, E has parents A and B,
        // so E is a collider on path A->E<-B. Conditioning on E opens that.
        // Through opened collider: A->(E conditioned)<-B->C->D
        // But B is also in Z, so chain B->C blocked.
        // So A and D should be d-separated given {B, E}.
        let z = HashSet::from(["B".to_string(), "E".to_string()]);
        assert!(d_separated(&dag, "A", "D", &z));

        // Conditioning on {C}: blocks chain B->C->D.
        // But A->E->D is still active. So A and D are not d-separated.
        let z2 = HashSet::from(["C".to_string()]);
        assert!(!d_separated(&dag, "A", "D", &z2));
    }
}
