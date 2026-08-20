//! D-separation testing for causal DAGs.
//!
//! Uses the Bayes-Ball algorithm to decide whether two variables are
//! conditionally independent given a conditioning set.
//!
//! # Unknown variables are an error, not an answer
//!
//! This function previously returned `true` for any variable name the graph did
//! not contain (finding C12). `true` means "d-separated", which means
//! "conditionally independent" — so a misspelled variable produced a *positive*
//! finding of independence, which is the answer most likely to be acted on. A
//! typo in an adjustment set silently validated it.
//!
//! Every name in the query, including every member of the conditioning set, is
//! now checked against the graph, and an unknown one returns
//! [`DsepError::UnknownVariable`] listing what the graph does contain.

use crate::dag::CausalDag;
use crate::error::DsepError;
use petgraph::Direction;
use petgraph::graph::NodeIndex;
use std::collections::{HashSet, VecDeque};

/// Test whether `x` and `y` are d-separated given conditioning set `z`.
///
/// A path through node W is blocked when W is in Z and the path is a chain
/// (`->W->`) or a fork (`<-W->`), or when W is a collider (`->W<-`) and neither
/// W nor any descendant of W is in Z. `x` and `y` are d-separated when no
/// active path connects them.
///
/// # Errors
///
/// - [`DsepError::UnknownVariable`] if `x`, `y`, or any member of `z` is not in
///   the graph.
/// - [`DsepError::OverlappingSets`] if the three sets are not disjoint. The
///   relation is only defined for disjoint sets, and answering anyway returned
///   `true` — a positive finding of independence for a query with no answer.
pub fn d_separated(
    dag: &CausalDag,
    x: &str,
    y: &str,
    z: &HashSet<String>,
) -> Result<bool, DsepError> {
    let unknown = |name: &str| DsepError::UnknownVariable {
        name: name.to_string(),
        known: dag.variables().to_vec(),
    };

    let x_idx = dag.node_index(x).ok_or_else(|| unknown(x))?;
    let y_idx = dag.node_index(y).ok_or_else(|| unknown(y))?;

    // D-separation is defined for three DISJOINT sets. Answering a malformed
    // query returned `true` — "conditionally independent" — which is the same
    // failure shape as C12: affirmatively wrong in the direction a caller acts
    // on. `networkx.is_d_separator` raises here too, and matching that is how
    // the disagreement was found.
    if x == y {
        return Err(DsepError::OverlappingSets {
            name: x.to_string(),
            roles: "x and y".to_string(),
        });
    }
    for (name, role) in [(x, "x and z"), (y, "y and z")] {
        if z.contains(name) {
            return Err(DsepError::OverlappingSets {
                name: name.to_string(),
                roles: role.to_string(),
            });
        }
    }
    // The conditioning set is checked too: adjusting for a variable that does
    // not exist is exactly the mistake this is meant to catch, and it is the
    // easiest one to make when the set is assembled programmatically.
    for name in z {
        if !dag.contains(name) {
            return Err(unknown(name));
        }
    }

    let graph = dag.inner_graph();

    // Precompute: which nodes are in Z or have a descendant in Z.
    let z_indices: HashSet<NodeIndex> = z.iter().filter_map(|name| dag.node_index(name)).collect();

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

        // Reached Y along an active path.
        if node == y_idx {
            return Ok(false);
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

    // Y unreachable along every active path.
    Ok(true)
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

    fn dag_from(edges: &[(&str, &str)]) -> CausalDag {
        let mut dag = CausalDag::new();
        for &(a, b) in edges {
            dag.add_edge(a, b).expect("acyclic fixture");
        }
        dag
    }

    fn sep(dag: &CausalDag, x: &str, y: &str, z: &[&str]) -> bool {
        let set: HashSet<String> = z.iter().map(|s| (*s).to_string()).collect();
        d_separated(dag, x, y, &set).expect("known variables")
    }

    #[test]
    fn chain_blocked_by_middle() {
        // X -> W -> Y. Conditioning on W blocks the chain.
        let dag = dag_from(&[("X", "W"), ("W", "Y")]);
        assert!(sep(&dag, "X", "Y", &["W"]));
        assert!(!sep(&dag, "X", "Y", &[]));
    }

    #[test]
    fn fork_blocked_by_common_cause() {
        // X <- W -> Y. Conditioning on the common cause blocks the fork.
        let dag = dag_from(&[("W", "X"), ("W", "Y")]);
        assert!(sep(&dag, "X", "Y", &["W"]));
        assert!(!sep(&dag, "X", "Y", &[]));
    }

    #[test]
    fn collider_opens_when_conditioned() {
        // X -> W <- Y. The collider blocks by default and opens on
        // conditioning — the direction that surprises people.
        let dag = dag_from(&[("X", "W"), ("Y", "W")]);
        assert!(sep(&dag, "X", "Y", &[]));
        assert!(!sep(&dag, "X", "Y", &["W"]));
    }

    #[test]
    fn collider_opens_when_a_descendant_is_conditioned() {
        // X -> W <- Y, W -> D. Conditioning on D also opens the collider.
        let dag = dag_from(&[("X", "W"), ("Y", "W"), ("W", "D")]);
        assert!(sep(&dag, "X", "Y", &[]));
        assert!(!sep(&dag, "X", "Y", &["D"]));
    }

    #[test]
    fn d_separation_is_symmetric() {
        let dag = dag_from(&[("A", "B"), ("B", "C"), ("A", "E"), ("E", "D"), ("C", "D")]);
        for (x, y) in [("A", "D"), ("B", "E"), ("C", "E")] {
            for z in [vec![], vec!["B"], vec!["E"], vec!["B", "E"]] {
                // The three sets must be disjoint for the query to be defined;
                // `d_separated` now rejects the rest rather than answering
                // `true`, so those combinations are skipped here rather than
                // asserted on.
                if z.contains(&x) || z.contains(&y) {
                    continue;
                }
                assert_eq!(
                    sep(&dag, x, y, &z),
                    sep(&dag, y, x, &z),
                    "asymmetric for {x}/{y} given {z:?}"
                );
            }
        }
    }

    #[test]
    fn unknown_x_is_an_error_not_independence() {
        // C12. This returned `true` — a positive finding of independence for a
        // variable the graph had never heard of.
        let dag = dag_from(&[("X", "Y")]);
        let empty = HashSet::new();
        let err = d_separated(&dag, "typo", "Y", &empty).unwrap_err();
        match err {
            DsepError::UnknownVariable { name, known } => {
                assert_eq!(name, "typo");
                assert!(known.contains(&"X".to_string()));
            }
            other => panic!("expected UnknownVariable, got {other}"),
        }
    }

    #[test]
    fn unknown_y_is_an_error() {
        let dag = dag_from(&[("X", "Y")]);
        let empty = HashSet::new();
        assert!(d_separated(&dag, "X", "typo", &empty).is_err());
    }

    #[test]
    fn unknown_conditioning_variable_is_an_error() {
        // The likeliest typo in practice: the adjustment set is assembled
        // programmatically and one name does not match.
        let dag = dag_from(&[("X", "W"), ("W", "Y")]);
        let z = HashSet::from(["Wt".to_string()]);
        let err = d_separated(&dag, "X", "Y", &z).unwrap_err();
        assert!(matches!(
            err,
            DsepError::UnknownVariable { ref name, .. } if name == "Wt"
        ));
    }

    #[test]
    fn disconnected_variables_are_d_separated() {
        let mut dag = dag_from(&[("A", "B")]);
        dag.add_variable("Z");
        let empty = HashSet::new();
        assert!(d_separated(&dag, "A", "Z", &empty).expect("known"));
    }
}
