use crate::dag::CausalDag;
use crate::dsep;
use petgraph::algo::has_path_connecting;
use std::collections::HashSet;

/// Backdoor criterion for identifying valid adjustment sets.
///
/// A set Z satisfies the backdoor criterion relative to (treatment, outcome) if:
/// 1. No node in Z is a descendant of treatment
/// 2. Z blocks every path between treatment and outcome that contains an arrow into treatment
pub struct BackdoorCriterion;

impl BackdoorCriterion {
    /// Find a minimal valid adjustment set using the backdoor criterion.
    ///
    /// Returns `Some(set)` if a valid adjustment set exists, `None` otherwise.
    pub fn find(dag: &CausalDag, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let treatment_idx = dag.node_index(treatment)?;
        let _outcome_idx = dag.node_index(outcome)?;
        let graph = dag.inner_graph();

        // Find all descendants of treatment (these cannot be in the adjustment set)
        let descendants: HashSet<_> = dag
            .variables()
            .iter()
            .filter(|v| {
                if let Some(idx) = dag.node_index(v) {
                    idx != treatment_idx && has_path_connecting(graph, treatment_idx, idx, None)
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        // The parents of treatment that are not descendants form the adjustment set
        let parents: HashSet<String> = dag
            .parents(treatment)
            .into_iter()
            .filter(|p| !descendants.contains(*p))
            .map(|p| p.to_string())
            .collect();

        Some(parents)
    }
}

/// Front-door criterion for identifying causal effects through mediators.
///
/// The front-door criterion provides an alternative identification strategy
/// when direct backdoor adjustment is not possible due to unmeasured confounders.
pub struct FrontDoorCriterion;

impl FrontDoorCriterion {
    /// Find a valid front-door adjustment set between treatment and outcome.
    ///
    /// Returns `None` if no valid front-door set exists.
    ///
    /// Front-door criterion requires a set M such that:
    /// 1. M intercepts all directed paths from treatment to outcome
    /// 2. No unblocked back-door path from treatment to M
    /// 3. All back-door paths from M to outcome are blocked by treatment
    pub fn find(dag: &CausalDag, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let treatment_idx = dag.node_index(treatment)?;
        let outcome_idx = dag.node_index(outcome)?;
        let graph = dag.inner_graph();

        // Collect candidate mediators: variables that lie on a directed path
        // from treatment to outcome (excluding treatment and outcome themselves).
        let candidates: Vec<String> = dag
            .variables()
            .iter()
            .filter(|v| {
                let name = v.as_str();
                if name == treatment || name == outcome {
                    return false;
                }
                if let Some(v_idx) = dag.node_index(name) {
                    // v must be on a directed path from treatment to outcome
                    has_path_connecting(graph, treatment_idx, v_idx, None)
                        && has_path_connecting(graph, v_idx, outcome_idx, None)
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        // Try each subset of candidates, starting with individual variables
        // and then larger subsets. For practical purposes, try singletons first,
        // then pairs, up to the full candidate set.
        let n = candidates.len();

        // Generate subsets in order of increasing size (1, 2, ..., n)
        for size in 1..=n {
            for subset in combinations(&candidates, size) {
                let m_set: HashSet<String> = subset.into_iter().collect();

                if Self::is_valid_frontdoor(dag, treatment, outcome, &m_set) {
                    return Some(m_set);
                }
            }
        }

        None
    }

    /// Check whether a candidate set M satisfies the front-door criterion.
    fn is_valid_frontdoor(
        dag: &CausalDag,
        treatment: &str,
        outcome: &str,
        m_set: &HashSet<String>,
    ) -> bool {
        let treatment_idx = match dag.node_index(treatment) {
            Some(idx) => idx,
            None => return false,
        };
        let outcome_idx = match dag.node_index(outcome) {
            Some(idx) => idx,
            None => return false,
        };

        // Condition 1: M intercepts all directed paths from treatment to outcome.
        // Remove M from the graph and check if there is still a directed path.
        // We simulate this by checking: in the graph with M nodes removed,
        // is there a directed path from treatment to outcome?
        if has_directed_path_excluding(dag, treatment_idx, outcome_idx, m_set) {
            return false;
        }

        // Condition 2: No unblocked back-door path from treatment to any M.
        // i.e., treatment and each m in M are d-separated given the empty set
        // when we look only at non-causal (back-door) paths.
        // Practically: there should be no back-door path from treatment to M.
        // The parents of treatment that are not on a causal path should not
        // connect to M without being blocked.
        // Simpler check: treatment d-separates the set of treatment's non-descendants
        // from M. Equivalently, no unblocked path from treatment to M that starts
        // with an arrow INTO treatment.
        // We check: the empty set d-separates treatment from each m in M
        // in the manipulated graph (remove edges into treatment).
        // But since we don't modify the graph, we can check:
        // no back-door from treatment to M means the parents of treatment
        // don't have paths to M that bypass treatment.
        // Use d-separation: T and M are d-separated given empty set in the
        // graph where we only look at non-causal paths.
        // For simplicity: check that there are no common ancestors of T and any m
        // that create a back-door path.
        // Actually, condition 2 says: no unblocked back-door path from T to M.
        // This means: there is no non-causal path from T to any m in M.
        // A back-door path from T to M is a path that starts with an arrow INTO T.
        // Check: parents of T should not have an unblocked path to M.
        let treatment_set = HashSet::from([treatment.to_string()]);
        for m in m_set {
            // Check that there's no back-door path: parents of treatment
            // shouldn't connect to m outside of treatment's causal paths.
            // Use d-separation: check that parents of T are d-separated from m
            // given {T}, which ensures no back-door path exists.
            // If all paths from T to m go through T's children (i.e., are causal),
            // that's fine. We need to check that no path goes T <- ... -> m.
            // We check: parents of T are d-separated from m given {T}.
            let parents = dag.parents(treatment);
            for parent in &parents {
                if dag.node_index(parent).is_some()
                    && !dsep::d_separated(dag, parent, m, &treatment_set)
                {
                    return false;
                }
            }
        }

        // Condition 3: All back-door paths from M to outcome are blocked by treatment.
        // i.e., for each m in M, m and outcome are d-separated given {treatment}
        // on non-causal paths from m.
        // Check: d-separate m and outcome given {treatment} in the graph
        // without edges from m.
        // Approximate: check d-separation of outcome from each m, given {treatment},
        // looking only at back-door paths (paths into m).
        // For simplicity, we check that the parents of each m are d-separated
        // from outcome given {treatment}.
        for m in m_set {
            let parents_of_m = dag.parents(m);
            for parent in &parents_of_m {
                if *parent == treatment {
                    continue; // This is the causal path, skip
                }
                if !dsep::d_separated(dag, parent, outcome, &treatment_set) {
                    return false;
                }
            }
        }

        true
    }
}

/// Check if there is a directed path from `start` to `end` excluding nodes in `exclude`.
fn has_directed_path_excluding(
    dag: &CausalDag,
    start: petgraph::graph::NodeIndex,
    end: petgraph::graph::NodeIndex,
    exclude: &HashSet<String>,
) -> bool {
    use petgraph::Direction;
    use std::collections::VecDeque;

    let graph = dag.inner_graph();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if node == end {
            return true;
        }
        if !visited.insert(node) {
            continue;
        }

        for child in graph.neighbors_directed(node, Direction::Outgoing) {
            let child_name = &graph[child];
            if !exclude.contains(child_name) {
                queue.push_back(child);
            }
        }
    }

    false
}

/// Generate all combinations of `size` elements from `items`.
fn combinations(items: &[String], size: usize) -> Vec<Vec<String>> {
    if size == 0 {
        return vec![vec![]];
    }
    if items.len() < size {
        return vec![];
    }

    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let rest = &items[i + 1..];
        for mut combo in combinations(rest, size - 1) {
            combo.insert(0, item.clone());
            result.push(combo);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backdoor_simple_confounding() {
        // Z -> X -> Y, Z -> Y  (Z is a confounder)
        let mut dag = CausalDag::new();
        dag.add_edge("Z", "X");
        dag.add_edge("X", "Y");
        dag.add_edge("Z", "Y");

        let adjustment = BackdoorCriterion::find(&dag, "X", "Y").unwrap();
        assert!(adjustment.contains("Z"));
    }

    #[test]
    fn backdoor_no_confounders() {
        // X -> Y (no confounders)
        let mut dag = CausalDag::new();
        dag.add_edge("X", "Y");

        let adjustment = BackdoorCriterion::find(&dag, "X", "Y").unwrap();
        assert!(adjustment.is_empty());
    }

    #[test]
    fn backdoor_excludes_descendants() {
        // Z -> X -> M -> Y, Z -> Y  (M is a descendant of X, should not be in set)
        let mut dag = CausalDag::new();
        dag.add_edge("Z", "X");
        dag.add_edge("X", "M");
        dag.add_edge("M", "Y");
        dag.add_edge("Z", "Y");

        let adjustment = BackdoorCriterion::find(&dag, "X", "Y").unwrap();
        assert!(adjustment.contains("Z"));
        assert!(!adjustment.contains("M"));
    }

    #[test]
    fn frontdoor_classic_smoking() {
        // Classic front-door example:
        // U -> Smoking, U -> Cancer (U is unobserved confounder)
        // Smoking -> Tar -> Cancer
        // Tar is the front-door adjustment set.
        // Note: we represent U explicitly in the DAG.
        let mut dag = CausalDag::new();
        dag.add_edge("U", "Smoking");
        dag.add_edge("U", "Cancer");
        dag.add_edge("Smoking", "Tar");
        dag.add_edge("Tar", "Cancer");

        let result = FrontDoorCriterion::find(&dag, "Smoking", "Cancer");
        assert!(result.is_some(), "Should find a front-door set");
        let fd_set = result.unwrap();
        assert!(
            fd_set.contains("Tar"),
            "Front-door set should contain Tar, got {:?}",
            fd_set
        );
    }

    #[test]
    fn frontdoor_no_mediator() {
        // X -> Y with confounder U -> X, U -> Y but no mediator.
        // No front-door set should exist.
        let mut dag = CausalDag::new();
        dag.add_edge("U", "X");
        dag.add_edge("U", "Y");
        dag.add_edge("X", "Y");

        let result = FrontDoorCriterion::find(&dag, "X", "Y");
        assert!(
            result.is_none(),
            "Should not find front-door set without mediator"
        );
    }

    #[test]
    fn frontdoor_direct_path_invalidates() {
        // Smoking -> Tar -> Cancer, Smoking -> Cancer (direct path)
        // U -> Smoking, U -> Cancer
        // Tar alone cannot be a front-door set because Smoking -> Cancer
        // is a direct path not intercepted by Tar.
        let mut dag = CausalDag::new();
        dag.add_edge("U", "Smoking");
        dag.add_edge("U", "Cancer");
        dag.add_edge("Smoking", "Tar");
        dag.add_edge("Tar", "Cancer");
        dag.add_edge("Smoking", "Cancer"); // direct path

        let result = FrontDoorCriterion::find(&dag, "Smoking", "Cancer");
        // Tar alone won't work because of the direct Smoking -> Cancer path.
        // No valid front-door set exists.
        assert!(
            result.is_none(),
            "Direct path should invalidate front-door with Tar only"
        );
    }
}
