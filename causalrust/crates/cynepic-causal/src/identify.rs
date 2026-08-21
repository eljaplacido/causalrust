//! Identification: deciding whether a causal effect can be computed from
//! observational data, and if so, what to adjust for.
//!
//! # A criterion that can say no
//!
//! `BackdoorCriterion::find` previously returned `Option<HashSet<String>>` and
//! in practice always `Some` — it handed back the parents of the treatment
//! without checking either that they were measurable or that they actually
//! blocked anything (finding C2). Two consequences:
//!
//! 1. On the crate's own front-door test DAG, where `U` is documented as
//!    unobserved, it returned `{U}`. Adjusting for `U` is impossible by
//!    construction; the answer was not merely wrong, it was unusable.
//! 2. Identification could not fail. `None` already meant "no adjustment
//!    needed" — a successful identification returning the empty set — so it
//!    could not also mean "not identifiable". Those are opposite conclusions
//!    and they shared one representation.
//!
//! Both are now [`Result`]. Success carries an [`AdjustmentSet`] whose members
//! are guaranteed observed; failure carries [`IdentificationError`] naming
//! either the latent variables that would have been required or the backdoor
//! paths that no observed set can block.
//!
//! # How validity is checked
//!
//! A set `Z` satisfies the backdoor criterion for `(T, Y)` when no member of
//! `Z` is a descendant of `T`, and `Z` d-separates `T` from `Y` in the graph
//! with `T`'s outgoing edges deleted. The second condition is checked directly
//! against [`crate::dsep::d_separated`] on that modified graph rather than
//! approximated, so a returned set has been *verified* to block every backdoor
//! path, not merely constructed by a rule of thumb.

use std::collections::{HashSet, VecDeque};

use petgraph::Direction;
use petgraph::algo::has_path_connecting;

use crate::dag::CausalDag;
use crate::dsep;
use crate::error::IdentificationError;

/// Largest adjustment set the search will consider.
///
/// The subset search is exponential, so it is bounded. Real adjustment sets are
/// small; a graph needing more than four variables to close its backdoors is
/// one where the structure, not the search, is the problem. Exceeding the bound
/// reports non-identifiability rather than silently returning nothing, and the
/// bound is named in this constant so the limitation is greppable.
pub const MAX_ADJUSTMENT_SET_SIZE: usize = 4;

/// Longest path considered when reporting which backdoor paths are unblocked.
///
/// Only affects the *explanation* attached to a failure, never the verdict —
/// that comes from d-separation, which is exact.
pub const MAX_REPORTED_PATH_LEN: usize = 12;

/// A verified adjustment set.
///
/// Every member is observed and the set has been checked to block all backdoor
/// paths. The empty set is a legitimate, successful result: it means no
/// adjustment is needed, which is a different statement from "no set works".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjustmentSet {
    variables: HashSet<String>,
}

impl AdjustmentSet {
    /// The variables to adjust for.
    pub fn variables(&self) -> &HashSet<String> {
        &self.variables
    }

    /// Whether no adjustment is required.
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// How many variables to adjust for.
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Whether a variable is in the set.
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains(name)
    }

    /// Members, sorted, for stable output.
    pub fn sorted(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.variables.iter().map(String::as_str).collect();
        v.sort_unstable();
        v
    }
}

/// Backdoor identification.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackdoorCriterion;

impl BackdoorCriterion {
    /// Find a minimal observed adjustment set satisfying the backdoor criterion.
    ///
    /// Searches by increasing size, so the returned set is minimal among those
    /// considered — smaller sets mean fewer variables to measure and less
    /// variance in the resulting estimate.
    ///
    /// # Errors
    ///
    /// - [`IdentificationError::UnknownVariable`] if `treatment` or `outcome`
    ///   is not in the graph.
    /// - [`IdentificationError::RequiresLatent`] if a valid set exists but only
    ///   by conditioning on variables marked [`crate::dag::VarKind::Latent`].
    ///   Reported separately from plain non-identifiability because the remedy
    ///   is different: measure the variable, rather than change strategy.
    /// - [`IdentificationError::NotIdentifiable`] if no observed set up to
    ///   [`MAX_ADJUSTMENT_SET_SIZE`] blocks every backdoor path, listing the
    ///   paths that remain open.
    pub fn find(
        dag: &CausalDag,
        treatment: &str,
        outcome: &str,
    ) -> Result<AdjustmentSet, IdentificationError> {
        let unknown = |name: &str| IdentificationError::UnknownVariable {
            name: name.to_string(),
            known: dag.variables().to_vec(),
        };
        if !dag.contains(treatment) {
            return Err(unknown(treatment));
        }
        if !dag.contains(outcome) {
            return Err(unknown(outcome));
        }

        let backdoor = backdoor_graph(dag, treatment);
        let descendants = dag.descendants(treatment);

        // Candidates: observed, not the treatment or outcome, and not a
        // descendant of treatment (conditioning on one of those opens paths
        // rather than closing them).
        let mut candidates: Vec<String> = dag
            .observed_variables()
            .into_iter()
            .filter(|v| *v != treatment && *v != outcome && !descendants.contains(*v))
            .map(str::to_string)
            .collect();
        candidates.sort_unstable();

        // Empty set first: if the effect is already identified, adjusting for
        // anything is unnecessary variance.
        if is_valid_backdoor(&backdoor, treatment, outcome, &HashSet::new())? {
            return Ok(AdjustmentSet {
                variables: HashSet::new(),
            });
        }

        let max_size = candidates.len().min(MAX_ADJUSTMENT_SET_SIZE);
        for size in 1..=max_size {
            for combo in combinations(&candidates, size) {
                let z: HashSet<String> = combo.into_iter().collect();
                if is_valid_backdoor(&backdoor, treatment, outcome, &z)? {
                    return Ok(AdjustmentSet { variables: z });
                }
            }
        }

        // Nothing observed worked. Distinguish "you would need to measure
        // something you cannot" from "no adjustment strategy exists at all",
        // because only the first has an obvious remedy.
        let latent_needed = latent_variables_on_backdoor_paths(dag, treatment, outcome);
        if !latent_needed.is_empty() {
            let mut latent: Vec<String> = latent_needed.into_iter().collect();
            latent.sort_unstable();
            return Err(IdentificationError::RequiresLatent {
                treatment: treatment.to_string(),
                outcome: outcome.to_string(),
                latent,
            });
        }

        Err(IdentificationError::NotIdentifiable {
            treatment: treatment.to_string(),
            outcome: outcome.to_string(),
            unblockable: backdoor_paths(dag, treatment, outcome),
        })
    }

    /// Check a caller-supplied adjustment set.
    ///
    /// For validating a set chosen on domain grounds rather than by search —
    /// which is the normal case in practice, since the search cannot know which
    /// variables are actually available in the data.
    ///
    /// # Errors
    ///
    /// [`IdentificationError::UnknownVariable`] if any name is not in the graph.
    /// [`IdentificationError::RequiresLatent`] if the set contains a latent
    /// variable.
    pub fn validate(
        dag: &CausalDag,
        treatment: &str,
        outcome: &str,
        candidate: &HashSet<String>,
    ) -> Result<bool, IdentificationError> {
        let unknown = |name: &str| IdentificationError::UnknownVariable {
            name: name.to_string(),
            known: dag.variables().to_vec(),
        };
        for name in candidate
            .iter()
            .map(String::as_str)
            .chain([treatment, outcome])
        {
            if !dag.contains(name) {
                return Err(unknown(name));
            }
        }

        let latent: Vec<String> = candidate
            .iter()
            .filter(|v| dag.is_latent(v))
            .cloned()
            .collect();
        if !latent.is_empty() {
            let mut latent = latent;
            latent.sort_unstable();
            return Err(IdentificationError::RequiresLatent {
                treatment: treatment.to_string(),
                outcome: outcome.to_string(),
                latent,
            });
        }

        // An adjustment set containing the treatment or the outcome is not a
        // valid adjustment set, and `false` says so truthfully — this is the
        // caller asking "may I use this?", which has a defensible `no`.
        //
        // The lower-level `d_separated` now *rejects* the same shape rather
        // than answering, because there the question is "are these
        // independent?", and for overlapping sets that has no answer at all
        // (finding C12). Screening here keeps that stricter rule from leaking
        // out as an error on a call that has a sensible answer.
        if candidate.contains(treatment) || candidate.contains(outcome) {
            return Ok(false);
        }

        // Conditioning on a descendant of the treatment is never valid.
        let descendants = dag.descendants(treatment);
        if candidate.iter().any(|v| descendants.contains(v)) {
            return Ok(false);
        }

        let backdoor = backdoor_graph(dag, treatment);
        is_valid_backdoor(&backdoor, treatment, outcome, candidate)
    }
}

/// Build the graph with the treatment's outgoing edges deleted.
///
/// Backdoor paths are exactly the `T`-`Y` paths that survive this deletion, so
/// "Z blocks every backdoor path" becomes an ordinary d-separation query.
/// Deleting edges cannot create a cycle, so every `add_edge` here succeeds.
fn backdoor_graph(dag: &CausalDag, treatment: &str) -> CausalDag {
    let mut out = CausalDag::new();
    for v in dag.variables() {
        out.add_variable(v);
    }
    for (cause, effect) in dag.edges() {
        if cause == treatment {
            continue;
        }
        // Cannot fail: a subgraph of an acyclic graph is acyclic.
        let _ = out.add_edge(cause, effect);
    }
    out
}

/// Whether `z` d-separates treatment from outcome in the backdoor graph.
fn is_valid_backdoor(
    backdoor: &CausalDag,
    treatment: &str,
    outcome: &str,
    z: &HashSet<String>,
) -> Result<bool, IdentificationError> {
    dsep::d_separated(backdoor, treatment, outcome, z).map_err(|e| match e {
        crate::error::DsepError::UnknownVariable { name, known } => {
            IdentificationError::UnknownVariable { name, known }
        }
        // Unreachable from `find` (candidates exclude the treatment and the
        // outcome by construction) and from `validate` (which screens for it
        // above and answers `Ok(false)`). Mapped rather than unwrapped anyway:
        // an `expect` here would be a panic in library code on a path a future
        // caller could reach.
        crate::error::DsepError::OverlappingSets { name, roles } => {
            IdentificationError::OverlappingSets { name, roles }
        }
    })
}

/// Latent variables lying on some backdoor path between treatment and outcome.
///
/// These are the variables that would have to be measured for adjustment to
/// become possible.
fn latent_variables_on_backdoor_paths(
    dag: &CausalDag,
    treatment: &str,
    outcome: &str,
) -> HashSet<String> {
    let mut out = HashSet::new();
    for path in backdoor_paths(dag, treatment, outcome) {
        for node in path {
            if dag.is_latent(&node) {
                out.insert(node);
            }
        }
    }
    out
}

/// Enumerate backdoor paths: undirected `T`-`Y` paths whose first edge points
/// into `T`.
///
/// Used only to explain a failure. Bounded by [`MAX_REPORTED_PATH_LEN`], so on
/// a dense graph the list may be partial — it is an explanation, not the
/// verdict.
fn backdoor_paths(dag: &CausalDag, treatment: &str, outcome: &str) -> Vec<Vec<String>> {
    let graph = dag.inner_graph();
    let (Some(t_idx), Some(y_idx)) = (dag.node_index(treatment), dag.node_index(outcome)) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    // Each frontier entry is a partial path. Seeded with T's parents only, so
    // every path enumerated begins with an arrow into T.
    let mut stack: Vec<Vec<petgraph::graph::NodeIndex>> = graph
        .neighbors_directed(t_idx, Direction::Incoming)
        .map(|p| vec![t_idx, p])
        .collect();

    while let Some(path) = stack.pop() {
        let tail = *path.last().expect("paths are never empty");
        if tail == y_idx {
            found.push(path.iter().map(|&n| graph[n].clone()).collect());
            continue;
        }
        if path.len() >= MAX_REPORTED_PATH_LEN {
            continue;
        }
        for next in graph
            .neighbors_directed(tail, Direction::Incoming)
            .chain(graph.neighbors_directed(tail, Direction::Outgoing))
        {
            if path.contains(&next) {
                continue;
            }
            let mut extended = path.clone();
            extended.push(next);
            stack.push(extended);
        }
    }

    found.sort();
    found.dedup();
    found
}

/// Front-door identification through mediators.
///
/// Applies when backdoor adjustment is impossible because of unmeasured
/// confounding, but a fully mediating, unconfounded set exists.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrontDoorCriterion;

impl FrontDoorCriterion {
    /// Find a valid front-door mediator set.
    ///
    /// Requires a set `M` that intercepts every directed path from treatment to
    /// outcome, has no unblocked backdoor path from the treatment, and whose
    /// backdoor paths to the outcome are all blocked by the treatment.
    ///
    /// Mediators must be observed — the whole point of the front door is to
    /// route around something that is not.
    ///
    /// # Errors
    ///
    /// [`IdentificationError::UnknownVariable`] for unknown names;
    /// [`IdentificationError::NotIdentifiable`] if no valid mediator set exists.
    pub fn find(
        dag: &CausalDag,
        treatment: &str,
        outcome: &str,
    ) -> Result<AdjustmentSet, IdentificationError> {
        let unknown = |name: &str| IdentificationError::UnknownVariable {
            name: name.to_string(),
            known: dag.variables().to_vec(),
        };
        let treatment_idx = dag
            .node_index(treatment)
            .ok_or_else(|| unknown(treatment))?;
        let outcome_idx = dag.node_index(outcome).ok_or_else(|| unknown(outcome))?;
        let graph = dag.inner_graph();

        let mut candidates: Vec<String> = dag
            .observed_variables()
            .into_iter()
            .filter(|name| {
                if *name == treatment || *name == outcome {
                    return false;
                }
                dag.node_index(name).is_some_and(|idx| {
                    has_path_connecting(graph, treatment_idx, idx, None)
                        && has_path_connecting(graph, idx, outcome_idx, None)
                })
            })
            .map(str::to_string)
            .collect();
        candidates.sort_unstable();

        for size in 1..=candidates.len() {
            for subset in combinations(&candidates, size) {
                let m: HashSet<String> = subset.into_iter().collect();
                if Self::is_valid_frontdoor(dag, treatment, outcome, &m)? {
                    return Ok(AdjustmentSet { variables: m });
                }
            }
        }

        Err(IdentificationError::NotIdentifiable {
            treatment: treatment.to_string(),
            outcome: outcome.to_string(),
            unblockable: backdoor_paths(dag, treatment, outcome),
        })
    }

    /// Check the three front-door conditions for a candidate mediator set.
    fn is_valid_frontdoor(
        dag: &CausalDag,
        treatment: &str,
        outcome: &str,
        m_set: &HashSet<String>,
    ) -> Result<bool, IdentificationError> {
        let (Some(t_idx), Some(y_idx)) = (dag.node_index(treatment), dag.node_index(outcome))
        else {
            return Ok(false);
        };

        // 1. M intercepts every directed path from T to Y: with M removed,
        //    no directed path survives.
        if has_directed_path_excluding(dag, t_idx, y_idx, m_set) {
            return Ok(false);
        }

        let treatment_set = HashSet::from([treatment.to_string()]);

        // 2. No unblocked backdoor path from T to any mediator.
        for m in m_set {
            for parent in dag.parents(treatment) {
                if !is_valid_backdoor_named(dag, parent, m, &treatment_set)? {
                    return Ok(false);
                }
            }
        }

        // 3. Every backdoor path from a mediator to Y is blocked by T.
        for m in m_set {
            for parent in dag.parents(m) {
                if parent == treatment {
                    continue; // the causal path, not a backdoor
                }
                if !is_valid_backdoor_named(dag, parent, outcome, &treatment_set)? {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

/// d-separation with the error mapped into identification's error type.
fn is_valid_backdoor_named(
    dag: &CausalDag,
    a: &str,
    b: &str,
    z: &HashSet<String>,
) -> Result<bool, IdentificationError> {
    dsep::d_separated(dag, a, b, z).map_err(|e| match e {
        crate::error::DsepError::UnknownVariable { name, known } => {
            IdentificationError::UnknownVariable { name, known }
        }
        crate::error::DsepError::OverlappingSets { name, roles } => {
            IdentificationError::OverlappingSets { name, roles }
        }
    })
}

/// Whether a directed path from `start` to `end` exists avoiding `exclude`.
fn has_directed_path_excluding(
    dag: &CausalDag,
    start: petgraph::graph::NodeIndex,
    end: petgraph::graph::NodeIndex,
    exclude: &HashSet<String>,
) -> bool {
    let graph = dag.inner_graph();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::from([start]);

    while let Some(node) = queue.pop_front() {
        if node == end {
            return true;
        }
        if !visited.insert(node) {
            continue;
        }
        for child in graph.neighbors_directed(node, Direction::Outgoing) {
            if !exclude.contains(&graph[child]) {
                queue.push_back(child);
            }
        }
    }
    false
}

/// All `size`-element combinations of `items`.
fn combinations(items: &[String], size: usize) -> Vec<Vec<String>> {
    if size == 0 {
        return vec![Vec::new()];
    }
    if items.len() < size {
        return Vec::new();
    }
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
        for mut combo in combinations(&items[i + 1..], size - 1) {
            combo.insert(0, item.clone());
            result.push(combo);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dag_from(edges: &[(&str, &str)]) -> CausalDag {
        let mut dag = CausalDag::new();
        for &(a, b) in edges {
            dag.add_edge(a, b).expect("acyclic fixture");
        }
        dag
    }

    #[test]
    fn confounder_is_found_and_verified() {
        // W confounds T and Y; adjusting for W identifies the effect.
        let dag = dag_from(&[("W", "T"), ("W", "Y"), ("T", "Y")]);
        let z = BackdoorCriterion::find(&dag, "T", "Y").expect("identifiable");
        assert_eq!(z.sorted(), vec!["W"]);
        assert!(BackdoorCriterion::validate(&dag, "T", "Y", z.variables()).expect("known"));
    }

    #[test]
    fn no_confounding_needs_no_adjustment() {
        // The empty set is a successful identification, not a failure.
        let dag = dag_from(&[("T", "Y")]);
        let z = BackdoorCriterion::find(&dag, "T", "Y").expect("identifiable");
        assert!(z.is_empty());
    }

    #[test]
    fn latent_confounder_makes_the_effect_unidentifiable() {
        // C2. This previously returned Some({U}) — an adjustment set nobody can
        // condition on, because U is unobserved by construction.
        let mut dag = dag_from(&[("U", "T"), ("U", "Y"), ("T", "Y")]);
        dag.mark_latent("U").expect("known");

        let err = BackdoorCriterion::find(&dag, "T", "Y").unwrap_err();
        match err {
            IdentificationError::RequiresLatent { latent, .. } => {
                assert_eq!(latent, vec!["U".to_string()]);
            }
            other => panic!("expected RequiresLatent, got {other:?}"),
        }
    }

    #[test]
    fn front_door_dag_does_not_yield_a_latent_backdoor_set() {
        // C2's exact case: the crate's own front-door fixture, where U is
        // documented as unobserved. Backdoor identification must decline.
        let mut dag = dag_from(&[
            ("U", "Smoking"),
            ("U", "Cancer"),
            ("Smoking", "Tar"),
            ("Tar", "Cancer"),
        ]);
        dag.mark_latent("U").expect("known");

        match BackdoorCriterion::find(&dag, "Smoking", "Cancer") {
            Ok(set) => assert!(
                !set.contains("U"),
                "returned the unobservable confounder: {:?}",
                set.sorted()
            ),
            Err(IdentificationError::RequiresLatent { .. }) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn identification_can_fail() {
        // A criterion that never says no is not a criterion.
        let mut dag = dag_from(&[("U1", "T"), ("U1", "Y"), ("U2", "T"), ("U2", "Y")]);
        dag.mark_latent("U1").expect("known");
        dag.mark_latent("U2").expect("known");
        assert!(BackdoorCriterion::find(&dag, "T", "Y").is_err());
    }

    #[test]
    fn descendants_of_treatment_are_never_adjusted_for() {
        // Conditioning on a mediator blocks the causal path being measured.
        let dag = dag_from(&[("W", "T"), ("W", "Y"), ("T", "M"), ("M", "Y")]);
        let z = BackdoorCriterion::find(&dag, "T", "Y").expect("identifiable");
        assert!(
            !z.contains("M"),
            "adjusted for a descendant: {:?}",
            z.sorted()
        );
        assert!(z.contains("W"));
    }

    #[test]
    fn validate_rejects_a_descendant_adjustment_set() {
        let dag = dag_from(&[("W", "T"), ("W", "Y"), ("T", "M"), ("M", "Y")]);
        let bad = HashSet::from(["M".to_string()]);
        assert!(!BackdoorCriterion::validate(&dag, "T", "Y", &bad).expect("known"));
    }

    #[test]
    fn validate_rejects_a_latent_adjustment_set() {
        let mut dag = dag_from(&[("U", "T"), ("U", "Y"), ("T", "Y")]);
        dag.mark_latent("U").expect("known");
        let z = HashSet::from(["U".to_string()]);
        let err = BackdoorCriterion::validate(&dag, "T", "Y", &z).unwrap_err();
        assert!(matches!(err, IdentificationError::RequiresLatent { .. }));
    }

    #[test]
    fn unknown_variable_is_an_error() {
        let dag = dag_from(&[("T", "Y")]);
        assert!(matches!(
            BackdoorCriterion::find(&dag, "ghost", "Y"),
            Err(IdentificationError::UnknownVariable { .. })
        ));
        assert!(matches!(
            BackdoorCriterion::find(&dag, "T", "ghost"),
            Err(IdentificationError::UnknownVariable { .. })
        ));
    }

    #[test]
    fn front_door_finds_the_mediator() {
        // The canonical smoking / tar / cancer graph with unobserved U.
        let mut dag = dag_from(&[
            ("U", "Smoking"),
            ("U", "Cancer"),
            ("Smoking", "Tar"),
            ("Tar", "Cancer"),
        ]);
        dag.mark_latent("U").expect("known");

        let m = FrontDoorCriterion::find(&dag, "Smoking", "Cancer").expect("front door applies");
        assert_eq!(m.sorted(), vec!["Tar"]);
    }

    #[test]
    fn front_door_declines_when_the_mediator_is_confounded() {
        // U2 confounds the mediator with the outcome, which breaks condition 3.
        let mut dag = dag_from(&[
            ("U", "Smoking"),
            ("U", "Cancer"),
            ("Smoking", "Tar"),
            ("Tar", "Cancer"),
            ("U2", "Tar"),
            ("U2", "Cancer"),
        ]);
        dag.mark_latent("U").expect("known");
        dag.mark_latent("U2").expect("known");
        assert!(FrontDoorCriterion::find(&dag, "Smoking", "Cancer").is_err());
    }

    #[test]
    fn returned_sets_are_verified_not_merely_constructed() {
        // Whatever the search returns must pass an independent validity check.
        let graphs = [
            vec![("W", "T"), ("W", "Y"), ("T", "Y")],
            vec![("A", "T"), ("B", "T"), ("A", "Y"), ("B", "Y"), ("T", "Y")],
            vec![("W", "T"), ("W", "M"), ("T", "M"), ("M", "Y"), ("W", "Y")],
        ];
        for edges in graphs {
            let dag = dag_from(&edges);
            if let Ok(z) = BackdoorCriterion::find(&dag, "T", "Y") {
                assert!(
                    BackdoorCriterion::validate(&dag, "T", "Y", z.variables()).expect("known"),
                    "search returned an invalid set {:?} for {edges:?}",
                    z.sorted()
                );
            }
        }
    }

    #[test]
    fn search_prefers_smaller_sets() {
        // Two confounders, but only W is needed; the search must not return
        // both when one suffices.
        let dag = dag_from(&[("W", "T"), ("W", "Y"), ("T", "Y"), ("Q", "Y")]);
        let z = BackdoorCriterion::find(&dag, "T", "Y").expect("identifiable");
        assert_eq!(z.len(), 1, "returned {:?}", z.sorted());
        assert!(z.contains("W"));
    }
}
