use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Detects repetitive execution patterns in workflow graphs.
/// Tracks node visits and edge traversals to identify loops and thrashing.
#[derive(Debug, Clone)]
pub struct LoopDetector {
    /// Max times any single node can be visited before flagging.
    max_node_visits: usize,
    /// Max consecutive alternations between two nodes (A->B->A->B...).
    max_alternations: usize,
    /// Track visit counts per node.
    visit_counts: HashMap<String, usize>,
    /// Recent node execution history (for pattern detection).
    history: Vec<String>,
    /// Max history length to retain.
    max_history: usize,
}

/// Describes a loop or runaway violation detected in execution history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopViolation {
    /// A node was visited more than the allowed limit.
    NodeOvervisited {
        node: String,
        visits: usize,
        limit: usize,
    },
    /// Two nodes are alternating back and forth (thrashing).
    AlternationDetected {
        node_a: String,
        node_b: String,
        count: usize,
        limit: usize,
    },
    /// Total execution steps exceeded global limit.
    TotalStepsExceeded { steps: usize, limit: usize },
}

impl LoopDetector {
    /// Create a new loop detector with the given limits.
    ///
    /// - `max_node_visits`: maximum times a single node can be visited
    /// - `max_alternations`: maximum consecutive A->B->A->B alternations allowed
    pub fn new(max_node_visits: usize, max_alternations: usize) -> Self {
        Self {
            max_node_visits,
            max_alternations,
            visit_counts: HashMap::new(),
            history: Vec::new(),
            max_history: max_alternations * 4, // keep enough history for detection
        }
    }

    /// Record a node visit. Returns a violation if a loop pattern is detected.
    pub fn record_visit(&mut self, node_id: &str) -> Option<LoopViolation> {
        // Update visit count
        let count = self.visit_counts.entry(node_id.to_string()).or_insert(0);
        *count += 1;
        let visits = *count;

        // Update history
        self.history.push(node_id.to_string());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Check for overvisit first
        if visits > self.max_node_visits {
            return Some(LoopViolation::NodeOvervisited {
                node: node_id.to_string(),
                visits,
                limit: self.max_node_visits,
            });
        }

        // Check for alternation pattern
        self.detect_alternation()
    }

    /// Check if a specific node has been visited too many times.
    pub fn is_overvisited(&self, node_id: &str) -> bool {
        self.visit_counts
            .get(node_id)
            .map(|&c| c > self.max_node_visits)
            .unwrap_or(false)
    }

    /// Get the visit count for a node.
    pub fn visit_count(&self, node_id: &str) -> usize {
        self.visit_counts.get(node_id).copied().unwrap_or(0)
    }

    /// Check for alternation pattern in recent history.
    /// Looks for A->B->A->B... pattern in the last `2 * max_alternations` entries.
    fn detect_alternation(&self) -> Option<LoopViolation> {
        let window_size = 2 * self.max_alternations;
        if self.history.len() < window_size {
            return None;
        }

        let window = &self.history[self.history.len() - window_size..];

        // All even-indexed entries should be the same node, all odd-indexed the other
        let node_a = &window[0];
        let node_b = &window[1];

        // Must be two distinct nodes
        if node_a == node_b {
            return None;
        }

        let is_alternating = window.iter().enumerate().all(|(i, node)| {
            if i % 2 == 0 {
                node == node_a
            } else {
                node == node_b
            }
        });

        if is_alternating {
            Some(LoopViolation::AlternationDetected {
                node_a: node_a.clone(),
                node_b: node_b.clone(),
                count: self.max_alternations,
                limit: self.max_alternations,
            })
        } else {
            None
        }
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.visit_counts.clear();
        self.history.clear();
    }

    /// Get the full execution history.
    pub fn history(&self) -> &[String] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_execution_no_violation() {
        let mut detector = LoopDetector::new(5, 3);
        assert!(detector.record_visit("A").is_none());
        assert!(detector.record_visit("B").is_none());
        assert!(detector.record_visit("C").is_none());
        assert!(detector.record_visit("A").is_none());
        assert_eq!(detector.visit_count("A"), 2);
        assert_eq!(detector.visit_count("B"), 1);
    }

    #[test]
    fn node_overvisit_detected() {
        let mut detector = LoopDetector::new(3, 10);
        assert!(detector.record_visit("X").is_none());
        assert!(detector.record_visit("X").is_none());
        assert!(detector.record_visit("X").is_none());
        let violation = detector.record_visit("X");
        assert!(violation.is_some());
        match violation.unwrap() {
            LoopViolation::NodeOvervisited { node, visits, limit } => {
                assert_eq!(node, "X");
                assert_eq!(visits, 4);
                assert_eq!(limit, 3);
            }
            _ => panic!("Expected NodeOvervisited"),
        }
        assert!(detector.is_overvisited("X"));
    }

    #[test]
    fn alternation_detected() {
        let mut detector = LoopDetector::new(100, 3);
        // Need 2 * 3 = 6 alternating entries: A B A B A B
        assert!(detector.record_visit("A").is_none());
        assert!(detector.record_visit("B").is_none());
        assert!(detector.record_visit("A").is_none());
        assert!(detector.record_visit("B").is_none());
        assert!(detector.record_visit("A").is_none());
        let violation = detector.record_visit("B");
        assert!(violation.is_some());
        match violation.unwrap() {
            LoopViolation::AlternationDetected {
                node_a,
                node_b,
                count,
                limit,
            } => {
                assert_eq!(node_a, "A");
                assert_eq!(node_b, "B");
                assert_eq!(count, 3);
                assert_eq!(limit, 3);
            }
            _ => panic!("Expected AlternationDetected"),
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut detector = LoopDetector::new(3, 3);
        detector.record_visit("A");
        detector.record_visit("A");
        detector.record_visit("A");
        assert_eq!(detector.visit_count("A"), 3);
        assert!(!detector.history().is_empty());

        detector.reset();

        assert_eq!(detector.visit_count("A"), 0);
        assert!(detector.history().is_empty());
        // Should work fine again after reset
        assert!(detector.record_visit("A").is_none());
    }
}
