//! A labelled query corpus for measuring routing accuracy.
//!
//! # How this corpus was written, and why it matters
//!
//! These queries were written **without reference to the classifier's keyword
//! lists**, from the four application domains the project targets: agentic
//! workflows, data engineering, systems and DevOps, and business intelligence.
//!
//! That constraint is the whole point. A corpus assembled by reading the
//! keyword table and writing queries around it measures nothing — the
//! classifier would score highly on it by construction, and the number would
//! be a restatement of the implementation rather than a test of it. Any
//! keyword that does appear below is there because it is how someone would
//! naturally phrase the question.
//!
//! Consequently a low score here is informative rather than embarrassing: it
//! measures the gap between a keyword matcher and real phrasing, which is
//! exactly the gap an embedding classifier would be bought to close. Publishing
//! that gap is what makes a later claim of improvement meaningful.
//!
//! # Labelling
//!
//! The Cynefin domain of a question is a property of the relationship between
//! cause and effect, not of its subject matter:
//!
//! - **Clear** — the answer is a lookup. Cause and effect are obvious and the
//!   response is a known best practice.
//! - **Complicated** — the answer requires analysis or expertise. Cause and
//!   effect are knowable but not obvious; there is a right answer and a
//!   procedure for finding it.
//! - **Complex** — the answer requires experiment. Cause and effect are only
//!   coherent in retrospect; the system responds to being probed.
//! - **Chaotic** — no discernible relationship between cause and effect yet.
//!   Act first to establish stability, analyse later.
//!
//! `Disorder` is deliberately absent as a label: it is the state of *not
//! knowing which domain applies*, so it is a classifier output for ambiguous
//! input, never a ground-truth property of a query. [`ambiguous_queries`]
//! covers that case separately.

use cynepic_core::CynefinDomain;

/// One labelled query.
#[derive(Debug, Clone)]
pub struct LabelledQuery {
    /// The query as a user would type it.
    pub text: &'static str,
    /// The domain the query belongs to.
    pub domain: CynefinDomain,
    /// Which application area it was drawn from, for per-vertical breakdowns.
    pub vertical: Vertical,
}

/// Application area a query came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vertical {
    /// Agent planning, tool selection, self-correction.
    Agentic,
    /// Pipelines, schemas, data quality, lineage.
    DataEngineering,
    /// Incidents, deployments, performance, reliability.
    Systems,
    /// Attribution, pricing, forecasting, policy evaluation.
    BusinessIntelligence,
}

impl Vertical {
    /// Short label for tables.
    pub fn label(self) -> &'static str {
        match self {
            Self::Agentic => "agentic",
            Self::DataEngineering => "data-eng",
            Self::Systems => "systems",
            Self::BusinessIntelligence => "bi",
        }
    }

    /// Every vertical, for iteration.
    pub fn all() -> [Self; 4] {
        [
            Self::Agentic,
            Self::DataEngineering,
            Self::Systems,
            Self::BusinessIntelligence,
        ]
    }
}

/// Shorthand for a corpus entry.
const fn q(text: &'static str, domain: CynefinDomain, vertical: Vertical) -> LabelledQuery {
    LabelledQuery {
        text,
        domain,
        vertical,
    }
}

/// The labelled routing corpus: 96 queries, 24 per domain, 6 per
/// domain-vertical pair.
///
/// Balanced deliberately. An unbalanced corpus lets a classifier score well by
/// favouring the majority class, and accuracy on it would say more about the
/// corpus than the classifier.
pub fn routing_corpus() -> Vec<LabelledQuery> {
    use CynefinDomain::{Chaotic, Clear, Complex, Complicated};
    use Vertical::{Agentic, BusinessIntelligence, DataEngineering, Systems};

    vec![
        // ---- Clear: the answer is a lookup ------------------------------
        q(
            "which model is configured for the summarisation step",
            Clear,
            Agentic,
        ),
        q("list the tools this agent has access to", Clear, Agentic),
        q("what is the current retry limit", Clear, Agentic),
        q("show me the system prompt for the router", Clear, Agentic),
        q("how many tokens did that call consume", Clear, Agentic),
        q("what is the timeout on tool invocations", Clear, Agentic),
        q(
            "what columns are in the orders table",
            Clear,
            DataEngineering,
        ),
        q(
            "when did the nightly job last succeed",
            Clear,
            DataEngineering,
        ),
        q(
            "how many rows landed in staging yesterday",
            Clear,
            DataEngineering,
        ),
        q(
            "what is the retention period on this bucket",
            Clear,
            DataEngineering,
        ),
        q(
            "which upstream feeds the customer dimension",
            Clear,
            DataEngineering,
        ),
        q(
            "show the schema version currently deployed",
            Clear,
            DataEngineering,
        ),
        q("what version is running in production", Clear, Systems),
        q("how much disk is free on the build host", Clear, Systems),
        q("list the open pull requests on this repo", Clear, Systems),
        q("what is the configured heap size", Clear, Systems),
        q("which region is this cluster in", Clear, Systems),
        q("show me the last deployment timestamp", Clear, Systems),
        q("what was revenue last quarter", Clear, BusinessIntelligence),
        q(
            "how many active accounts do we have",
            Clear,
            BusinessIntelligence,
        ),
        q(
            "what is our current list price in Germany",
            Clear,
            BusinessIntelligence,
        ),
        q("show headcount by department", Clear, BusinessIntelligence),
        q(
            "what is the definition of a qualified lead",
            Clear,
            BusinessIntelligence,
        ),
        q(
            "how many support tickets closed this week",
            Clear,
            BusinessIntelligence,
        ),
        // ---- Complicated: analysis by a knowable procedure ---------------
        q(
            "why did the agent choose the search tool over the database",
            Complicated,
            Agentic,
        ),
        q(
            "which step in the chain is responsible for the latency",
            Complicated,
            Agentic,
        ),
        q(
            "diagnose why this plan keeps looping back to the same node",
            Complicated,
            Agentic,
        ),
        q(
            "attribute the cost increase across the tool calls",
            Complicated,
            Agentic,
        ),
        q(
            "determine whether the reranker is improving answer quality",
            Complicated,
            Agentic,
        ),
        q(
            "work out which prompt change reduced refusals",
            Complicated,
            Agentic,
        ),
        q(
            "why are these two tables disagreeing on totals",
            Complicated,
            DataEngineering,
        ),
        q(
            "trace where the duplicate records are being introduced",
            Complicated,
            DataEngineering,
        ),
        q(
            "what is driving the increase in null rates on this column",
            Complicated,
            DataEngineering,
        ),
        q(
            "analyse which join is producing the row explosion",
            Complicated,
            DataEngineering,
        ),
        q(
            "figure out why the incremental load diverges from a full refresh",
            Complicated,
            DataEngineering,
        ),
        q(
            "explain the discrepancy between the warehouse and the source system",
            Complicated,
            DataEngineering,
        ),
        q(
            "why has p99 latency doubled since the last release",
            Complicated,
            Systems,
        ),
        q(
            "determine which query is saturating the connection pool",
            Complicated,
            Systems,
        ),
        q(
            "analyse the memory growth in the worker process",
            Complicated,
            Systems,
        ),
        q(
            "which change in this release explains the error rate",
            Complicated,
            Systems,
        ),
        q(
            "profile where the request spends its time",
            Complicated,
            Systems,
        ),
        q(
            "work out why the cache hit rate dropped after the migration",
            Complicated,
            Systems,
        ),
        q(
            "what is driving the decline in conversion this month",
            Complicated,
            BusinessIntelligence,
        ),
        q(
            "attribute revenue growth across channels",
            Complicated,
            BusinessIntelligence,
        ),
        q(
            "determine the effect of the discount on margin",
            Complicated,
            BusinessIntelligence,
        ),
        q(
            "analyse which segment is responsible for the churn increase",
            Complicated,
            BusinessIntelligence,
        ),
        q(
            "quantify how much the campaign contributed to signups",
            Complicated,
            BusinessIntelligence,
        ),
        q(
            "explain the gap between forecast and actual",
            Complicated,
            BusinessIntelligence,
        ),
        // ---- Complex: the answer requires experiment ---------------------
        q(
            "how should we structure the agent's memory for long sessions",
            Complex,
            Agentic,
        ),
        q(
            "would a different decomposition strategy work better here",
            Complex,
            Agentic,
        ),
        q(
            "explore whether giving the agent fewer tools improves reliability",
            Complex,
            Agentic,
        ),
        q(
            "what might happen if we let the agent retry indefinitely",
            Complex,
            Agentic,
        ),
        q(
            "trial a self-critique step and see whether quality moves",
            Complex,
            Agentic,
        ),
        q(
            "we are not sure which planning approach suits this workload",
            Complex,
            Agentic,
        ),
        q(
            "how should we model slowly changing dimensions for this business",
            Complex,
            DataEngineering,
        ),
        q(
            "would streaming ingestion suit this workload better than batch",
            Complex,
            DataEngineering,
        ),
        q(
            "experiment with partitioning strategies for this table",
            Complex,
            DataEngineering,
        ),
        q(
            "it is unclear which grain the fact table should be at",
            Complex,
            DataEngineering,
        ),
        q(
            "probe whether denormalising would help the analysts",
            Complex,
            DataEngineering,
        ),
        q(
            "what data contract would actually hold up between these teams",
            Complex,
            DataEngineering,
        ),
        q(
            "would splitting this service reduce our failure blast radius",
            Complex,
            Systems,
        ),
        q(
            "we are unsure whether the bottleneck is architectural or configuration",
            Complex,
            Systems,
        ),
        q(
            "try a canary and see how the system responds under real traffic",
            Complex,
            Systems,
        ),
        q(
            "explore whether backpressure or sharding is the better direction",
            Complex,
            Systems,
        ),
        q(
            "it is not obvious how this system will behave at ten times the load",
            Complex,
            Systems,
        ),
        q(
            "what might a graceful degradation strategy look like here",
            Complex,
            Systems,
        ),
        q(
            "what would happen to demand if we changed the pricing model",
            Complex,
            BusinessIntelligence,
        ),
        q(
            "we do not know which market to enter next",
            Complex,
            BusinessIntelligence,
        ),
        q(
            "run a small test to see whether the new packaging resonates",
            Complex,
            BusinessIntelligence,
        ),
        q(
            "how might customers respond to a usage-based plan",
            Complex,
            BusinessIntelligence,
        ),
        q(
            "explore whether the loyalty scheme is worth expanding",
            Complex,
            BusinessIntelligence,
        ),
        q(
            "it is uncertain how the competitor's move will affect us",
            Complex,
            BusinessIntelligence,
        ),
        // ---- Chaotic: act now, analyse later -----------------------------
        q(
            "the agent is calling the payments tool in a loop right now",
            Chaotic,
            Agentic,
        ),
        q(
            "stop everything, it is deleting records it should not touch",
            Chaotic,
            Agentic,
        ),
        q(
            "the assistant is leaking customer data into responses",
            Chaotic,
            Agentic,
        ),
        q(
            "halt the run, the tool is charging real money on every retry",
            Chaotic,
            Agentic,
        ),
        q(
            "it has gone rogue and is escalating its own permissions",
            Chaotic,
            Agentic,
        ),
        q(
            "kill the agent now before it sends any more emails",
            Chaotic,
            Agentic,
        ),
        q(
            "the pipeline has been writing corrupt data for six hours",
            Chaotic,
            DataEngineering,
        ),
        q(
            "we have just overwritten production with test data",
            Chaotic,
            DataEngineering,
        ),
        q(
            "the warehouse is down and the board meeting is in an hour",
            Chaotic,
            DataEngineering,
        ),
        q(
            "someone dropped the source table, we need to act now",
            Chaotic,
            DataEngineering,
        ),
        q(
            "customer records are being written to the wrong tenant",
            Chaotic,
            DataEngineering,
        ),
        q(
            "stop the backfill, it is corrupting historical partitions",
            Chaotic,
            DataEngineering,
        ),
        q("the site is down and we do not know why", Chaotic, Systems),
        q(
            "we are being actively exploited, roll back immediately",
            Chaotic,
            Systems,
        ),
        q(
            "everything is failing and the pager will not stop",
            Chaotic,
            Systems,
        ),
        q(
            "production is on fire, we need to stop the bleeding",
            Chaotic,
            Systems,
        ),
        q(
            "the database is unreachable and customers cannot log in",
            Chaotic,
            Systems,
        ),
        q(
            "all regions are erroring, cut traffic now",
            Chaotic,
            Systems,
        ),
        q(
            "we have just published incorrect financial figures publicly",
            Chaotic,
            BusinessIntelligence,
        ),
        q(
            "the pricing page is showing zero for every plan",
            Chaotic,
            BusinessIntelligence,
        ),
        q(
            "we are invoicing customers twice and they are noticing",
            Chaotic,
            BusinessIntelligence,
        ),
        q(
            "regulators have called and we need an answer today",
            Chaotic,
            BusinessIntelligence,
        ),
        q(
            "a competitor has published our confidential roadmap",
            Chaotic,
            BusinessIntelligence,
        ),
        q(
            "our largest account is cancelling this afternoon",
            Chaotic,
            BusinessIntelligence,
        ),
    ]
}

/// Queries with no defensible single label.
///
/// The correct output for these is `Disorder` or a low-confidence, high-entropy
/// result — not a confident guess. A classifier that answers these as decisively
/// as it answers the rest is miscalibrated, and that is worth measuring
/// separately because it is invisible in an accuracy score.
pub fn ambiguous_queries() -> Vec<&'static str> {
    vec![
        "help",
        "what should we do",
        "the thing is broken",
        "can you look at this",
        "hmm",
        "not sure about this one",
        "please advise",
        "thoughts?",
        "is that normal",
        "any ideas",
    ]
}

/// Every query for one domain.
pub fn queries_for(domain: CynefinDomain) -> Vec<LabelledQuery> {
    routing_corpus()
        .into_iter()
        .filter(|q| q.domain == domain)
        .collect()
}

/// Every query for one vertical.
pub fn queries_for_vertical(vertical: Vertical) -> Vec<LabelledQuery> {
    routing_corpus()
        .into_iter()
        .filter(|q| q.vertical == vertical)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn corpus_is_balanced_across_domains() {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for entry in routing_corpus() {
            *counts.entry(format!("{:?}", entry.domain)).or_default() += 1;
        }
        assert_eq!(counts.len(), 4, "expected exactly four labelled domains");
        for (domain, n) in &counts {
            assert_eq!(*n, 24, "{domain} has {n} queries, expected 24");
        }
    }

    #[test]
    fn corpus_is_balanced_across_verticals() {
        for vertical in Vertical::all() {
            let n = queries_for_vertical(vertical).len();
            assert_eq!(n, 24, "{} has {n} queries, expected 24", vertical.label());
        }
    }

    #[test]
    fn every_domain_vertical_pair_is_represented() {
        // Without this, a per-vertical breakdown could silently be computed
        // from cells that do not exist.
        let mut seen: HashMap<(String, &str), usize> = HashMap::new();
        for entry in routing_corpus() {
            *seen
                .entry((format!("{:?}", entry.domain), entry.vertical.label()))
                .or_default() += 1;
        }
        assert_eq!(seen.len(), 16, "expected 4 domains x 4 verticals");
        for ((domain, vertical), n) in &seen {
            assert_eq!(*n, 6, "{domain}/{vertical} has {n}, expected 6");
        }
    }

    #[test]
    fn no_query_appears_twice() {
        let corpus = routing_corpus();
        let unique: HashSet<&str> = corpus.iter().map(|q| q.text).collect();
        assert_eq!(
            unique.len(),
            corpus.len(),
            "duplicate queries inflate whichever class they land in"
        );
    }

    #[test]
    fn disorder_is_never_a_ground_truth_label() {
        // Disorder means "we cannot tell which domain applies", which is a
        // statement about the classifier's state, not about the query.
        assert!(
            routing_corpus()
                .iter()
                .all(|q| q.domain != CynefinDomain::Disorder),
            "Disorder is a classifier output, not a property of a query"
        );
    }

    #[test]
    fn ambiguous_queries_are_genuinely_short_of_signal() {
        // Guards against someone "fixing" a low ambiguity score by quietly
        // making these queries more specific.
        for text in ambiguous_queries() {
            assert!(
                text.split_whitespace().count() <= 6,
                "'{text}' is too specific to be an ambiguity probe"
            );
        }
    }

    #[test]
    fn queries_are_lowercase_and_unpunctuated_like_real_input() {
        for entry in routing_corpus() {
            assert!(
                !entry.text.ends_with('?') && !entry.text.ends_with('.'),
                "'{}' — corpus entries should be raw input, not tidied prose",
                entry.text
            );
        }
    }
}
