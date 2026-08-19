//! Measured routing accuracy, as regression guards and open findings.
//!
//! The corpus in `cynepic-testkit::corpus` was written without reference to the
//! classifier's keyword lists, so these numbers measure the classifier rather
//! than restate it.
//!
//! # The headline result
//!
//! `KeywordClassifier` scores **macro F1 0.290** on 96 balanced queries, with
//! **0.000 recall on Chaotic** — every one of 24 live-incident queries is
//! missed. The decomposition says why:
//!
//! ```text
//! no keyword matched at all (-> Disorder):  75/96  (78%)
//! matched, but the wrong domain:             2/96  ( 2%)
//! ```
//!
//! Those two failures call for opposite remedies, and only the second is a
//! tuning problem. At 78% no-signal the keyword approach has no *reach* on
//! natural phrasing, and editing the lists against this corpus would overfit to
//! it rather than fix anything. That is finding R1, and it is what the
//! embedding classifier on the roadmap is for.
//!
//! # The result that makes it survivable
//!
//! When the classifier has no signal it returns `Disorder` with zero
//! confidence, rather than guessing. Only 2% of queries are confidently
//! misrouted. A system that abstains is one an escalation policy can work with;
//! a system that guesses confidently is not. Those guards are asserted below
//! and must not regress while R1 is open.

use cynepic_core::CynefinDomain;
use cynepic_router::classifier::{KeywordClassifier, QueryClassifier};
use cynepic_router::eval::ClassifierMetrics;
use cynepic_testkit::corpus::{ambiguous_queries, routing_corpus};

/// Run the whole corpus and return the metrics.
async fn measure() -> (ClassifierMetrics, usize, usize) {
    let classifier = KeywordClassifier::default_patterns();
    let mut metrics = ClassifierMetrics::new();
    let mut no_signal = 0usize;
    let mut wrong_with_signal = 0usize;

    for entry in routing_corpus() {
        let result = classifier
            .classify(entry.text)
            .await
            .expect("keyword classification is infallible");
        metrics.record(result.domain, entry.domain);
        if result.domain == CynefinDomain::Disorder {
            no_signal += 1;
        } else if result.domain != entry.domain {
            wrong_with_signal += 1;
        }
    }
    (metrics, no_signal, wrong_with_signal)
}

fn macro_f1(metrics: &ClassifierMetrics) -> f64 {
    [
        CynefinDomain::Clear,
        CynefinDomain::Complicated,
        CynefinDomain::Complex,
        CynefinDomain::Chaotic,
    ]
    .iter()
    .map(|d| metrics.f1(*d))
    .sum::<f64>()
        / 4.0
}

// ===========================================================================
// Guards — behaviour that must not regress while R1 is open
// ===========================================================================

/// The classifier must abstain rather than guess when it has no signal.
///
/// This is the property that makes a weak classifier safe to deploy behind an
/// escalation policy. A confident wrong route is worse than an admitted
/// unknown, because only the second can be caught downstream.
#[tokio::test]
async fn abstains_on_ambiguous_input_rather_than_guessing() {
    let classifier = KeywordClassifier::default_patterns();
    for text in ambiguous_queries() {
        let result = classifier.classify(text).await.expect("infallible");
        assert_eq!(
            result.domain,
            CynefinDomain::Disorder,
            "'{text}' should not produce a confident domain"
        );
        assert!(
            result.confidence <= 0.0,
            "'{text}' produced confidence {}",
            result.confidence
        );
    }
}

/// Confident misrouting must stay rare.
///
/// Currently 2 of 96. This is the number that must not grow: a classifier that
/// starts guessing to raise its accuracy would trade an honest abstention for a
/// silent error, which is the wrong trade at every ratio.
#[tokio::test]
async fn confident_misrouting_stays_rare() {
    let (_, _, wrong_with_signal) = measure().await;
    assert!(
        wrong_with_signal <= 4,
        "{wrong_with_signal} of 96 queries were confidently misrouted; \
         abstention is preferable to a guess"
    );
}

/// Precision must stay high on the domains that do fire.
///
/// The classifier is silent far too often, but when it does speak it is almost
/// always right — precision is 1.000 on Complicated and Complex, 0.833 on
/// Clear. That is the property worth protecting while recall is fixed.
#[tokio::test]
async fn precision_is_high_where_the_classifier_fires() {
    let (metrics, _, _) = measure().await;
    for domain in [
        CynefinDomain::Clear,
        CynefinDomain::Complicated,
        CynefinDomain::Complex,
    ] {
        let p = metrics.precision(domain);
        assert!(
            p >= 0.8,
            "{domain:?} precision fell to {p:.3}; the classifier has started guessing"
        );
    }
}

/// Ambiguous input must be measurably more uncertain than labelled input.
///
/// Escalation is triggered by exactly this gap, so a classifier that cannot
/// separate the two cannot support an escalation policy no matter what its
/// accuracy is.
#[tokio::test]
async fn entropy_separates_ambiguous_from_answerable() {
    let classifier = KeywordClassifier::default_patterns();

    let mut ambiguous_total = 0.0;
    let ambiguous = ambiguous_queries();
    for text in &ambiguous {
        ambiguous_total += classifier.classify(text).await.expect("infallible").entropy;
    }
    #[allow(clippy::cast_precision_loss)]
    let ambiguous_mean = ambiguous_total / ambiguous.len() as f64;

    // Restricted to queries the classifier can actually read; including the
    // 78% it cannot would compare ambiguity against ambiguity.
    let mut answered_total = 0.0;
    let mut answered = 0usize;
    for entry in routing_corpus() {
        let result = classifier.classify(entry.text).await.expect("infallible");
        if result.domain != CynefinDomain::Disorder {
            answered_total += result.entropy;
            answered += 1;
        }
    }
    assert!(answered > 10, "too few answered queries to compare");
    #[allow(clippy::cast_precision_loss)]
    let answered_mean = answered_total / answered as f64;

    assert!(
        ambiguous_mean > answered_mean + 0.05,
        "entropy does not separate ambiguous ({ambiguous_mean:.3}) from \
         answerable ({answered_mean:.3})"
    );
}

/// Scoring must not depend on how many keywords an author wrote.
///
/// The original score was `matches / keywords.len()`, so a single match against
/// the four-keyword `Clear` list outscored a single match against the
/// seven-keyword `Complicated` list — 0.25 to 0.14. List length is an authoring
/// artifact and carries no evidence about the query.
#[tokio::test]
async fn score_does_not_depend_on_keyword_list_length() {
    let classifier = KeywordClassifier::default_patterns();

    // "cause" is a Complicated keyword; "define" is a Clear one. A query
    // containing only the Complicated keyword must route to Complicated,
    // regardless of that list being the longer one.
    let result = classifier
        .classify("what was the cause")
        .await
        .expect("infallible");
    assert_eq!(
        result.domain,
        CynefinDomain::Complicated,
        "a Complicated-only query routed to {:?}; scores were {:?}",
        result.domain,
        result.all_scores
    );
}

// ===========================================================================
// R1 — the keyword classifier has no reach on natural phrasing (OPEN)
// ===========================================================================

/// Live incidents must be routed to Chaotic.
///
/// This is the highest-stakes route in the system: answering "the site is down"
/// from a cached lookup is the failure mode the whole Cynefin split exists to
/// prevent. Measured recall is **0.000** — all 24 incident queries are missed,
/// because none of them happen to contain the words "emergency", "crisis",
/// "outage", "breach", "urgent" or "critical failure".
///
/// The fix is not more keywords. It is the embedding classifier: 78% of the
/// corpus matches nothing at all, which is a reach problem, and tuning the
/// lists against this corpus would overfit to it while leaving the next
/// phrasing just as unreachable.
#[tokio::test]
#[ignore = "R1: keyword classifier has 0.000 recall on Chaotic; needs the embedding classifier"]
async fn r1_chaotic_queries_are_routed_to_chaotic() {
    let (metrics, _, _) = measure().await;
    let recall = metrics.recall(CynefinDomain::Chaotic);
    assert!(
        recall >= 0.80,
        "Chaotic recall is {recall:.3}; a live incident answered from cache is \
         the failure this routing layer exists to prevent"
    );
}

/// Overall routing quality must be usable.
///
/// Macro F1 rather than accuracy, because the corpus is balanced by
/// construction and macro F1 cannot be raised by favouring a majority class.
/// Measured 0.290 against a random baseline of roughly 0.25.
#[tokio::test]
#[ignore = "R1: macro F1 is 0.290, barely above the 0.25 random baseline"]
async fn r1_macro_f1_is_usable() {
    let (metrics, _, _) = measure().await;
    let f1 = macro_f1(&metrics);
    assert!(
        f1 >= 0.70,
        "macro F1 is {f1:.3}; a four-class random baseline is about 0.25"
    );
}

/// The classifier must have signal on most natural phrasing.
///
/// The root cause behind both specs above, stated directly so a fix can be
/// aimed at it: 75 of 96 queries match no keyword at all.
#[tokio::test]
#[ignore = "R1: 78% of natural queries match no keyword; this is a reach problem"]
async fn r1_most_queries_produce_signal() {
    let (_, no_signal, _) = measure().await;
    assert!(
        no_signal <= 20,
        "{no_signal} of 96 queries produced no signal at all; the classifier \
         cannot read ordinary phrasing"
    );
}
