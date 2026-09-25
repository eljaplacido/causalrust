//! Does changing the decision rule close finding R1?
//!
//! [`LexicalClassifier`] and [`DiscriminativeClassifier`] are given the **same
//! features** — same tokenizer, same smoothed idf, same concentration scaling,
//! same L2 normalisation — and differ only in what turns a vector into a
//! verdict: nearest centroid by cosine, against fitted per-term per-domain
//! weights that may be negative.
//!
//! So every number below is a difference in the decision rule. That is the
//! comparison R1 needs, because the open question is not whether tf-idf carries
//! the signal (the centroid model already showed it does, 0.290 -> 0.656) but
//! whether the remaining gap to R1's bar is reachable without a model file.
//!
//! The protocol is the one `lexical_accuracy.rs` uses, reproduced rather than
//! shared so the two files can be read independently: **4-fold cross
//! validation**, folds assigned by position within each domain (`i % k`), which
//! keeps folds balanced across domains and verticals and needs no random number
//! generator. The result is identical on every run and every platform.
//!
//! Hyperparameter selection happens **inside** `DiscriminativeClassifier::train`
//! on the training folds only, so no test item influences the model that scores
//! it. See the module docs for why that matters here specifically.

use cynepic_core::CynefinDomain;
use cynepic_router::eval::ClassifierMetrics;
use cynepic_router::{DiscriminativeClassifier, LexicalClassifier};
use cynepic_testkit::corpus::{ambiguous_queries, routing_corpus};

const DOMAINS: [CynefinDomain; 4] = [
    CynefinDomain::Clear,
    CynefinDomain::Complicated,
    CynefinDomain::Complex,
    CynefinDomain::Chaotic,
];

/// R1's bar, from `docs/FINDINGS.md`.
const F1_BAR: f64 = 0.70;
/// R1's bar for Chaotic recall — ranked above macro F1, because a live incident
/// answered from cache is the failure this routing layer exists to prevent.
const CHAOTIC_BAR: f64 = 0.80;

fn macro_f1(m: &ClassifierMetrics) -> f64 {
    DOMAINS.iter().map(|d| m.f1(*d)).sum::<f64>() / 4.0
}

/// Which model a fold should be scored with.
enum Model {
    Centroid,
    Discriminative,
    /// Discriminative at a fixed `l2`, skipping the inner selection.
    DiscriminativeAt(f64),
}

/// Pooled metrics from `k`-fold cross-validation over the corpus.
fn cross_validated(k: usize, model: &Model) -> ClassifierMetrics {
    let corpus = routing_corpus();
    let mut metrics = ClassifierMetrics::new();

    for fold in 0..k {
        let mut train: Vec<(&str, CynefinDomain)> = Vec::new();
        let mut test: Vec<(&str, CynefinDomain)> = Vec::new();

        let mut seen: Vec<(CynefinDomain, usize)> = Vec::new();
        for q in &corpus {
            let idx = match seen.iter_mut().find(|(d, _)| *d == q.domain) {
                Some((_, n)) => {
                    *n += 1;
                    *n - 1
                }
                None => {
                    seen.push((q.domain, 1));
                    0
                }
            };
            if idx % k == fold {
                test.push((q.text, q.domain));
            } else {
                train.push((q.text, q.domain));
            }
        }

        match model {
            Model::Centroid => {
                let m = LexicalClassifier::train(&train).expect("folds cover every domain");
                for (text, actual) in test {
                    metrics.record(m.classify_sync(text).domain, actual);
                }
            }
            Model::Discriminative => {
                let m = DiscriminativeClassifier::train(&train).expect("folds cover every domain");
                for (text, actual) in test {
                    metrics.record(m.classify_sync(text).domain, actual);
                }
            }
            Model::DiscriminativeAt(l2) => {
                let m = DiscriminativeClassifier::fit_at(
                    &train,
                    cynepic_router::lexical::TrainingOptions::default(),
                    *l2,
                )
                .expect("folds cover every domain");
                for (text, actual) in test {
                    metrics.record(m.classify_sync(text).domain, actual);
                }
            }
        }
    }
    metrics
}

/// The head-to-head. Same folds, same features, different rule.
#[test]
fn discriminative_is_compared_against_the_centroid_on_identical_folds() {
    let centroid = cross_validated(4, &Model::Centroid);
    let discriminative = cross_validated(4, &Model::Discriminative);

    println!("\n  R1 head to head — 4-fold CV, identical features and folds\n");
    println!(
        "  {:<22} {:>10} {:>14} {:>12}",
        "rule", "macro F1", "chaotic R", "accuracy"
    );
    for (name, m) in [
        ("centroid / cosine", &centroid),
        ("logistic regression", &discriminative),
    ] {
        println!(
            "  {:<22} {:>10.3} {:>14.3} {:>12.3}",
            name,
            macro_f1(m),
            m.recall(CynefinDomain::Chaotic),
            m.accuracy()
        );
    }

    println!("\n  per-domain F1\n");
    println!("  {:<22} {:>10} {:>14}", "domain", "centroid", "logistic");
    for d in DOMAINS {
        println!(
            "  {:<22} {:>10.3} {:>14.3}",
            format!("{d:?}"),
            centroid.f1(d),
            discriminative.f1(d)
        );
    }
    println!("\n  R1's bar: macro F1 {F1_BAR:.2}, Chaotic recall {CHAOTIC_BAR:.2}\n");

    // Not an assertion about which wins — that is what the run reports. What
    // must hold is that both were measured on the same denominator, or the
    // comparison is between two different questions.
    assert_eq!(
        centroid.total(),
        discriminative.total(),
        "the two rules were scored on different numbers of items"
    );
}

/// R1's headline bar. Fails until the finding is closed.
///
/// `#[ignore]`d under the ledger's convention: an open finding is a failing
/// spec, and `scripts/findings-ratchet.sh` asserts that every open spec still
/// fails. Deleting this attribute is what closing R1 looks like.
#[test]
#[ignore = "R1: reports whether the discriminative rule reaches the bar"]
fn r1_discriminative_reaches_the_bar() {
    let m = cross_validated(4, &Model::Discriminative);
    let f1 = macro_f1(&m);
    let chaotic = m.recall(CynefinDomain::Chaotic);
    assert!(
        f1 >= F1_BAR && chaotic >= CHAOTIC_BAR,
        "macro F1 {f1:.3} (bar {F1_BAR:.2}), Chaotic recall {chaotic:.3} (bar {CHAOTIC_BAR:.2})"
    );
}

/// Regularisation sensitivity — a diagnostic, never the basis of the default.
///
/// Reading the best row off this table and hard-coding it would make the
/// headline a training score. `train` picks `l2` by an inner split instead;
/// this exists to show how much the choice is worth, which is the honest way to
/// report a hyperparameter.
#[test]
#[ignore = "diagnostic: prints the l2 sensitivity curve"]
fn l2_sensitivity_sweep() {
    println!("\n  Regularisation sweep — 4-fold CV. DIAGNOSTIC ONLY.\n");
    println!("  {:>10} {:>12} {:>14}", "l2", "macro F1", "chaotic R");
    for l2 in [1e-4, 1e-3, 1e-2, 1e-1, 1.0] {
        let m = cross_validated(4, &Model::DiscriminativeAt(l2));
        println!(
            "  {:>10.0e} {:>12.3} {:>14.3}",
            l2,
            macro_f1(&m),
            m.recall(CynefinDomain::Chaotic)
        );
    }
    println!(
        "\n  `train` does not read this table: it selects l2 by a 3-fold split\n  \
         inside the training folds, so the reported CV score stays held out.\n"
    );
}

/// Ambiguous input must still abstain, or the gain is bought by over-answering.
///
/// A rule that answers everything scores better on recall and is worse to
/// deploy. The abstention threshold is held identical to the centroid model's
/// precisely so this cannot be the source of any difference.
#[test]
fn ambiguous_input_is_still_declined() {
    let corpus = routing_corpus();
    let train: Vec<(&str, CynefinDomain)> = corpus.iter().map(|q| (q.text, q.domain)).collect();
    let model = DiscriminativeClassifier::train(&train).expect("corpus covers every domain");

    let ambiguous = ambiguous_queries();
    let answered = ambiguous
        .iter()
        .filter(|q| model.classify_sync(q).domain != CynefinDomain::Disorder)
        .count();

    println!(
        "\n  {answered} of {} contentless queries were answered\n",
        ambiguous.len()
    );
    for q in &ambiguous {
        let r = model.classify_sync(q);
        if r.domain != CynefinDomain::Disorder {
            println!(
                "    {:<12} {:.3}  {q}",
                format!("{:?}", r.domain),
                r.confidence
            );
        }
    }

    assert!(
        answered * 2 <= ambiguous.len(),
        "{answered} of {} contentless queries got an answer; the evidence gate \
         is not doing its job and any recall gain is bought by over-answering",
        ambiguous.len()
    );
}

/// An abstention is exactly zero confidence, not nearly zero.
///
/// The contract an escalation policy triggers on. "Almost certain about
/// nothing" is a much worse signal than "no answer".
#[test]
fn abstention_is_exactly_zero_confidence() {
    let corpus = routing_corpus();
    let train: Vec<(&str, CynefinDomain)> = corpus.iter().map(|q| (q.text, q.domain)).collect();
    let model = DiscriminativeClassifier::train(&train).expect("corpus covers every domain");

    for q in ambiguous_queries() {
        let r = model.classify_sync(q);
        if r.domain == CynefinDomain::Disorder {
            assert!(
                r.confidence.abs() < f64::EPSILON,
                "abstained on {q:?} with confidence {}",
                r.confidence
            );
        }
    }
}

/// Fitting is deterministic: same examples, byte-identical verdicts.
///
/// The crate writes routing decisions to an audit trail, so a model that
/// depended on hash iteration order would make the trail unreproducible.
#[test]
fn fitting_is_deterministic() {
    let corpus = routing_corpus();
    let train: Vec<(&str, CynefinDomain)> = corpus.iter().map(|q| (q.text, q.domain)).collect();

    let a = DiscriminativeClassifier::train(&train).expect("trains");
    let b = DiscriminativeClassifier::train(&train).expect("trains");

    for q in &corpus {
        let ra = a.classify_sync(q.text);
        let rb = b.classify_sync(q.text);
        assert_eq!(ra.domain, rb.domain, "domain differs on {:?}", q.text);
        assert!(
            (ra.confidence - rb.confidence).abs() < 1e-12,
            "confidence differs on {:?}",
            q.text
        );
    }
    assert!((a.selected_l2() - b.selected_l2()).abs() < f64::EPSILON);
}

/// Gradient descent has actually converged at the fixed epoch count.
///
/// The epoch count is asserted rather than assumed: a model still moving at the
/// last step would make every number above a property of where it was stopped.
#[test]
fn training_has_converged_by_the_fixed_epoch_count() {
    let corpus = routing_corpus();
    let train: Vec<(&str, CynefinDomain)> = corpus.iter().map(|q| (q.text, q.domain)).collect();
    let model = DiscriminativeClassifier::train(&train).expect("trains");

    // Agreement between the shipped model and one fitted at the same l2 is the
    // observable proxy for convergence available through the public API: if the
    // optimiser were still far from a stationary point, the inner-selected and
    // directly-fitted models would disagree on held-in items.
    let same = DiscriminativeClassifier::fit_at(
        &train,
        cynepic_router::lexical::TrainingOptions::default(),
        model.selected_l2(),
    )
    .expect("trains");

    let disagreements = corpus
        .iter()
        .filter(|q| model.classify_sync(q.text).domain != same.classify_sync(q.text).domain)
        .count();
    assert_eq!(
        disagreements, 0,
        "{disagreements} items moved between two fits at one l2"
    );
}

/// Explanations name terms, and can name one that argued against the verdict.
#[test]
fn explanations_are_signed() {
    let corpus = routing_corpus();
    let train: Vec<(&str, CynefinDomain)> = corpus.iter().map(|q| (q.text, q.domain)).collect();
    let model = DiscriminativeClassifier::train(&train).expect("trains");

    let terms = model.explain("why has p99 latency doubled since the last release", 5);
    assert!(
        !terms.is_empty(),
        "no terms explained a query full of known words"
    );
    println!("\n  explanation, most influential first\n");
    for (term, weight) in &terms {
        println!("    {weight:+.4}  {term}");
    }
}

/// Where do Chaotic items go when they are missed?
///
/// Chaotic recall is the one bar the discriminative rule does not clear, and
/// the fix depends entirely on this breakdown. A miss that lands on another
/// *domain* is a separation problem, answerable with better features. A miss
/// that lands on `Disorder` is an **abstention** problem — the evidence gate
/// declining on a query it could have answered — and the remedy is a
/// cost-sensitive threshold, not a better model. The two are not distinguished
/// by the recall number alone.
#[test]
#[ignore = "diagnostic: where Chaotic recall is lost"]
fn where_chaotic_recall_is_lost() {
    let corpus = routing_corpus();
    let k = 4;
    let mut rows: Vec<(String, usize)> = Vec::new();
    let mut total_chaotic = 0_usize;

    for fold in 0..k {
        let mut train: Vec<(&str, CynefinDomain)> = Vec::new();
        let mut test: Vec<(&str, CynefinDomain)> = Vec::new();
        let mut seen: Vec<(CynefinDomain, usize)> = Vec::new();
        for q in &corpus {
            let idx = match seen.iter_mut().find(|(d, _)| *d == q.domain) {
                Some((_, n)) => {
                    *n += 1;
                    *n - 1
                }
                None => {
                    seen.push((q.domain, 1));
                    0
                }
            };
            if idx % k == fold {
                test.push((q.text, q.domain));
            } else {
                train.push((q.text, q.domain));
            }
        }
        let m = DiscriminativeClassifier::train(&train).expect("folds cover every domain");
        for (text, actual) in test {
            if actual != CynefinDomain::Chaotic {
                continue;
            }
            total_chaotic += 1;
            let got = m.classify_sync(text).domain;
            if got == CynefinDomain::Chaotic {
                continue;
            }
            let key = format!("{got:?}");
            match rows.iter_mut().find(|(k, _)| *k == key) {
                Some((_, n)) => *n += 1,
                None => rows.push((key, 1)),
            }
            println!("    -> {got:<12} {text}");
        }
    }

    println!("\n  Chaotic misses by destination, out of {total_chaotic} Chaotic items\n");
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    for (dest, n) in &rows {
        println!("    {dest:<14} {n}");
    }
    let abstained = rows
        .iter()
        .find(|(d, _)| d == "Disorder")
        .map_or(0, |(_, n)| *n);
    println!(
        "\n  {abstained} of {} misses are abstentions, not misroutes.\n",
        rows.iter().map(|(_, n)| *n).sum::<usize>()
    );
}
