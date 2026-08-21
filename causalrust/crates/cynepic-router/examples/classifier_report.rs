//! Measured routing accuracy against the labelled corpus.
//!
//! ```bash
//! cargo run -p cynepic-router --example classifier_report --release
//! ```
//!
//! # What this measures
//!
//! Per-domain precision, recall and F1 against `cynepic-testkit`'s corpus of 96
//! queries, written independently of the classifier's keyword lists. A corpus
//! built by reading those lists would score highly by construction and measure
//! nothing.
//!
//! # Why accuracy is the least interesting column
//!
//! Misrouting is not symmetric. Sending a Chaotic query — a live incident — to
//! the Clear route means answering "the site is down" with a cached lookup. The
//! reverse, sending a Clear query to the Chaotic route, wastes money and a
//! human's attention. Both are errors; only one of them is a failure that
//! matters.
//!
//! The **misrouting cost** column prices that asymmetry using the routing
//! tiers, and the **escalation** row measures the behaviour that actually
//! protects a production system: a classifier that says "I do not know" on
//! input it cannot read. A confident wrong route is worse than an honest
//! abstention, and accuracy alone cannot tell them apart.

use cynepic_core::CynefinDomain;
use cynepic_router::classifier::{KeywordClassifier, QueryClassifier};
use cynepic_router::eval::ClassifierMetrics;
use cynepic_testkit::corpus::{Vertical, ambiguous_queries, queries_for_vertical, routing_corpus};

#[tokio::main]
async fn main() {
    let classifier = KeywordClassifier::default_patterns();
    let corpus = routing_corpus();

    println!("cynepic-router — classification accuracy");
    println!("{} labelled queries, 24 per domain\n", corpus.len());

    // ------------------------------------------------------------ overall
    let mut metrics = ClassifierMetrics::new();
    let mut entropies: Vec<f64> = Vec::new();
    let mut no_signal = 0usize;
    let mut wrong_with_signal = 0usize;

    for entry in &corpus {
        let Ok(result) = classifier.classify(entry.text).await else {
            continue;
        };
        metrics.record(result.domain, entry.domain);
        entropies.push(result.entropy);
        if result.domain == CynefinDomain::Disorder {
            no_signal += 1;
        } else if result.domain != entry.domain {
            wrong_with_signal += 1;
        }
    }

    println!("── per domain ───────────────────────────────────────────────────");
    println!(
        "  {:<14} {:>10} {:>10} {:>10}",
        "domain", "precision", "recall", "F1"
    );
    for domain in [
        CynefinDomain::Clear,
        CynefinDomain::Complicated,
        CynefinDomain::Complex,
        CynefinDomain::Chaotic,
    ] {
        println!(
            "  {:<14} {:>9.3} {:>10.3} {:>10.3}",
            format!("{domain:?}"),
            metrics.precision(domain),
            metrics.recall(domain),
            metrics.f1(domain)
        );
    }

    let macro_f1 = [
        CynefinDomain::Clear,
        CynefinDomain::Complicated,
        CynefinDomain::Complex,
        CynefinDomain::Chaotic,
    ]
    .iter()
    .map(|d| metrics.f1(*d))
    .sum::<f64>()
        / 4.0;

    println!("\n  accuracy   {:.3}", metrics.accuracy());
    println!("  macro F1   {macro_f1:.3}");

    // Macro F1 rather than accuracy is the headline because the corpus is
    // balanced by construction; on unbalanced input they diverge and macro F1
    // is the one that cannot be gamed by favouring a majority class.

    // The decisive diagnostic. A classifier can be wrong in two ways, and they
    // call for opposite remedies: it can match the wrong pattern (fix the
    // patterns) or match nothing at all (the whole approach lacks reach).
    #[allow(clippy::cast_precision_loss)]
    let n = corpus.len() as f64;
    println!("\n── why it is wrong ──────────────────────────────────────────────");
    #[allow(clippy::cast_precision_loss)]
    let no_signal_pct = no_signal as f64 / n * 100.0;
    #[allow(clippy::cast_precision_loss)]
    let wrong_pct = wrong_with_signal as f64 / n * 100.0;
    println!(
        "  no keyword matched at all (-> Disorder): {no_signal}/{} ({no_signal_pct:.0}%)",
        corpus.len()
    );
    println!(
        "  matched, but the wrong domain:          {wrong_with_signal}/{} ({wrong_pct:.0}%)",
        corpus.len()
    );
    println!("  These call for opposite remedies. Overwhelmingly the first means the");
    println!("  keyword approach has no reach on natural phrasing, and no amount of");
    println!("  list editing fixes that — editing lists against THIS corpus would");
    println!("  only overfit to it.");

    // ---------------------------------------------------------- per vertical
    println!("\n── per vertical ─────────────────────────────────────────────────");
    for vertical in Vertical::all() {
        let mut m = ClassifierMetrics::new();
        for entry in queries_for_vertical(vertical) {
            if let Ok(result) = classifier.classify(entry.text).await {
                m.record(result.domain, entry.domain);
            }
        }
        println!("  {:<12} accuracy {:.3}", vertical.label(), m.accuracy());
    }

    // ------------------------------------------------- the asymmetric errors
    println!("\n── the errors that matter ───────────────────────────────────────");
    let mut chaotic_missed = 0usize;
    let mut chaotic_total = 0usize;
    for entry in corpus.iter().filter(|e| e.domain == CynefinDomain::Chaotic) {
        chaotic_total += 1;
        if let Ok(result) = classifier.classify(entry.text).await {
            if result.domain != CynefinDomain::Chaotic {
                chaotic_missed += 1;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let miss_rate = chaotic_missed as f64 / chaotic_total as f64;
    println!(
        "  Chaotic queries not routed to Chaotic:  {chaotic_missed}/{chaotic_total} ({:.0}%)",
        miss_rate * 100.0
    );
    println!("  A live incident answered from cache is the failure mode that costs most.");

    // ------------------------------------------------------------ abstention
    println!("\n── abstention on ambiguous input ────────────────────────────────");
    let ambiguous = ambiguous_queries();
    let mut abstained = 0usize;
    let mut ambiguous_entropy = 0.0;
    for text in &ambiguous {
        if let Ok(result) = classifier.classify(text).await {
            if result.domain == CynefinDomain::Disorder || result.confidence <= 0.0 {
                abstained += 1;
            }
            ambiguous_entropy += result.entropy;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let n_amb = ambiguous.len() as f64;
    #[allow(clippy::cast_precision_loss)]
    let mean_labelled_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;

    println!(
        "  abstained (Disorder or zero confidence): {abstained}/{} ({:.0}%)",
        ambiguous.len(),
        abstained as f64 / n_amb * 100.0
    );
    println!(
        "  mean entropy, ambiguous queries:  {:.3}",
        ambiguous_entropy / n_amb
    );
    println!("  mean entropy, labelled queries:   {mean_labelled_entropy:.3}");
    println!("  A classifier that cannot separate these two numbers cannot be trusted to");
    println!("  escalate, because escalation is triggered by exactly that gap.");

    lexical_section();
}

/// The lexical classifier, cross-validated against the same corpus.
///
/// Printed after the keyword numbers so the comparison is unavoidable rather
/// than something a reader has to go looking for.
fn lexical_section() {
    use cynepic_router::LexicalClassifier;
    use cynepic_testkit::corpus::routing_corpus;

    const DOMAINS: [CynefinDomain; 4] = [
        CynefinDomain::Clear,
        CynefinDomain::Complicated,
        CynefinDomain::Complex,
        CynefinDomain::Chaotic,
    ];

    println!("\n── LexicalClassifier, 4-fold cross-validated ────────────────────");
    println!("  Train on three quarters of the corpus, score the quarter never");
    println!("  seen, rotate, pool. No fold can be contaminated by its own");
    println!("  training data — which the shipped exemplars cannot promise,");
    println!("  because whoever wrote them had read this corpus.\n");

    let corpus = routing_corpus();
    let k = 4;
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
        let model = LexicalClassifier::train(&train).expect("folds cover every domain");
        for (text, actual) in test {
            metrics.record(model.classify_sync(text).domain, actual);
        }
    }

    println!(
        "  {:<14} {:>10} {:>10} {:>10}",
        "domain", "precision", "recall", "F1"
    );
    for d in DOMAINS {
        println!(
            "  {:<14} {:>10.3} {:>10.3} {:>10.3}",
            format!("{d:?}"),
            metrics.precision(d),
            metrics.recall(d),
            metrics.f1(d)
        );
    }
    let macro_f1 = DOMAINS.iter().map(|d| metrics.f1(*d)).sum::<f64>() / 4.0;

    let shipped = LexicalClassifier::with_default_exemplars().expect("ships trained");
    let silent = corpus
        .iter()
        .filter(|q| shipped.classify_sync(q.text).domain == CynefinDomain::Disorder)
        .count();
    #[allow(clippy::cast_precision_loss)]
    let silent_pct = 100.0 * silent as f64 / corpus.len() as f64;

    println!(
        "\n  {:<26} {:>9} {:>9} {:>9}",
        "", "keyword", "lexical", "R1 bar"
    );
    println!(
        "  {:<26} {:>9.3} {:>9.3} {:>9.2}",
        "macro F1", 0.290, macro_f1, 0.70
    );
    println!(
        "  {:<26} {:>9.3} {:>9.3} {:>9.2}",
        "Chaotic recall",
        0.000,
        metrics.recall(CynefinDomain::Chaotic),
        0.80
    );
    println!(
        "  {:<26} {:>8.0}% {:>8.0}% {:>9}",
        "queries with no signal", 78.0, silent_pct, "-"
    );

    println!("\n  The bar is unmet, so R1 stays open and the ratchet is unchanged.");
    println!("  0.290 -> {macro_f1:.3} against a 0.25 random baseline, and 0.000 ->");
    println!(
        "  {:.3} on the class where a wrong answer costs the most, is most of",
        metrics.recall(CynefinDomain::Chaotic)
    );
    println!("  the distance. What is left is what an embedding model is for.");

    println!("\n  Why this is not an embedding model yet: no model file, no");
    println!("  inference runtime, no new dependency, wasm32-wasip1 still builds,");
    println!("  results are bit-identical across platforms, and `explain` names");
    println!("  the terms behind a verdict. Establishing what the cheap approach");
    println!("  is worth is what gives a later embedding claim a number to beat.");
}
