//! RT1 — every router implementation on one spine, under one protocol.
//!
//! RT1 asks whether the published per-implementation router figures disagree by
//! more than presentation, with the falsifier: *rank order and magnitudes
//! reproduce the published figures → the numbers were always comparable.*
//!
//! THE PROTOCOL DIFFERENCE IS NOT A DETAIL, IT IS THE ANSWER
//! =========================================================
//!
//! The two corpora these implementations were measured on disagree about
//! whether `Disorder` is a thing a query can BE.
//!
//! * `cynepic-testkit::corpus` — 96 queries, four labels. Its own docstring:
//!   *"Disorder is deliberately absent as a label: it is the state of not
//!   knowing which domain applies, so it is a classifier output for ambiguous
//!   input, never a ground-truth property of a query."*
//!
//! * CARF's `test_set.jsonl` — 456 queries, five labels, of which **102 (22%)
//!   are Disorder**.
//!
//! So an abstention — which every classifier here emits as `Disorder` when
//! evidence is thin — is a CORRECT answer on 22% of one corpus and a wrong
//! answer on 100% of the other. A single implementation can therefore post two
//! very different macro F1 scores without changing a line, and comparing those
//! two numbers is comparing two questions.
//!
//! This example runs both readings over the same 456 items so the size of that
//! effect is a measured quantity rather than an argument:
//!
//!     five_class   all 456 items, Disorder is a class to be predicted
//!     four_class   the 354 non-Disorder items, Disorder counted as an error
//!
//! One protocol for everything else: identical items, identical folds, and the
//! trainable classifiers cross-validated 4-fold so no implementation is scored
//! on data it was fitted to. `KeywordClassifier` needs no training and sees
//! every item as held-out, which flatters it if anything.
//!
//!     cargo run -p cynepic-router --example rt1_spine -- <test_set.jsonl>

use std::collections::BTreeMap;
use std::env;
use std::fs;

use cynepic_core::CynefinDomain;
use cynepic_router::discriminative::DiscriminativeClassifier;
use cynepic_router::eval::ClassifierMetrics;
use cynepic_router::lexical::TrainingOptions;
use cynepic_router::{KeywordClassifier, LexicalClassifier, QueryClassifier};
use serde_json::{Value, json};

const FOLDS: usize = 4;

fn parse_domain(s: &str) -> Option<CynefinDomain> {
    match s.to_ascii_lowercase().as_str() {
        "clear" => Some(CynefinDomain::Clear),
        "complicated" => Some(CynefinDomain::Complicated),
        "complex" => Some(CynefinDomain::Complex),
        "chaotic" => Some(CynefinDomain::Chaotic),
        "disorder" => Some(CynefinDomain::Disorder),
        _ => None,
    }
}

fn load(path: &str) -> Result<Vec<(String, CynefinDomain)>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        let q = v["query"].as_str().ok_or("no query")?.to_string();
        let d = v["domain"].as_str().ok_or("no domain")?;
        let d = parse_domain(d).ok_or_else(|| format!("line {}: unknown domain {d}", i + 1))?;
        out.push((q, d));
    }
    Ok(out)
}

/// Fold assignment by position WITHIN each domain, so every fold is balanced
/// across labels. Positional rather than random: the result must be identical
/// on every machine, and a shuffle buys nothing at this size.
type Labelled<'a> = Vec<(&'a str, CynefinDomain)>;

fn split(items: &[(String, CynefinDomain)], fold: usize) -> (Labelled<'_>, Labelled<'_>) {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (q, d) in items {
        let key = format!("{d:?}");
        let idx = seen.entry(key).or_insert(0);
        let pos = *idx;
        *idx += 1;
        if pos % FOLDS == fold {
            test.push((q.as_str(), *d));
        } else {
            train.push((q.as_str(), *d));
        }
    }
    (train, test)
}

fn macro_f1(m: &ClassifierMetrics, labels: &[CynefinDomain]) -> f64 {
    labels.iter().map(|d| m.f1(*d)).sum::<f64>() / labels.len() as f64
}

/// Score every implementation over one item set, under one protocol.
fn score(items: &[(String, CynefinDomain)], labels: &[CynefinDomain], protocol: &str) -> Value {
    let keyword = KeywordClassifier::default_patterns();
    // `KeywordClassifier` exposes only the async trait method, and it does no
    // I/O, so a current-thread runtime is enough to drive it.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime");
    let mut m_keyword = ClassifierMetrics::new();
    let mut m_lexical = ClassifierMetrics::new();
    let mut m_discrim = ClassifierMetrics::new();

    for fold in 0..FOLDS {
        let (train, test) = split(items, fold);

        let lex = LexicalClassifier::train_with(&train, TrainingOptions::default()).ok();
        let dis = DiscriminativeClassifier::train(&train).ok();

        for (q, actual) in &test {
            // The keyword classifier is untrained, so every item is held out
            // for it. That flatters it relative to the other two, and it still
            // loses, which is the direction that makes the comparison safe.
            let kw = rt
                .block_on(keyword.classify(q))
                .map(|r| r.domain)
                // A classifier that errors has not abstained; record it as
                // Disorder so the denominator stays whole and the failure shows
                // up as a wrong answer rather than as a missing row.
                .unwrap_or(CynefinDomain::Disorder);
            m_keyword.record(kw, *actual);
            if let Some(l) = &lex {
                m_lexical.record(l.classify_sync(q).domain, *actual);
            }
            if let Some(d) = &dis {
                m_discrim.record(d.classify_sync(q).domain, *actual);
            }
        }
    }

    let rows: Vec<Value> = [
        ("keyword", &m_keyword),
        ("lexical_centroid", &m_lexical),
        ("discriminative_logreg", &m_discrim),
    ]
    .iter()
    .map(|(name, m)| {
        json!({
            "implementation": name,
            "macro_f1": macro_f1(m, labels),
            "accuracy": m.accuracy(),
            "per_domain_f1": labels.iter().map(|d| json!({
                "domain": format!("{d:?}"),
                "f1": m.f1(*d),
                "recall": m.recall(*d),
                "precision": m.precision(*d),
            })).collect::<Vec<_>>(),
            "n_scored": m.total(),
        })
    })
    .collect();

    json!({
        "protocol": protocol,
        "n_items": items.len(),
        "labels": labels.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
        "results": rows,
    })
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: rt1_spine <test_set.jsonl>");
        std::process::exit(2);
    });
    let all = match load(&path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let five: Vec<CynefinDomain> = vec![
        CynefinDomain::Clear,
        CynefinDomain::Complicated,
        CynefinDomain::Complex,
        CynefinDomain::Chaotic,
        CynefinDomain::Disorder,
    ];
    let four: Vec<CynefinDomain> = five[..4].to_vec();

    let no_disorder: Vec<(String, CynefinDomain)> = all
        .iter()
        .filter(|(_, d)| *d != CynefinDomain::Disorder)
        .cloned()
        .collect();

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_, d) in &all {
        *counts.entry(format!("{d:?}")).or_insert(0) += 1;
    }

    let report = json!({
        "benchmark": "rt1_spine",
        "hypothesis": "RT1",
        "spine": path,
        "label_counts": counts,
        "folds": FOLDS,
        "five_class": score(&all, &five, "all 456 items; Disorder is a class to predict"),
        "four_class": score(&no_disorder, &four, "354 non-Disorder items; an abstention is an error"),
        "why_two_protocols": "cynepic-testkit's corpus excludes Disorder as a label by \
             design; CARF's spine makes it 22% of the ground truth. An abstention is \
             therefore a correct answer on one corpus and a wrong answer on the other, \
             so one implementation can post two very different scores without changing.",
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
