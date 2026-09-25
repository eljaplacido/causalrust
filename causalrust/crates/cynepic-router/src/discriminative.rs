//! A discriminative lexical classifier — the second half of the answer to R1.
//!
//! # Why a second classifier
//!
//! [`LexicalClassifier`](crate::LexicalClassifier) took R1 from macro F1 0.290
//! to **0.656** and Chaotic recall from 0.000 to **0.625** under cross
//! validation. Both are short of R1's bar (0.70 / 0.80), and the error analysis
//! says where the remainder sits: **Complicated** is the weakest class, and it
//! leaks to Disorder, Complex, Clear and Chaotic in roughly equal measure.
//!
//! That leak is a property of the decision rule rather than of the features. A
//! nearest-centroid model asks *which domain does this query most resemble*,
//! and a diagnostic question resembles every domain a little: it shares
//! `latency`, `column`, `agent` and `release` with all of them. What separates
//! Complicated is not the presence of those terms but the presence of *why*,
//! *which step*, *what is driving* — terms that are individually weak and only
//! decisive in combination with the absence of others. A centroid cannot
//! express "this term counts against Chaotic"; it has no negative weights.
//!
//! This module keeps the feature pipeline **byte-identical** to
//! [`LexicalClassifier`] — the same tokenizer, the same smoothed idf, the same
//! optional concentration scaling, the same L2 normalisation — and changes only
//! the rule that turns a vector into a verdict: multinomial logistic
//! regression, fitted by gradient descent. So a difference in the score is a
//! difference in the decision rule and not in what the model was shown, which
//! is the comparison finding R1 actually needs and the one
//! `tests/discriminative_accuracy.rs` reports.
//!
//! # Why not a neural embedding model, again
//!
//! The same three reasons as [`LexicalClassifier`]: no model file, no download,
//! no inference runtime; deterministic across platforms, which a component that
//! writes to an audit trail needs; and inspectable, because
//! [`DiscriminativeClassifier::explain`] can name the terms that *pushed toward*
//! a domain separately from those that pushed away — something the centroid
//! model cannot do at all.
//!
//! # Honesty about hyperparameters
//!
//! Choosing the regularisation strength by reading the cross-validated score
//! off this corpus would contaminate the headline exactly the way reading the
//! corpus before writing exemplars contaminated
//! [`LexicalClassifier::default_exemplars`]. So it is not chosen that way.
//!
//! [`DiscriminativeClassifier::train`] performs **nested selection**: within the
//! examples it is given, and never seeing the outer test fold, it runs an inner
//! 3-fold split to pick `l2` from a fixed grid, then refits on everything it
//! was given. When the outer harness rotates folds, the inner selection rotates
//! with it, so no fold's test items influence the model that scores them. The
//! sweep in the test file is reported as a *diagnostic*, not as the basis of
//! the default.
//!
//! # Determinism
//!
//! Weights initialise at zero, the epoch count and learning rate are fixed, and
//! the term ordering is derived from a `BTreeMap` rather than a `HashMap`, so
//! the fitted model is identical on every run and every platform. There is no
//! random number generator anywhere in this module.

use std::collections::{BTreeMap, HashMap};

use async_trait::async_trait;
use cynepic_core::CynefinDomain;

use crate::classifier::{ClassificationResult, ClassifierError, QueryClassifier};
use crate::lexical::{TrainingOptions, Weighting, tokenize_with};

/// Minimum evidence, in units of "one maximally specific term".
///
/// Carried over unchanged from [`LexicalClassifier`](crate::LexicalClassifier)
/// so that abstention behaviour is held constant across the two models. A
/// classifier that abstained less would score higher on recall and worse on the
/// ambiguous set, and the comparison would then be measuring the threshold
/// rather than the decision rule.
const MIN_EVIDENCE: f64 = 1.5;

/// Regularisation strengths the inner selection chooses between.
///
/// Four points an order of magnitude apart. A finer grid would be selection
/// noise at 48–72 training examples.
const L2_GRID: [f64; 4] = [1e-4, 1e-3, 1e-2, 1e-1];

/// Full-batch gradient descent settings.
///
/// Fixed rather than tuned: with L2-normalised tf-idf features every gradient
/// component is bounded by 1, so a constant step of 0.5 is stable, and the loss
/// is flat well before 500 epochs on a corpus this size. Convergence is
/// asserted by `weights_have_converged` in the test file rather than assumed.
const EPOCHS: usize = 500;
/// Learning rate for [`EPOCHS`] steps of full-batch gradient descent.
const LEARNING_RATE: f64 = 0.5;

/// A tf-idf multinomial logistic-regression classifier over unigrams and bigrams.
///
/// Shares [`LexicalClassifier`](crate::LexicalClassifier)'s feature pipeline and
/// replaces its nearest-centroid rule with fitted per-term, per-domain weights
/// that may be negative.
#[derive(Debug, Clone)]
pub struct DiscriminativeClassifier {
    /// Term -> column index. Ordered, so fitting is reproducible.
    vocabulary: BTreeMap<String, usize>,
    /// Inverse document frequency per column, concentration-scaled if asked.
    idf: Vec<f64>,
    /// Largest idf in the model; the unit `min_evidence` is expressed in.
    max_idf: f64,
    /// `domains.len() * (vocabulary.len() + 1)` weights, bias last per class.
    weights: Vec<f64>,
    /// Domains in a fixed order, matching the rows of `weights`.
    domains: Vec<CynefinDomain>,
    /// Feature options, kept so `classify` tokenizes as `train` did.
    options: TrainingOptions,
    /// Evidence below which the classifier declines to answer.
    min_evidence: f64,
    /// The `l2` the inner selection chose, kept for reporting.
    selected_l2: f64,
}

impl DiscriminativeClassifier {
    /// Fit on labelled examples, selecting `l2` by an inner 3-fold split.
    ///
    /// The inner split never sees anything outside `examples`, so when an outer
    /// cross-validation harness holds a fold out, that fold cannot influence
    /// this model.
    ///
    /// # Errors
    ///
    /// [`ClassifierError::Untrainable`] if `examples` is empty or covers fewer
    /// than two domains — a classifier that can only answer one way is a
    /// malformed classifier, not a confident one, which is the distinction
    /// finding C12 turned on.
    pub fn train(examples: &[(&str, CynefinDomain)]) -> Result<Self, ClassifierError> {
        Self::train_with(examples, TrainingOptions::default())
    }

    /// As [`train`](Self::train), with the feature options overridden.
    ///
    /// # Errors
    ///
    /// As [`train`](Self::train).
    pub fn train_with(
        examples: &[(&str, CynefinDomain)],
        options: TrainingOptions,
    ) -> Result<Self, ClassifierError> {
        let l2 = Self::select_l2(examples, options)?;
        Self::fit(examples, options, l2)
    }

    /// Fit at a fixed regularisation strength, skipping the inner selection.
    ///
    /// Exposed for the sweep diagnostic. Prefer [`train`](Self::train): a model
    /// fitted at an `l2` chosen by looking at the score it is about to be
    /// judged on is not a held-out measurement.
    ///
    /// # Errors
    ///
    /// As [`train`](Self::train).
    pub fn fit_at(
        examples: &[(&str, CynefinDomain)],
        options: TrainingOptions,
        l2: f64,
    ) -> Result<Self, ClassifierError> {
        Self::fit(examples, options, l2)
    }

    /// Which `l2` the inner selection chose.
    #[must_use]
    pub fn selected_l2(&self) -> f64 {
        self.selected_l2
    }

    /// Replace the abstention threshold.
    #[must_use]
    pub fn with_min_evidence(mut self, threshold: f64) -> Self {
        self.min_evidence = threshold;
        self
    }

    /// The abstention threshold in force.
    #[must_use]
    pub fn min_evidence(&self) -> f64 {
        self.min_evidence
    }

    /// Evidence mass and peak term weight for a query, in `max_idf` units.
    ///
    /// Identical in definition to
    /// [`LexicalClassifier::evidence`](crate::LexicalClassifier::evidence), so
    /// the two models abstain on exactly the same queries.
    #[must_use]
    pub fn evidence(&self, query: &str) -> (f64, f64) {
        let mut mass = 0.0;
        let mut peak: f64 = 0.0;
        for t in tokenize_with(query, self.options.stem) {
            if let Some(&idx) = self.vocabulary.get(&t) {
                let w = self.idf.get(idx).copied().unwrap_or(0.0);
                mass += w;
                peak = peak.max(w);
            }
        }
        (mass / self.max_idf, peak / self.max_idf)
    }

    /// The terms that moved the verdict, most influential first.
    ///
    /// Unlike the centroid model's version, the contribution is signed: a term
    /// may argue *against* the domain that won. `limit` bounds the list.
    #[must_use]
    pub fn explain(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        let Some(features) = self.features(query) else {
            return Vec::new();
        };
        let result = self.classify_sync(query);
        let Some(class) = self.domains.iter().position(|d| *d == result.domain) else {
            return Vec::new();
        };
        let stride = self.vocabulary.len() + 1;
        let mut named: Vec<(String, f64)> = features
            .iter()
            .filter_map(|(idx, value)| {
                let w = self.weights.get(class * stride + idx)?;
                let term = self
                    .vocabulary
                    .iter()
                    .find(|(_, column)| *column == idx)
                    .map(|(t, _)| t.clone())?;
                Some((term, w * value))
            })
            .collect();
        named.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        named.truncate(limit);
        named
    }

    /// Classify without the async wrapper.
    #[must_use]
    pub fn classify_sync(&self, query: &str) -> ClassificationResult {
        let scores = self.softmax_scores(query);
        let (mass, _) = self.evidence(query);
        if mass < self.min_evidence {
            return ClassificationResult::abstained(scores);
        }
        ClassificationResult::from_scores(scores)
    }

    // ── internals ──

    /// Softmax probability per domain, in the crate's score-pair form.
    fn softmax_scores(&self, query: &str) -> Vec<(CynefinDomain, f64)> {
        let Some(features) = self.features(query) else {
            return self.domains.iter().map(|d| (*d, 0.0)).collect();
        };
        let logits = self.logits(&features);
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = logits.iter().map(|z| (z - max).exp()).collect();
        let total: f64 = exp.iter().sum();
        self.domains
            .iter()
            .zip(exp)
            .map(|(d, e)| (*d, if total > 0.0 { e / total } else { 0.0 }))
            .collect()
    }

    /// One logit per domain for an already-extracted feature vector.
    fn logits(&self, features: &[(usize, f64)]) -> Vec<f64> {
        let stride = self.vocabulary.len() + 1;
        (0..self.domains.len())
            .map(|k| {
                let base = k * stride;
                let mut z = self.weights[base + stride - 1]; // bias
                for (idx, value) in features {
                    z += self.weights[base + idx] * value;
                }
                z
            })
            .collect()
    }

    /// L2-normalised tf-idf features for a query, or `None` if no term is known.
    ///
    /// Normalising is what makes a long query and a short one comparable; query
    /// length is an artefact of how someone typed it, not evidence about domain.
    fn features(&self, query: &str) -> Option<Vec<(usize, f64)>> {
        let mut counts: BTreeMap<usize, f64> = BTreeMap::new();
        for t in tokenize_with(query, self.options.stem) {
            if let Some(&idx) = self.vocabulary.get(&t) {
                *counts.entry(idx).or_insert(0.0) += 1.0;
            }
        }
        if counts.is_empty() {
            return None;
        }
        let mut v: Vec<(usize, f64)> = counts
            .into_iter()
            .map(|(idx, tf)| (idx, tf * self.idf[idx]))
            .collect();
        let norm: f64 = v.iter().map(|(_, x)| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for (_, x) in &mut v {
                *x /= norm;
            }
        }
        Some(v)
    }

    /// Pick `l2` by an inner 3-fold split over `examples` alone.
    fn select_l2(
        examples: &[(&str, CynefinDomain)],
        options: TrainingOptions,
    ) -> Result<f64, ClassifierError> {
        const INNER_K: usize = 3;
        let mut best = (L2_GRID[0], f64::NEG_INFINITY);

        for l2 in L2_GRID {
            let mut correct = 0_usize;
            let mut total = 0_usize;
            for fold in 0..INNER_K {
                let (train, test) = split_by_domain(examples, INNER_K, fold);
                // A degenerate inner fold is skipped rather than failed: the
                // outer call still has a model to return, and skipping is
                // recorded by the score simply having fewer items.
                let Ok(model) = Self::fit(&train, options, l2) else {
                    continue;
                };
                for (text, actual) in test {
                    total += 1;
                    if model.classify_sync(text).domain == actual {
                        correct += 1;
                    }
                }
            }
            if total == 0 {
                continue;
            }
            #[allow(clippy::cast_precision_loss)]
            let score = correct as f64 / total as f64;
            if score > best.1 {
                best = (l2, score);
            }
        }

        if best.1.is_finite() {
            Ok(best.0)
        } else {
            // Nothing could be evaluated — fall back to the middle of the grid
            // rather than refusing, and let `fit` report a real Untrainable.
            Ok(L2_GRID[1])
        }
    }

    /// Build the vocabulary, the idf and the fitted weights.
    fn fit(
        examples: &[(&str, CynefinDomain)],
        options: TrainingOptions,
        l2: f64,
    ) -> Result<Self, ClassifierError> {
        if examples.is_empty() {
            return Err(ClassifierError::Untrainable(
                "cannot train on an empty example set".into(),
            ));
        }

        let tokenized: Vec<(Vec<String>, CynefinDomain)> = examples
            .iter()
            .map(|(text, domain)| (tokenize_with(text, options.stem), *domain))
            .collect();

        let mut domains: Vec<CynefinDomain> = Vec::new();
        for (_, d) in &tokenized {
            if !domains.contains(d) {
                domains.push(*d);
            }
        }
        if domains.len() < 2 {
            return Err(ClassifierError::Untrainable(format!(
                "training examples cover only {} domain(s); a classifier needs at least 2",
                domains.len()
            )));
        }
        domains.sort_by_key(|d| format!("{d:?}"));

        // Vocabulary in term order, so column indices depend on the terms
        // rather than on hash iteration order — which is what makes a fitted
        // model byte-identical across runs and platforms.
        let mut terms_seen: BTreeMap<String, ()> = BTreeMap::new();
        for (terms, _) in &tokenized {
            for t in terms {
                terms_seen.insert(t.clone(), ());
            }
        }
        let vocabulary: BTreeMap<String, usize> = terms_seen
            .into_keys()
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect();

        let width = vocabulary.len();
        let mut doc_freq = vec![0_usize; width];
        for (terms, _) in &tokenized {
            let mut seen = vec![false; width];
            for t in terms {
                if let Some(&idx) = vocabulary.get(t) {
                    if !seen[idx] {
                        seen[idx] = true;
                        doc_freq[idx] += 1;
                    }
                }
            }
        }

        // Smoothed idf, identical to `LexicalClassifier`'s.
        #[allow(clippy::cast_precision_loss)]
        let n_docs = tokenized.len() as f64;
        let idf: Vec<f64> = doc_freq
            .iter()
            .map(|df| {
                #[allow(clippy::cast_precision_loss)]
                let df = *df as f64;
                ((n_docs + 1.0) / (df + 1.0)).ln() + 1.0
            })
            .collect();

        let idf = match options.weighting {
            Weighting::Idf => idf,
            Weighting::IdfTimesConcentration => {
                let mut per_domain: Vec<Vec<f64>> = vec![vec![0.0; width]; domains.len()];
                for (terms, domain) in &tokenized {
                    let Some(d) = domains.iter().position(|x| x == domain) else {
                        continue;
                    };
                    for t in terms {
                        if let Some(&idx) = vocabulary.get(t) {
                            per_domain[d][idx] += 1.0;
                        }
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                let ln_k = (domains.len() as f64).ln();
                idf.iter()
                    .enumerate()
                    .map(|(idx, w)| {
                        let total: f64 = per_domain.iter().map(|c| c[idx]).sum();
                        if total <= 0.0 || ln_k <= 0.0 {
                            return *w;
                        }
                        let entropy: f64 = per_domain
                            .iter()
                            .map(|c| {
                                let p = c[idx] / total;
                                if p > 0.0 { -p * p.ln() } else { 0.0 }
                            })
                            .sum();
                        let concentration = (1.0 - entropy / ln_k).clamp(0.0, 1.0);
                        w * (1.0 + concentration)
                    })
                    .collect()
            }
        };

        let max_idf = idf
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(f64::EPSILON);

        // Design matrix, sparse and L2-normalised per row.
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(tokenized.len());
        let mut labels: Vec<usize> = Vec::with_capacity(tokenized.len());
        for (terms, domain) in &tokenized {
            let mut counts: BTreeMap<usize, f64> = BTreeMap::new();
            for t in terms {
                if let Some(&idx) = vocabulary.get(t) {
                    *counts.entry(idx).or_insert(0.0) += 1.0;
                }
            }
            let mut v: Vec<(usize, f64)> = counts
                .into_iter()
                .map(|(idx, tf)| (idx, tf * idf[idx]))
                .collect();
            let norm: f64 = v.iter().map(|(_, x)| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for (_, x) in &mut v {
                    *x /= norm;
                }
            }
            let Some(k) = domains.iter().position(|d| d == domain) else {
                continue;
            };
            rows.push(v);
            labels.push(k);
        }

        let n_classes = domains.len();
        let stride = width + 1;
        let mut weights = vec![0.0_f64; n_classes * stride];

        #[allow(clippy::cast_precision_loss)]
        let n = rows.len() as f64;
        for _ in 0..EPOCHS {
            let mut grad = vec![0.0_f64; n_classes * stride];
            for (row, &label) in rows.iter().zip(&labels) {
                // Logits for this row.
                let mut z = vec![0.0_f64; n_classes];
                for (k, zk) in z.iter_mut().enumerate() {
                    let base = k * stride;
                    let mut acc = weights[base + stride - 1];
                    for (idx, value) in row {
                        acc += weights[base + idx] * value;
                    }
                    *zk = acc;
                }
                let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = z.iter().map(|x| (x - max).exp()).collect();
                let total: f64 = exp.iter().sum();
                for (k, e) in exp.iter().enumerate() {
                    let p = if total > 0.0 { e / total } else { 0.0 };
                    let err = p - if k == label { 1.0 } else { 0.0 };
                    let base = k * stride;
                    for (idx, value) in row {
                        grad[base + idx] += err * value;
                    }
                    grad[base + stride - 1] += err;
                }
            }
            // Mean gradient, plus L2 on the weights but never on the bias:
            // penalising the intercept would shrink the class priors toward
            // uniform, which is a different model than the one intended.
            for k in 0..n_classes {
                let base = k * stride;
                for j in 0..width {
                    let g = grad[base + j] / n + l2 * weights[base + j];
                    weights[base + j] -= LEARNING_RATE * g;
                }
                let gb = grad[base + stride - 1] / n;
                weights[base + stride - 1] -= LEARNING_RATE * gb;
            }
        }

        Ok(Self {
            vocabulary,
            idf,
            max_idf,
            weights,
            domains,
            options,
            min_evidence: MIN_EVIDENCE,
            selected_l2: l2,
        })
    }
}

/// One side of a fold: labelled examples to train on, or to score against.
type LabelledSet<'a> = Vec<(&'a str, CynefinDomain)>;

/// Split `examples` into train/test for one fold, balanced per domain.
///
/// Position within a domain decides the fold (`i % k`), the same rule the outer
/// harness uses, so no random number generator is involved and the split is
/// identical on every platform.
fn split_by_domain<'a>(
    examples: &[(&'a str, CynefinDomain)],
    k: usize,
    fold: usize,
) -> (LabelledSet<'a>, LabelledSet<'a>) {
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (text, domain) in examples {
        let key = format!("{domain:?}");
        let idx = seen.entry(key).or_insert(0);
        let position = *idx;
        *idx += 1;
        if position % k == fold {
            test.push((*text, *domain));
        } else {
            train.push((*text, *domain));
        }
    }
    (train, test)
}

#[async_trait]
impl QueryClassifier for DiscriminativeClassifier {
    async fn classify(&self, query: &str) -> Result<ClassificationResult, ClassifierError> {
        Ok(self.classify_sync(query))
    }
}
