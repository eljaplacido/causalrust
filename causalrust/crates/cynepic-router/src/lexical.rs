//! A trainable lexical classifier, and the answer to finding R1.
//!
//! # Why this exists
//!
//! [`KeywordClassifier`](crate::KeywordClassifier) scores macro F1 0.290
//! against a four-class random baseline of 0.25, with **0.000 recall on
//! Chaotic**. The decomposition in `docs/FINDINGS.md` says what kind of failure
//! that is:
//!
//! ```text
//! no keyword matched at all (-> Disorder):  75/96  (78%)
//! matched, but the wrong domain:             2/96  ( 2%)
//! ```
//!
//! 78% no-signal is a **reach** problem, not a tuning problem. "The agent is
//! calling the payments tool in a loop right now" is unmistakably a Chaotic
//! query and contains none of `emergency`, `crisis`, `outage`, `breach`,
//! `urgent` or `critical failure`. Adding those words to a list would not help;
//! the next phrasing would miss too.
//!
//! # What it does instead
//!
//! Every term in the query contributes, weighted by how much it distinguishes
//! one domain from another, rather than a handful of terms deciding and the
//! rest being discarded. Concretely: tf-idf over unigrams and bigrams, one
//! centroid per domain, cosine similarity.
//!
//! That is a deliberately unfashionable choice, so it is worth saying why it is
//! not a neural embedding model:
//!
//! * **No model file, no download, no inference runtime.** The crate stays
//!   `forbid(unsafe_code)`, builds for `wasm32-wasip1`, and adds no
//!   dependencies.
//! * **Deterministic.** The same query gives the same route on every platform,
//!   which matters for a component whose decisions are written to an audit
//!   trail.
//! * **Inspectable.** [`LexicalClassifier::explain`] names the terms that drove
//!   a verdict. "Why did this route to Chaotic" has an answer.
//!
//! An embedding model would very likely score higher, and the roadmap still
//! wants one. This establishes what the cheap approach is worth first, so that
//! a later claim about embeddings has a real number to beat rather than 0.290.
//!
//! # Cynefin is about grammar more than topic
//!
//! The domain of a question is a property of the relationship between cause and
//! effect, not of its subject matter. "The pipeline is failing" could be any
//! domain. What separates them is closer to mood and tense:
//!
//! | domain | the shape of the question |
//! |---|---|
//! | Clear | interrogative lookup — *what is*, *list*, *show me*, *how many* |
//! | Complicated | diagnostic — *why did*, *root cause*, *analyse*, *explain* |
//! | Complex | subjunctive — *what if*, *should we try*, *experiment*, *might* |
//! | Chaotic | imperative, present continuous — *stop*, *halt*, *right now*, *is …ing* |
//!
//! Bigrams matter more than unigrams for exactly this reason: `right now` and
//! `what if` carry the signal that `now` and `what` do not.
//!
//! # Honesty about the measurement
//!
//! The default exemplars in [`LexicalClassifier::default_exemplars`] were
//! written by someone who had read the evaluation corpus. That cannot be
//! un-read, so a score of those exemplars against that corpus is **not** a
//! clean held-out number, and it is not the one to quote.
//!
//! The headline is instead **k-fold cross-validation on the corpus itself** —
//! train on three quarters, score the quarter never seen, rotate. That measures
//! the *method* rather than the exemplar list, and no fold can be contaminated
//! by its own training data. Both numbers are reported by
//! `cargo run -p cynepic-router --example classifier_report`.

use std::collections::HashMap;

use async_trait::async_trait;
use cynepic_core::CynefinDomain;

use crate::classifier::{ClassificationResult, ClassifierError, QueryClassifier};

/// Cosine similarity below which the classifier declines to answer.
///
/// A weak direction is still a direction, so this is a floor rather than the
/// main rule; [`MIN_EVIDENCE`] does most of the work.
const MIN_SIMILARITY: f64 = 0.10;

/// Minimum gap between the best and second-best domain, relative to the best.
///
/// Kept only for genuine ties. It does most of the work in a keyword scheme and
/// almost none here: measured, ambiguous queries have a *median margin of
/// 0.328*, higher than one might expect, because when a single term matches,
/// one domain wins it outright. Margin measures decisiveness, not evidence.
const MIN_RELATIVE_MARGIN: f64 = 0.05;

/// Minimum evidence, in units of "one maximally specific term".
///
/// # The rule that actually separates ambiguity from ignorance
///
/// Cosine similarity is length-normalised, which is what makes it robust to
/// query length and also what makes it blind to *how much* evidence there is:
/// normalising divides the magnitude away. Measured over the two populations
/// (`thresholds_diagnostic` in `tests/lexical_accuracy.rs`, p10/p50/p90):
///
/// ```text
///                        cosine              evidence mass
///   ambiguous       0.00 / 0.09 / 0.19     0.0 /  4.2 / 14.0
///   answerable      0.16 / 0.24 / 0.35     7.7 / 19.5 / 31.2
/// ```
///
/// Cosine overlaps. Mass barely does — "help" and "stop the job, it is
/// corrupting rows as we speak" can point in similar directions, but only one
/// of them has said anything.
///
/// Expressed relative to the largest idf in the model rather than absolutely,
/// so the threshold survives a change in training-set size: idf scales with
/// `ln(n_docs)`, so an absolute cut calibrated on 48 examples would quietly
/// mean something else on 72 — which is exactly what cross-validation does.
const MIN_EVIDENCE: f64 = 1.5;

/// Evidence at which the classifier is treated as fully informed.
///
/// Used to shrink the score distribution toward uniform when evidence is thin —
/// see [`LexicalClassifier::classify_sync`]. Set near the answerable median.
const FULL_EVIDENCE: f64 = 4.0;

/// A tf-idf nearest-centroid classifier over unigrams and bigrams.
#[derive(Debug, Clone)]
pub struct LexicalClassifier {
    /// Term -> column index.
    vocabulary: HashMap<String, usize>,
    /// Inverse document frequency per column.
    idf: Vec<f64>,
    /// The largest idf in the model, i.e. the weight of a term seen once.
    ///
    /// Evidence is measured in units of this so the abstention threshold does
    /// not silently change meaning when the training set changes size.
    max_idf: f64,
    /// L2-normalised centroid per domain.
    centroids: Vec<(CynefinDomain, Vec<f64>)>,
    /// Evidence below which the classifier abstains. See [`MIN_EVIDENCE`].
    ///
    /// Exposed because it is the single knob that trades recall against
    /// abstention, and the right setting depends on what sits downstream: a
    /// router in front of a human reviewer wants it lower than one in front of
    /// an automated action. The default is measured, not guessed — see
    /// `evidence_threshold_sweep` in `tests/lexical_accuracy.rs`.
    min_evidence: f64,
}

/// Lowercase, split on non-alphanumerics, emit unigrams then bigrams.
///
/// Bigrams are not a refinement here, they are most of the point: `what if`,
/// `right now` and `why did` are the terms that carry Cynefin signal, and each
/// is invisible to a unigram model that sees only `what`, `now` and `did`.
fn tokenize(text: &str) -> Vec<String> {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();

    let mut terms = words.clone();
    for pair in words.windows(2) {
        terms.push(format!("{} {}", pair[0], pair[1]));
    }
    terms
}

impl LexicalClassifier {
    /// Train from labelled examples.
    ///
    /// # Errors
    ///
    /// [`ClassifierError::Untrainable`] if the examples are empty or cover
    /// fewer than two domains — a classifier that can only answer one way is
    /// not a classifier, and returning one silently would be the same class of
    /// mistake as finding C12.
    pub fn train(examples: &[(&str, CynefinDomain)]) -> Result<Self, ClassifierError> {
        if examples.is_empty() {
            return Err(ClassifierError::Untrainable(
                "cannot train on an empty example set".into(),
            ));
        }

        let tokenized: Vec<(Vec<String>, CynefinDomain)> = examples
            .iter()
            .map(|(text, domain)| (tokenize(text), *domain))
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

        // Vocabulary and document frequency.
        let mut vocabulary: HashMap<String, usize> = HashMap::new();
        let mut doc_freq: Vec<usize> = Vec::new();
        for (terms, _) in &tokenized {
            let mut seen: Vec<usize> = Vec::new();
            for t in terms {
                let next = vocabulary.len();
                let idx = *vocabulary.entry(t.clone()).or_insert(next);
                if idx == doc_freq.len() {
                    doc_freq.push(0);
                }
                if !seen.contains(&idx) {
                    seen.push(idx);
                    doc_freq[idx] += 1;
                }
            }
        }

        // Smoothed idf. The `+1` inside the log keeps a term appearing in every
        // document at weight zero rather than negative, which would invert its
        // meaning rather than merely ignoring it.
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

        // Centroid per domain: the mean of its examples' normalised vectors.
        // Normalising *before* averaging is what stops a long example from
        // dominating a short one; the length of a query is an artefact of how
        // someone typed it, not evidence about its domain.
        let width = vocabulary.len();
        let mut centroids: Vec<(CynefinDomain, Vec<f64>)> =
            domains.iter().map(|d| (*d, vec![0.0; width])).collect();

        for (terms, domain) in &tokenized {
            let v = Self::vectorize(terms, &vocabulary, &idf, width);
            if let Some((_, c)) = centroids.iter_mut().find(|(d, _)| d == domain) {
                for (slot, value) in c.iter_mut().zip(v.iter()) {
                    *slot += value;
                }
            }
        }
        for (_, c) in &mut centroids {
            normalize(c);
        }

        let max_idf = idf.iter().copied().fold(1.0_f64, f64::max);

        Ok(Self {
            vocabulary,
            idf,
            max_idf,
            centroids,
            min_evidence: MIN_EVIDENCE,
        })
    }

    /// Set the evidence threshold below which the classifier abstains.
    ///
    /// Higher abstains more often and answers more accurately when it does.
    /// The frontier is measured; see [`Self::min_evidence`].
    #[must_use]
    pub fn with_min_evidence(mut self, threshold: f64) -> Self {
        self.min_evidence = threshold;
        self
    }

    /// The current evidence threshold.
    #[must_use]
    pub fn min_evidence(&self) -> f64 {
        self.min_evidence
    }

    /// Build the sparse-in-spirit tf-idf vector for a token list.
    fn vectorize(
        terms: &[String],
        vocabulary: &HashMap<String, usize>,
        idf: &[f64],
        width: usize,
    ) -> Vec<f64> {
        let mut v = vec![0.0; width];
        for t in terms {
            if let Some(&idx) = vocabulary.get(t) {
                // Raw count, scaled by idf. Sub-linear tf scaling is a common
                // refinement and made no measurable difference on queries this
                // short — a term appearing twice in an eight-word question is
                // already rare.
                v[idx] += idf.get(idx).copied().unwrap_or(1.0);
            }
        }
        normalize(&mut v);
        v
    }

    /// How much the classifier actually knows about this query.
    ///
    /// Returns `(mass, peak)`, both in units of the model's largest idf — the
    /// weight of a term seen exactly once in training. A mass of 2.0 means the
    /// query carried about as much evidence as two maximally specific terms.
    ///
    /// Relative units matter: idf scales with `ln(n_docs)`, so an absolute
    /// threshold calibrated on one training-set size means something different
    /// on another.
    ///
    /// # Why cosine alone is not enough
    ///
    /// Cosine similarity is length-normalised. That is what makes it robust to
    /// query length — and it is also what makes it blind to *how much evidence
    /// there is*, because normalising divides the magnitude away. "help" and
    /// "stop the job, it is corrupting rows as we speak" can both land close to
    /// a centroid; only one of them has told us anything.
    ///
    /// `peak` is the sharper of the two. A generic query matches only low-idf
    /// terms — *what*, *is*, *the*, *this* — which appear in every training
    /// example and are therefore worth almost nothing. A specific query
    /// contains at least one term that is rare in training and so carries real
    /// weight. Requiring one such term is what separates "I have no idea what
    /// you are asking" from "I am fairly sure this is a lookup".
    pub fn evidence(&self, query: &str) -> (f64, f64) {
        let mut mass = 0.0;
        let mut peak: f64 = 0.0;
        for t in tokenize(query) {
            if let Some(&idx) = self.vocabulary.get(&t) {
                let w = self.idf.get(idx).copied().unwrap_or(0.0);
                mass += w;
                peak = peak.max(w);
            }
        }
        (mass / self.max_idf, peak / self.max_idf)
    }

    /// Cosine similarity to each domain centroid, highest first.
    fn similarities(&self, query: &str) -> Vec<(CynefinDomain, f64)> {
        let terms = tokenize(query);
        let v = Self::vectorize(&terms, &self.vocabulary, &self.idf, self.vocabulary.len());

        let mut scores: Vec<(CynefinDomain, f64)> = self
            .centroids
            .iter()
            .map(|(d, c)| {
                let dot: f64 = v.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
                (*d, dot.max(0.0))
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// The terms that pushed a query toward its winning domain.
    ///
    /// Returns up to `limit` `(term, contribution)` pairs, largest first. This
    /// is the property a keyword list had for free and an embedding model gives
    /// up: "why did this route to Chaotic" has an answer, and a reviewer can
    /// check whether it is a sensible one.
    pub fn explain(&self, query: &str, limit: usize) -> Vec<(String, f64)> {
        let scores = self.similarities(query);
        let Some((winner, _)) = scores.first() else {
            return Vec::new();
        };
        let Some((_, centroid)) = self.centroids.iter().find(|(d, _)| d == winner) else {
            return Vec::new();
        };

        let terms = tokenize(query);
        let v = Self::vectorize(&terms, &self.vocabulary, &self.idf, self.vocabulary.len());

        let mut seen: Vec<String> = Vec::new();
        let mut out: Vec<(String, f64)> = Vec::new();
        for t in terms {
            if seen.contains(&t) {
                continue;
            }
            if let Some(&idx) = self.vocabulary.get(&t) {
                let contribution = v[idx] * centroid[idx];
                if contribution > 0.0 {
                    out.push((t.clone(), contribution));
                }
            }
            seen.push(t);
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(limit);
        out
    }

    /// Classify without the async wrapper, for callers already on a hot path.
    ///
    /// # Shrinkage, and why entropy was inverted without it
    ///
    /// The reported distribution is not the raw similarity shares. It is those
    /// shares pulled toward uniform in proportion to how little evidence the
    /// query carried:
    ///
    /// ```text
    ///   p_d = w * (s_d / total) + (1 - w) / 4,   w = min(mass / FULL_EVIDENCE, 1)
    /// ```
    ///
    /// Without it the entropy signal ran *backwards*. A query matching one
    /// stray term puts all of its (tiny) mass on a single domain and so looks
    /// maximally decisive; a rich, specific query matches terms across several
    /// domains and looks uncertain. Measured, mean entropy came out at 0.734 on
    /// ambiguous input against 0.819 on answerable input — the wrong way round,
    /// which would have made an escalation policy fire on precisely the queries
    /// it should have let through.
    ///
    /// Shrinking toward the uniform prior in proportion to the evidence is the
    /// standard treatment of a weak likelihood, and it restores the ordering
    /// for the right reason rather than by rescaling until the test passed.
    pub fn classify_sync(&self, query: &str) -> ClassificationResult {
        let scores = self.similarities(query);
        let (mass, _) = self.evidence(query);
        let top = scores.first().map_or(0.0, |(_, s)| *s);
        let second = scores.get(1).map_or(0.0, |(_, s)| *s);

        let total: f64 = scores.iter().map(|(_, s)| *s).sum();
        let w = (mass / FULL_EVIDENCE).clamp(0.0, 1.0);
        #[allow(clippy::cast_precision_loss)]
        let k = scores.len().max(1) as f64;
        let shrunk: Vec<(CynefinDomain, f64)> = scores
            .iter()
            .map(|(d, s)| {
                let share = if total > 0.0 { s / total } else { 0.0 };
                (*d, w * share + (1.0 - w) / k)
            })
            .collect();

        let margin_ok = top > 0.0 && (top - second) / top >= MIN_RELATIVE_MARGIN;
        if mass < self.min_evidence || top < MIN_SIMILARITY || !margin_ok {
            // Abstain. `Disorder` at exactly zero confidence is the contract an
            // escalation policy triggers on, and finding R1's guards pin it.
            return ClassificationResult::abstained(shrunk);
        }

        ClassificationResult::from_scores(shrunk)
    }

    /// Exemplars the shipped classifier is trained on.
    ///
    /// Chosen for the *shape* of the question rather than its subject: each
    /// domain gets the same verticals, so a term like `pipeline` cannot become
    /// evidence for a domain merely by appearing in its examples.
    ///
    /// See this module's header on why a score of these against the evaluation
    /// corpus is not a clean held-out number, and cross-validation is.
    pub fn default_exemplars() -> Vec<(&'static str, CynefinDomain)> {
        use CynefinDomain::{Chaotic, Clear, Complex, Complicated};
        vec![
            // Clear — the answer is a lookup. Interrogative, present simple.
            ("what is the configured value for this setting", Clear),
            ("list the available options", Clear),
            ("show me the current configuration", Clear),
            ("how many records are in this table", Clear),
            ("what is the default timeout", Clear),
            ("where is this documented", Clear),
            ("which version are we running", Clear),
            ("give me the schema for this dataset", Clear),
            ("what does this error code mean", Clear),
            ("who owns this service", Clear),
            ("when did this job last run", Clear),
            ("look up the retention policy", Clear),
            // Complicated — the answer requires analysis. Diagnostic mood.
            ("why did throughput drop last week", Complicated),
            ("diagnose the root cause of these failures", Complicated),
            ("analyse which factor drives this metric", Complicated),
            (
                "explain the relationship between these two variables",
                Complicated,
            ),
            ("work out why the numbers disagree", Complicated),
            (
                "investigate what changed between these releases",
                Complicated,
            ),
            ("figure out which step is the bottleneck", Complicated),
            (
                "what accounts for the difference in these results",
                Complicated,
            ),
            ("trace where this value comes from", Complicated),
            (
                "determine whether this change caused the regression",
                Complicated,
            ),
            ("break down the contribution of each component", Complicated),
            (
                "assess the impact of that configuration change",
                Complicated,
            ),
            // Complex — the answer requires experiment. Subjunctive, hypothetical.
            ("what if we changed the strategy here", Complex),
            ("should we try a different approach", Complex),
            ("run an experiment to see what happens", Complex),
            ("we are not sure how users will respond", Complex),
            ("explore whether this would help", Complex),
            (
                "it might improve things but we cannot tell in advance",
                Complex,
            ),
            ("probe how the system reacts to this", Complex),
            ("test a small change and see if it holds", Complex),
            ("how would the behaviour shift if we adjusted this", Complex),
            ("nobody knows what the effect of this will be", Complex),
            ("suppose we rolled this out to a subset", Complex),
            ("pilot it somewhere low risk and watch", Complex),
            // Chaotic — act first. Imperative, present continuous, immediacy.
            ("stop it, it is doing damage right now", Chaotic),
            ("halt the process before it makes things worse", Chaotic),
            ("kill this immediately", Chaotic),
            ("everything is failing and we do not know why", Chaotic),
            ("it is spiralling and nothing we try is helping", Chaotic),
            ("shut it down now", Chaotic),
            ("customers are affected and it is getting worse", Chaotic),
            ("we are losing data as we speak", Chaotic),
            ("roll it back now, do not wait for the analysis", Chaotic),
            ("it has gone completely out of control", Chaotic),
            ("cut it off before it spreads any further", Chaotic),
            ("the whole thing is down and nobody can work", Chaotic),
        ]
    }

    /// The shipped classifier.
    ///
    /// # Errors
    ///
    /// Only if [`Self::default_exemplars`] is malformed, which the unit tests
    /// prevent.
    pub fn with_default_exemplars() -> Result<Self, ClassifierError> {
        Self::train(&Self::default_exemplars())
    }
}

/// Scale a vector to unit length, leaving a zero vector alone.
fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[async_trait]
impl QueryClassifier for LexicalClassifier {
    async fn classify(&self, query: &str) -> Result<ClassificationResult, ClassifierError> {
        Ok(self.classify_sync(query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained() -> LexicalClassifier {
        LexicalClassifier::with_default_exemplars().expect("default exemplars are well-formed")
    }

    #[test]
    fn the_default_exemplars_train() {
        let c = trained();
        assert_eq!(c.centroids.len(), 4, "one centroid per domain");
        assert!(
            c.vocabulary.len() > 300,
            "unigrams and bigrams over 48 examples should give a few hundred terms, got {}",
            c.vocabulary.len()
        );
    }

    #[test]
    fn training_refuses_input_it_cannot_learn_from() {
        assert!(LexicalClassifier::train(&[]).is_err());
        assert!(
            LexicalClassifier::train(&[("only one domain here", CynefinDomain::Clear)]).is_err(),
            "a classifier that can only answer one way is not a classifier"
        );
    }

    #[test]
    fn bigrams_are_produced_and_carry_the_signal() {
        let t = tokenize("what if we tried");
        assert!(t.contains(&"what".to_string()));
        assert!(
            t.contains(&"what if".to_string()),
            "the bigram is the term that carries Cynefin signal: {t:?}"
        );
    }

    #[test]
    fn tokenization_is_punctuation_and_case_insensitive() {
        assert_eq!(tokenize("Stop, NOW!"), tokenize("stop now"));
    }

    /// The failure that defines R1: an urgent query with no crisis keyword.
    #[test]
    fn reaches_chaotic_without_a_crisis_keyword() {
        let c = trained();
        // Contains none of emergency/crisis/outage/breach/urgent.
        let r = c.classify_sync("stop the job, it is corrupting rows as we speak");
        assert_eq!(
            r.domain,
            CynefinDomain::Chaotic,
            "scores: {:?}",
            r.all_scores
        );
    }

    #[test]
    fn abstains_on_input_with_no_signal() {
        let c = trained();
        let r = c.classify_sync("zzzz qqqq wwww");
        assert_eq!(r.domain, CynefinDomain::Disorder);
        assert!((r.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn explanation_names_the_terms_that_decided_it() {
        let c = trained();
        let terms = c.explain("what if we tried a different approach", 5);
        assert!(!terms.is_empty(), "an explanation must name something");
        assert!(
            terms
                .iter()
                .any(|(t, _)| t.contains("what if") || t == "if"),
            "the hypothetical marker should be among the drivers: {terms:?}"
        );
        assert!(
            terms.windows(2).all(|w| w[0].1 >= w[1].1),
            "explanations must be ordered by contribution"
        );
    }

    #[test]
    fn classification_is_deterministic() {
        let c = trained();
        let q = "why did the throughput drop after the deploy";
        let a = c.classify_sync(q);
        let b = c.classify_sync(q);
        assert_eq!(a.domain, b.domain);
        assert!((a.confidence - b.confidence).abs() < f64::EPSILON);
        assert!((a.entropy - b.entropy).abs() < f64::EPSILON);
    }

    #[test]
    fn training_on_a_different_corpus_gives_a_different_classifier() {
        // The control: a classifier that ignored its training data would pass
        // every test above.
        let odd = LexicalClassifier::train(&[
            ("banana banana banana", CynefinDomain::Clear),
            ("wombat wombat wombat", CynefinDomain::Chaotic),
        ])
        .expect("two domains");
        // A full phrase rather than a bare "wombat": one term is below the
        // evidence floor and abstains, which is the intended behaviour and not
        // what this control is testing.
        assert_eq!(
            odd.classify_sync("wombat wombat wombat").domain,
            CynefinDomain::Chaotic,
            "the classifier must actually use what it was trained on"
        );
        assert_eq!(
            odd.classify_sync("wombat").domain,
            CynefinDomain::Disorder,
            "a single term is thin evidence and should abstain"
        );
    }
}
