//! Print the refusal table: what the estimators decline, and what they answer.
//!
//! ```bash
//! cargo run -p cynepic-causal --example refusal_report --release
//! ```
//!
//! # The number this publishes
//!
//! The **false-answer rate**: inputs that cannot support an estimate and got one
//! anyway. Target zero.
//!
//! Not the refusal rate. A function that refuses everything scores a perfect
//! refusal rate and is useless, which is why the table below also reports what
//! happens to well-posed data.
//!
//! # Why this is worth publishing at all
//!
//! It is the column where this crate differs from the Python tools it is
//! compared against, and the difference is not a speedup.
//!
//! `numpy.linalg.lstsq` returns a minimum-norm solution for a singular design.
//! `statsmodels` fits a model with an empty treatment arm. Neither is *wrong* —
//! they are general numerical tools, and returning the least-squares solution to
//! an underdetermined system is a defensible thing for one to do.
//!
//! But a causal estimate is acted on. When the design is rank deficient the
//! treatment coefficient is not identified by the data, and a number that looks
//! like an effect is worse than an error, because nothing downstream can tell
//! them apart. Reporting `RankDeficient { aliased: ["covariate[2]"] }` is the
//! product.

use cynepic_causal::estimate::linear::LinearATEEstimator;
use cynepic_causal::estimate::propensity::PropensityScoreEstimator;
use cynepic_testkit::Dgp;
use ndarray::{Array1, Array2};

/// An estimator reduced to a printable outcome: the estimate, or the reason it
/// declined.
type Estimator = fn(&Case) -> Result<String, String>;

/// What the correct behaviour is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Expect {
    Refuse,
    Answer,
}

struct Case {
    name: &'static str,
    /// Why this input is unanswerable, for the table. Empty when it is fine.
    why: &'static str,
    expect: Expect,
    treatment: Array1<f64>,
    outcome: Array1<f64>,
    covariates: Array2<f64>,
}

fn make(
    name: &'static str,
    why: &'static str,
    expect: Expect,
    t: Vec<f64>,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
) -> Case {
    let (n, p) = (x.len(), x.first().map_or(0, Vec::len));
    Case {
        name,
        why,
        expect,
        treatment: Array1::from_vec(t),
        outcome: Array1::from_vec(y),
        covariates: Array2::from_shape_fn((n, p), |(i, j)| x[i][j]),
    }
}

fn corpus() -> Vec<Case> {
    let mut cases = vec![
        make(
            "empty dataset",
            "nothing to estimate from",
            Expect::Refuse,
            vec![],
            vec![],
            vec![],
        ),
        make(
            "single unit",
            "no contrast, no variance",
            Expect::Refuse,
            vec![1.0],
            vec![5.0],
            vec![vec![0.5]],
        ),
        make(
            "all treated",
            "no control arm",
            Expect::Refuse,
            vec![1.0; 20],
            (0..20).map(f64::from).collect(),
            (0..20).map(|i| vec![f64::from(i)]).collect(),
        ),
        make(
            "all control",
            "no treated arm",
            Expect::Refuse,
            vec![0.0; 20],
            (0..20).map(f64::from).collect(),
            (0..20).map(|i| vec![f64::from(i)]).collect(),
        ),
        make(
            "mismatched lengths",
            "inputs disagree on n",
            Expect::Refuse,
            vec![1.0, 0.0, 1.0],
            vec![1.0, 2.0],
            vec![vec![0.0], vec![1.0], vec![2.0]],
        ),
    ];

    let n = 60;
    cases.push(make(
        "exactly collinear design",
        "effect not identified by the data",
        Expect::Refuse,
        (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect(),
        (0..n).map(|i| 1.0 + f64::from(i % 3)).collect(),
        (0..n)
            .map(|i| {
                let (a, b) = (f64::from(i % 7), f64::from(i % 5));
                vec![a, b, a + b]
            })
            .collect(),
    ));

    cases.push(make(
        "constant treatment",
        "treatment never varies",
        Expect::Refuse,
        vec![1.0; 40],
        (0..40).map(f64::from).collect(),
        (0..40).map(|i| vec![f64::from(i)]).collect(),
    ));

    for (name, seed, n, p) in [
        ("benign, n=200", 1u64, 200usize, 3usize),
        ("benign, n=2000", 2, 2_000, 3),
        ("small but adequate, n=40", 3, 40, 1),
        ("high dimensional, n=400 p=25", 4, 400, 25),
    ] {
        let d = Dgp::new().with_n(n).with_p(p).sample(seed);
        cases.push(Case {
            name,
            why: "",
            expect: Expect::Answer,
            treatment: d.treatment,
            outcome: d.outcome,
            covariates: d.covariates,
        });
    }

    cases
}

fn main() {
    println!("cynepic-causal — refusal behaviour\n");
    println!("The number that matters is FALSE ANSWERS: inputs that cannot support");
    println!("an estimate and got one anyway. Target zero.\n");

    let estimators: [(&str, Estimator); 3] = [
        ("ols_adjusted", |c| {
            LinearATEEstimator::ols_adjusted(&c.treatment, &c.outcome, &c.covariates)
                .map(|r| format!("{:.4}", r.ate()))
                .map_err(|e| format!("{e}"))
        }),
        ("ipw", |c| {
            PropensityScoreEstimator::ipw(&c.treatment, &c.outcome, &c.covariates)
                .map(|r| format!("{:.4}", r.ate()))
                .map_err(|e| format!("{e}"))
        }),
        ("difference_in_means", |c| {
            LinearATEEstimator::difference_in_means(&c.treatment, &c.outcome)
                .map(|r| format!("{:.4}", r.ate()))
                .map_err(|e| format!("{e}"))
        }),
    ];

    let mut false_answers = 0usize;
    let mut refusable = 0usize;

    println!("── inputs that cannot support an estimate ───────────────────────");
    println!("  {:<28} {:<34} ols_adjusted", "case", "why");
    for c in corpus().iter().filter(|c| c.expect == Expect::Refuse) {
        refusable += 1;
        let verdict = match (estimators[0].1)(c) {
            Ok(v) => {
                false_answers += 1;
                format!("ANSWERED {v}  <- false answer")
            }
            Err(e) => {
                // Just the error kind, not the whole sentence — the table is for
                // scanning.
                let short = e
                    .split([':', ';', '('])
                    .next()
                    .unwrap_or(&e)
                    .trim()
                    .to_string();
                format!("refused: {short}")
            }
        };
        println!("  {:<28} {:<34} {}", c.name, c.why, verdict);
    }

    let mut false_refusals = 0usize;
    let mut answerable = 0usize;

    println!("\n── well-posed inputs ────────────────────────────────────────────");
    println!(
        "  {:<28} {:>14} {:>14} {:>14}",
        "case", "ols", "ipw", "diff-in-means"
    );
    for c in corpus().iter().filter(|c| c.expect == Expect::Answer) {
        answerable += 1;
        let mut cells = Vec::new();
        for (i, (_, f)) in estimators.iter().enumerate() {
            match f(c) {
                Ok(v) => cells.push(v),
                Err(_) => {
                    if i == 0 {
                        false_refusals += 1;
                    }
                    cells.push("refused".to_string());
                }
            }
        }
        println!(
            "  {:<28} {:>14} {:>14} {:>14}",
            c.name, cells[0], cells[1], cells[2]
        );
    }

    println!();
    println!("── summary ──────────────────────────────────────────────────────");
    println!("  unanswerable inputs        {refusable}");
    println!("  FALSE ANSWERS              {false_answers}   <- must be 0");
    println!("  well-posed inputs          {answerable}");
    println!("  false refusals (ols)       {false_refusals}");

    println!("\nA `refused` under ipw on well-posed data is not a false refusal:");
    println!("weighting legitimately declines when overlap is insufficient, and");
    println!("that is the estimator working. `ols_adjusted` has no such excuse,");
    println!("which is why only its column is scored above.");

    println!("\n── the same inputs, through numpy ───────────────────────────────");
    println!("  Measured, not asserted (numpy 2.4.4, reproduce with the snippet in");
    println!("  scripts/generate_parity_fixtures.py's header):");
    println!();
    println!("  exactly collinear design   numpy.linalg.lstsq -> ATE = -0.000562");
    println!("                             rank 4 of 5 columns, smallest singular");
    println!("                             value 1.8e-15. No error. No warning.");
    println!("  all treated (no control)   numpy.linalg.lstsq -> coefficient 4.75");
    println!("                             No error.");
    println!();
    println!("  The first number is the one to look at. -0.000562 does not read as");
    println!("  \"undefined\" — it reads as \"no effect\", which is a finding somebody");
    println!("  might publish. The effect is not identified by that design at all.");
    println!();
    println!("  numpy is not wrong here: returning the minimum-norm solution to an");
    println!("  underdetermined system is the documented, correct behaviour for a");
    println!("  linear-algebra routine. It is wrong for a causal estimate, and the");
    println!("  difference is the reason this crate returns");
    println!("  RankDeficient {{ aliased: [\"covariate[2]\"] }} instead.");
}
