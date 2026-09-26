//! Properties a workflow orchestrator must hold.
//!
//! # What this suite is for
//!
//! `StateGraph` is the component that decides what an agent does next. Its unit
//! tests check that a graph runs. These check the properties that make a run
//! *trustworthy*:
//!
//! - **Determinism** — the same graph and the same input produce the same
//!   output. Without it a failure cannot be reproduced and a fix cannot be
//!   verified.
//! - **Budget** — `max_steps` is a hard bound, not a suggestion. It is the only
//!   thing standing between a routing bug and an unbounded spend.
//! - **Checkpoint fidelity** — resuming from a checkpoint must produce exactly
//!   what running straight through would have. A checkpoint that returns a
//!   different answer is worse than no checkpoint, because the difference is
//!   silent.
//! - **Observability** — every started node reports exactly one terminal event.
//!   A hook stream with unmatched events cannot be used for billing or tracing.
//!
//! Each test names the production consequence of its failing, because a
//! property test that only says "assertion failed" is hard to act on at 3am.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use cynepic_graph::checkpoint::Checkpoint;
use cynepic_graph::graph::{GraphError, StateGraph};
use cynepic_graph::hooks::{EventCollector, GraphEvent};
use cynepic_graph::node::{FnNode, NodeId};

/// A linear pipeline of `n` nodes, each adding its index to the state.
fn linear_graph(n: usize) -> StateGraph<i64> {
    let mut graph = StateGraph::new();
    for i in 0..n {
        #[allow(clippy::cast_possible_wrap)]
        let delta = i as i64;
        graph = graph.add_node(Arc::new(FnNode::new(
            format!("n{i}"),
            move |x: i64| async move { Ok(x + delta) },
        )));
    }
    for i in 0..n.saturating_sub(1) {
        graph = graph.add_edge(
            NodeId::new(format!("n{i}")),
            NodeId::new(format!("n{}", i + 1)),
        );
    }
    graph.set_entry(NodeId::new("n0"))
}

// ===========================================================================
// Determinism
// ===========================================================================

/// The same graph on the same input must produce the same output, every time.
///
/// Without this a failure report is unactionable: the run that produced it
/// cannot be recreated.
#[tokio::test]
async fn execution_is_deterministic() {
    let graph = linear_graph(8);
    let first = graph.execute(0, 100).await.expect("valid graph");
    for run in 0..20 {
        let again = graph.execute(0, 100).await.expect("valid graph");
        assert_eq!(again, first, "run {run} diverged from the first");
    }
}

/// Conditional routing must be a pure function of the state.
///
/// A router that consults anything else — a clock, a counter, an RNG — makes
/// the whole graph irreproducible, and the effect is invisible until a rare
/// path is taken.
#[tokio::test]
async fn conditional_routing_is_a_pure_function_of_state() {
    let build = || {
        StateGraph::new()
            .add_node(Arc::new(FnNode::new(
                "check",
                |x: i64| async move { Ok(x) },
            )))
            .add_node(Arc::new(FnNode::new(
                "even",
                |x: i64| async move { Ok(x * 10) },
            )))
            .add_node(Arc::new(FnNode::new("odd", |x: i64| async move { Ok(-x) })))
            .set_entry(NodeId::new("check"))
            .add_conditional_edge(NodeId::new("check"), |x: &i64| {
                if x % 2 == 0 {
                    NodeId::new("even")
                } else {
                    NodeId::new("odd")
                }
            })
    };

    for input in [0i64, 1, 2, 3, 100, -7] {
        let expected = build().execute(input, 10).await.expect("valid");
        for _ in 0..5 {
            assert_eq!(
                build().execute(input, 10).await.expect("valid"),
                expected,
                "routing on input {input} was not deterministic"
            );
        }
    }
}

// ===========================================================================
// Step budget
// ===========================================================================

/// `max_steps` must be a hard bound.
///
/// It is the only thing between a routing bug and an unbounded spend, so it
/// must never be exceeded — not by one step, and not "usually".
#[tokio::test]
async fn max_steps_is_a_hard_bound() {
    let executions = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&executions);

    // A deliberate infinite loop through a conditional edge, which `validate`
    // permits because a conditional cycle can be intentional.
    let graph = StateGraph::new()
        .add_node(Arc::new(FnNode::new("loop", move |x: i64| {
            let c = Arc::clone(&counter);
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(x + 1)
            }
        })))
        .set_entry(NodeId::new("loop"))
        .add_conditional_edge(NodeId::new("loop"), |_: &i64| NodeId::new("loop"));

    for budget in [1usize, 5, 17, 50] {
        executions.store(0, Ordering::SeqCst);
        let result = graph.execute(0, budget).await;

        assert!(
            matches!(result, Err(GraphError::MaxStepsExceeded(b)) if b == budget),
            "budget {budget}: expected MaxStepsExceeded, got {result:?}"
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            budget,
            "budget {budget}: the node ran a different number of times"
        );
    }
}

/// A graph that terminates naturally must not consume its whole budget.
#[tokio::test]
async fn a_terminating_graph_stops_at_its_terminal_node() {
    let graph = linear_graph(4);
    let collector = Arc::new(EventCollector::new());
    let graph = graph.add_hook(Arc::clone(&collector) as Arc<_>);

    graph.execute(0, 1_000).await.expect("valid graph");

    let started = collector
        .events()
        .iter()
        .filter(|e| matches!(e, GraphEvent::NodeStarted { .. }))
        .count();
    assert_eq!(started, 4, "a 4-node pipeline ran {started} nodes");
}

// ===========================================================================
// Checkpoint fidelity
// ===========================================================================

/// Resuming from a checkpoint must produce exactly what running straight
/// through would have.
///
/// This is the property that makes a checkpoint worth having. A resume that
/// returns a *different* answer is worse than no checkpoint at all, because
/// nothing surfaces the difference.
#[tokio::test]
async fn resume_reproduces_an_uninterrupted_run() {
    let graph = linear_graph(6);
    let straight_through = graph.execute(0, 100).await.expect("valid graph");

    // Interrupt at every possible point and confirm the answer is unchanged.
    for cut in 1..6 {
        // Replay the prefix by hand to get the state at the cut.
        #[allow(clippy::cast_possible_wrap)]
        let state_at_cut: i64 = (0..cut as i64).sum();

        let checkpoint = Checkpoint::new(state_at_cut, NodeId::new(format!("n{cut}")), cut);
        let resumed = graph
            .resume(checkpoint, 100)
            .await
            .expect("resumable checkpoint");

        assert_eq!(
            resumed, straight_through,
            "resuming at step {cut} gave {resumed}, straight through gave {straight_through}"
        );
    }
}

/// A checkpoint must survive a serialisation round trip unchanged.
///
/// Checkpoints exist to cross a process boundary. One that only works in memory
/// is a struct, not a checkpoint.
#[tokio::test]
async fn checkpoint_survives_a_json_round_trip() {
    let graph = linear_graph(6);
    let straight_through = graph.execute(0, 100).await.expect("valid graph");

    let checkpoint = Checkpoint::new(3i64, NodeId::new("n3"), 3);
    let json = checkpoint.to_json().expect("serialises");
    let restored: Checkpoint<i64> = Checkpoint::from_json(&json).expect("deserialises");

    assert_eq!(restored.state, 3);
    assert_eq!(restored.next_node, NodeId::new("n3"));
    assert_eq!(restored.steps_completed, 3);

    let resumed = graph.resume(restored, 100).await.expect("resumable");
    assert_eq!(
        resumed, straight_through,
        "a round-tripped checkpoint produced a different answer"
    );
}

/// The resumed run must respect the *remaining* budget, not a fresh one.
///
/// Otherwise a checkpoint becomes a way to launder a step budget: interrupt,
/// resume, and the cap resets.
#[tokio::test]
async fn resume_charges_against_the_original_budget() {
    let graph = StateGraph::new()
        .add_node(Arc::new(FnNode::new(
            "loop",
            |x: i64| async move { Ok(x + 1) },
        )))
        .set_entry(NodeId::new("loop"))
        .add_conditional_edge(NodeId::new("loop"), |_: &i64| NodeId::new("loop"));

    // 8 of a 10-step budget already spent: only 2 remain.
    let checkpoint = Checkpoint::new(0i64, NodeId::new("loop"), 8);
    let result = graph.resume(checkpoint, 10).await;
    assert!(
        matches!(result, Err(GraphError::MaxStepsExceeded(10))),
        "expected the remaining budget to be honoured, got {result:?}"
    );

    // And an exhausted budget must refuse outright.
    let spent = Checkpoint::new(0i64, NodeId::new("loop"), 10);
    assert!(
        matches!(
            graph.resume(spent, 10).await,
            Err(GraphError::MaxStepsExceeded(10))
        ),
        "a fully spent budget must not resume"
    );
}

// ===========================================================================
// Timeouts
// ===========================================================================

/// A hanging node must be cut at roughly the configured timeout.
///
/// The bound that stops one stuck tool call from holding a workflow open
/// indefinitely.
#[tokio::test]
async fn a_hanging_node_is_cut_at_the_timeout() {
    let graph = StateGraph::new()
        .add_node(Arc::new(FnNode::new("hang", |x: i64| async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok(x)
        })))
        .set_entry(NodeId::new("hang"));

    let started = std::time::Instant::now();
    let result = graph
        .execute_with_timeout(0, 10, Duration::from_millis(50))
        .await;
    let elapsed = started.elapsed();

    assert!(
        matches!(result, Err(GraphError::NodeTimedOut { .. })),
        "expected NodeTimedOut, got {result:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "the timeout took {elapsed:?} to fire; it must bound the wait, not merely report it"
    );
}

/// A node that finishes inside the timeout must not be disturbed by it.
#[tokio::test]
async fn a_fast_node_is_unaffected_by_the_timeout() {
    let graph = linear_graph(4);
    let with_timeout = graph
        .execute_with_timeout(0, 100, Duration::from_secs(5))
        .await
        .expect("fast nodes");
    let without = linear_graph(4).execute(0, 100).await.expect("fast nodes");
    assert_eq!(
        with_timeout, without,
        "adding a generous timeout changed the result"
    );
}

// ===========================================================================
// Observability
// ===========================================================================

/// Every started node must report exactly one terminal event.
///
/// An event stream with unmatched starts cannot be used for billing or tracing:
/// a missing completion is indistinguishable from a node still running.
#[tokio::test]
async fn every_started_node_reports_exactly_one_terminal_event() {
    let collector = Arc::new(EventCollector::new());
    let graph = linear_graph(5).add_hook(Arc::clone(&collector) as Arc<_>);
    graph.execute(0, 100).await.expect("valid graph");

    let events = collector.events();
    let started = events
        .iter()
        .filter(|e| matches!(e, GraphEvent::NodeStarted { .. }))
        .count();
    let terminal = events
        .iter()
        .filter(|e| {
            matches!(
                e,
                GraphEvent::NodeCompleted { .. } | GraphEvent::NodeFailed { .. }
            )
        })
        .count();

    assert_eq!(
        started, terminal,
        "{started} starts but {terminal} terminals"
    );
    assert!(started > 0, "no events were emitted at all");
}

/// A failing node must emit `NodeFailed`, not `NodeCompleted`.
///
/// The event stream is what an operator reads. If a failure is reported as a
/// completion, the dashboard is lying about the thing it exists to show.
#[tokio::test]
async fn a_failing_node_emits_failure_not_completion() {
    let collector = Arc::new(EventCollector::new());
    let graph = StateGraph::new()
        .add_node(Arc::new(FnNode::new("boom", |_: i64| async move {
            Err(cynepic_graph::node::NodeError::ExecutionFailed(
                "deliberate".into(),
            ))
        })))
        .set_entry(NodeId::new("boom"))
        .add_hook(Arc::clone(&collector) as Arc<_>);

    let result = graph.execute(0, 10).await;
    assert!(matches!(result, Err(GraphError::NodeFailed { .. })));

    let events = collector.events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, GraphEvent::NodeFailed { .. })),
        "no NodeFailed event was emitted"
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, GraphEvent::NodeCompleted { .. })),
        "a failing node reported completion"
    );
}

/// A timed-out node must also be reported as a failure.
#[tokio::test]
async fn a_timed_out_node_emits_failure() {
    let collector = Arc::new(EventCollector::new());
    let graph = StateGraph::new()
        .add_node(Arc::new(FnNode::new("hang", |x: i64| async move {
            tokio::time::sleep(Duration::from_secs(30)).await;
            Ok(x)
        })))
        .set_entry(NodeId::new("hang"))
        .add_hook(Arc::clone(&collector) as Arc<_>);

    let _ = graph
        .execute_with_timeout(0, 10, Duration::from_millis(30))
        .await;

    assert!(
        collector
            .events()
            .iter()
            .any(|e| matches!(e, GraphEvent::NodeFailed { .. })),
        "a timeout produced no failure event; it would be invisible to an operator"
    );
}

// ===========================================================================
// Validation
// ===========================================================================

/// A fixed-edge cycle must be rejected before anything runs.
///
/// Cheaper to catch at validation than at the step budget, and the error names
/// the cycle rather than reporting exhaustion.
#[tokio::test]
async fn a_fixed_edge_cycle_is_rejected_by_validation() {
    let graph = StateGraph::new()
        .add_node(Arc::new(FnNode::new("a", |x: i64| async move { Ok(x) })))
        .add_node(Arc::new(FnNode::new("b", |x: i64| async move { Ok(x) })))
        .set_entry(NodeId::new("a"))
        .add_edge(NodeId::new("a"), NodeId::new("b"))
        .add_edge(NodeId::new("b"), NodeId::new("a"));

    assert!(
        matches!(graph.validate(), Err(GraphError::CycleDetected { .. })),
        "a -> b -> a was not detected"
    );
    assert!(
        matches!(
            graph.execute(0, 100).await,
            Err(GraphError::CycleDetected { .. })
        ),
        "execute must validate before running"
    );
}

/// An edge to a node that does not exist must be caught at validation.
#[tokio::test]
async fn an_edge_to_a_missing_node_is_rejected() {
    let graph = StateGraph::new()
        .add_node(Arc::new(FnNode::new("a", |x: i64| async move { Ok(x) })))
        .set_entry(NodeId::new("a"))
        .add_edge(NodeId::new("a"), NodeId::new("ghost"));

    assert!(
        graph.validate().is_err(),
        "an edge to a nonexistent node was accepted"
    );
}

/// A graph with no entry node must refuse to run.
#[tokio::test]
async fn a_graph_without_an_entry_node_refuses() {
    let graph: StateGraph<i64> =
        StateGraph::new().add_node(Arc::new(FnNode::new("a", |x: i64| async move { Ok(x) })));

    assert!(matches!(graph.validate(), Err(GraphError::NoEntryNode)));
    assert!(matches!(
        graph.execute(0, 10).await,
        Err(GraphError::NoEntryNode)
    ));
}

// ===========================================================================
// OR2 / OR3 — the same properties, at scale and under concurrency
// ===========================================================================
//
// The fifteen properties above hold on graphs of six nodes carrying an `i64`.
// "No findings" on a fixture that small is a weaker statement than it looks:
// a checkpoint that truncates, a step counter that overflows a small type, or
// a serialiser that degrades on a large payload would all pass at n = 6.
//
// OR2 asks whether checkpoint/resume still reproduces an uninterrupted run at
// length and at size. OR3 asks whether determinism survives concurrent node
// execution.

/// A pipeline long enough that an off-by-one or a truncation has room to show.
fn long_linear_graph(n: usize) -> StateGraph<i64> {
    linear_graph(n)
}

/// OR2, length: resume must reproduce the straight-through answer on a long graph.
///
/// The existing property cuts a six-node graph at five points. This cuts a
/// 400-node graph at eighty, which is the version that would catch a resume
/// path that works near the ends and drifts in the middle.
#[tokio::test]
async fn or2_resume_reproduces_an_uninterrupted_run_at_length() {
    const N: usize = 400;
    let graph = long_linear_graph(N);
    let straight_through = graph.execute(0, N + 10).await.expect("valid graph");

    #[allow(clippy::cast_possible_wrap)]
    let expected: i64 = (0..N as i64).sum();
    assert_eq!(
        straight_through, expected,
        "the long pipeline did not compute its own sum; the fixture is wrong, not the graph"
    );

    for cut in (5..N).step_by(5) {
        #[allow(clippy::cast_possible_wrap)]
        let state_at_cut: i64 = (0..cut as i64).sum();
        let checkpoint = Checkpoint::new(state_at_cut, NodeId::new(format!("n{cut}")), cut);
        let resumed = graph
            .resume(checkpoint, N + 10)
            .await
            .expect("resumable checkpoint");
        assert_eq!(
            resumed, straight_through,
            "resuming a {N}-node graph at step {cut} gave {resumed}, straight through gave \
             {straight_through}; checkpoint fidelity holds at six nodes and not at {N}"
        );
    }
}

/// OR2, size: a checkpoint carrying a large state must round-trip exactly.
///
/// Checkpoints exist to cross a process boundary, and the boundary is where
/// size starts to matter — a truncating writer, a lossy float path, or a
/// serialiser with an implicit cap would all be invisible on an `i64`.
#[tokio::test]
async fn or2_checkpoint_round_trips_a_large_state() {
    const WIDTH: usize = 50_000;

    // A wide state that a lossy path would visibly damage.
    let mut graph: StateGraph<Vec<i64>> = StateGraph::new();
    for i in 0..4 {
        #[allow(clippy::cast_possible_wrap)]
        let delta = i as i64;
        graph = graph.add_node(Arc::new(FnNode::new(
            format!("n{i}"),
            move |mut v: Vec<i64>| async move {
                for x in &mut v {
                    *x += delta;
                }
                Ok(v)
            },
        )));
    }
    for i in 0..3 {
        graph = graph.add_edge(
            NodeId::new(format!("n{i}")),
            NodeId::new(format!("n{}", i + 1)),
        );
    }
    let graph = graph.set_entry(NodeId::new("n0"));

    #[allow(clippy::cast_possible_wrap)]
    let initial: Vec<i64> = (0..WIDTH as i64).collect();
    let straight_through = graph
        .execute(initial.clone(), 10)
        .await
        .expect("valid graph");
    assert_eq!(
        straight_through.len(),
        WIDTH,
        "the run changed the state's length"
    );

    // Cut after the first node. n0 adds its index, which is zero, so the state
    // at the cut is the initial state -- written as a clone rather than as an
    // arithmetic no-op so the reason is visible.
    let after_first: Vec<i64> = initial.clone();
    let checkpoint = Checkpoint::new(after_first, NodeId::new("n1"), 1);
    let json = checkpoint.to_json().expect("serialises");
    let restored: Checkpoint<Vec<i64>> = Checkpoint::from_json(&json).expect("deserialises");

    assert_eq!(
        restored.state.len(),
        WIDTH,
        "a {WIDTH}-element state came back {} long; the checkpoint truncates at size",
        restored.state.len()
    );
    let resumed = graph.resume(restored, 10).await.expect("resumable");
    assert_eq!(
        resumed, straight_through,
        "a large state resumed to a different answer than running straight through"
    );
}

/// OR3: there is no concurrent node execution, so its determinism is serial.
///
/// OR3 asks whether determinism holds under concurrent node execution, with the
/// falsifier "non-determinism under concurrency -> the determinism claim needs
/// a serial-only caveat".
///
/// **That falsifier cannot fire, because the executor has no concurrency.**
/// `graph.rs` contains no `tokio::spawn`, no `JoinSet` and no `join_all`; the
/// run loop visits one node, awaits it, and follows an edge. The determinism
/// claim is therefore serial-only by construction rather than by caveat, and
/// OR3 is not a question this component can currently answer.
///
/// This test pins that state of affairs by observing it rather than asserting
/// it about the source: nodes record entry and exit, and no two node bodies
/// may overlap. If parallel branches are ever added, this fails, and OR3 stops
/// being inapplicable and becomes live — which is the point of writing it now
/// instead of a comment.
#[tokio::test]
async fn or3_execution_is_serial_so_the_determinism_claim_is_serial_only() {
    use std::sync::Mutex;

    #[derive(Default)]
    struct Overlap {
        in_flight: usize,
        max_in_flight: usize,
        order: Vec<String>,
    }

    let seen: Arc<Mutex<Overlap>> = Arc::new(Mutex::new(Overlap::default()));

    let mut graph: StateGraph<i64> = StateGraph::new();
    for i in 0..12 {
        let seen = Arc::clone(&seen);
        let name = format!("n{i}");
        let label = name.clone();
        graph = graph.add_node(Arc::new(FnNode::new(name, move |x: i64| {
            let seen = Arc::clone(&seen);
            let label = label.clone();
            async move {
                {
                    let mut g = seen.lock().expect("lock");
                    g.in_flight += 1;
                    g.max_in_flight = g.max_in_flight.max(g.in_flight);
                    g.order.push(label.clone());
                }
                // Yield so that, if the executor ever ran branches together,
                // a second body would have the opportunity to enter here.
                tokio::task::yield_now().await;
                tokio::time::sleep(Duration::from_millis(1)).await;
                {
                    let mut g = seen.lock().expect("lock");
                    g.in_flight -= 1;
                }
                Ok(x + 1)
            }
        })));
    }
    for i in 0..11 {
        graph = graph.add_edge(
            NodeId::new(format!("n{i}")),
            NodeId::new(format!("n{}", i + 1)),
        );
    }
    let graph = graph.set_entry(NodeId::new("n0"));

    let out = graph.execute(0, 50).await.expect("valid graph");
    assert_eq!(out, 12, "every node should have run exactly once");

    let g = seen.lock().expect("lock");
    assert_eq!(
        g.max_in_flight, 1,
        "two node bodies overlapped: the executor has gained concurrency, so OR3 is \
         now a live question and determinism under parallel branches must be measured \
         rather than assumed"
    );
    let expected: Vec<String> = (0..12).map(|i| format!("n{i}")).collect();
    assert_eq!(g.order, expected, "nodes did not run in edge order");
}
