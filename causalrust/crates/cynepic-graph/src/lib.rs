//! # cynepic-graph
//!
//! Typed, compile-time verified stateful agent workflow graphs.
//!
//! Inspired by LangGraph but leveraging Rust's type system to enforce
//! that every workflow state transition is handled. The `StateGraph<S>`
//! builder ensures at compile time that all `CynefinDomain` variants
//! have registered handlers — unhandled branches are compile errors.
//!
//! # Architecture
//!
//! ```text
//! StateGraph<S>
//!   ├── Node<S>: async function (S) -> S
//!   ├── ConditionalEdge: (S) -> NodeId
//!   ├── Checkpoint: serialize S for HITL interrupts
//!   └── GraphHook: observe execution events
//! ```

pub mod checkpoint;
pub mod graph;
pub mod hooks;
pub mod node;

pub use checkpoint::Checkpoint;
pub use graph::{GraphError, StateGraph};
pub use hooks::{EventCollector, GraphEvent, GraphHook, TracingHook};
pub use node::{Node, NodeId};
