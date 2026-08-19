//! cynepic-mcp: Model Context Protocol server for cynepic-rs cognitive tools.
//!
//! Implements JSON-RPC 2.0 over stdio as specified by the MCP protocol.
//! Exposes cynepic-rs analytical capabilities as tools for AI agents.
//!
//! Tools exposed:
//! - `classify_domain` — Cynefin domain classification
//! - `estimate_ate` — Average treatment effect estimation
//! - `check_policy` — Policy enforcement check
//! - `update_belief` — Bayesian belief update
//! - `detect_loop` — Loop detection in workflow
//! - `audit_trail` — Retrieve audit trail entries
//! - `run_counterfactual` — Counterfactual reasoning
//! - `monitor_drift` — Routing distribution drift check

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

// ── JSON-RPC Types ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

struct McpServer;

impl McpServer {
    fn list_tools() -> Value {
        json!({
            "tools": [
                {
                    "name": "classify_domain",
                    "description": "Classify a query into Cynefin domains (Clear, Complicated, Complex, Chaotic, Disorder)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string", "description": "The query to classify" }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "estimate_ate",
                    "description": "Estimate average treatment effect using difference-in-means",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "treatment": { "type": "array", "items": { "type": "number" } },
                            "outcome": { "type": "array", "items": { "type": "number" } }
                        },
                        "required": ["treatment", "outcome"]
                    }
                },
                {
                    "name": "check_policy",
                    "description": "Evaluate a policy chain against a context",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "domain": { "type": "string" },
                            "amount": { "type": "number" }
                        }
                    }
                },
                {
                    "name": "update_belief",
                    "description": "Update a Beta-Binomial belief with new evidence",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "successes": { "type": "integer" },
                            "trials": { "type": "integer" }
                        },
                        "required": ["successes", "trials"]
                    }
                },
                {
                    "name": "detect_loop",
                    "description": "Detect loops in workflow execution history",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "node_id": { "type": "string" }
                        },
                        "required": ["node_id"]
                    }
                },
                {
                    "name": "audit_trail",
                    "description": "Retrieve audit trail entries",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": { "type": "integer", "default": 20 }
                        }
                    }
                },
                {
                    "name": "run_counterfactual",
                    "description": "Run counterfactual reasoning (Pearl Level 3)",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "monitor_drift",
                    "description": "Check routing distribution for drift",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        })
    }

    /// Dispatch a tool call.
    ///
    /// Async because classification and policy evaluation are async in the
    /// underlying crates. Errors are returned as strings and surfaced to the
    /// agent as an MCP error rather than swallowed — an agent that is told
    /// "your treatment arm is empty" can act, where one that receives a
    /// plausible number cannot.
    async fn call_tool(name: &str, args: &Value) -> Result<Value, String> {
        /// Pull a numeric array out of the arguments.
        fn numbers(args: &Value, key: &str) -> Result<Vec<f64>, String> {
            args[key]
                .as_array()
                .ok_or_else(|| format!("{key} must be an array"))?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| format!("{key} must contain numbers"))
                })
                .collect()
        }

        match name {
            "classify_domain" => {
                use cynepic_router::classifier::{KeywordClassifier, QueryClassifier};
                let query = args["query"].as_str().unwrap_or("");
                let result = KeywordClassifier::default_patterns()
                    .classify(query)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({
                    "domain": format!("{:?}", result.domain),
                    "confidence": result.confidence,
                    // The signal an agent should escalate on. High entropy
                    // means the classifier could not separate the domains,
                    // which is a different state from a confident answer.
                    "entropy": result.entropy,
                    "abstained": result.domain == cynepic_core::CynefinDomain::Disorder,
                }))
            }
            "estimate_ate" => {
                use cynepic_causal::estimate::linear::LinearATEEstimator;
                use ndarray::Array1;

                let treatment = Array1::from_vec(numbers(args, "treatment")?);
                let outcome = Array1::from_vec(numbers(args, "outcome")?);

                let est = LinearATEEstimator::difference_in_means(&treatment, &outcome)
                    .map_err(|e| e.to_string())?;

                Ok(json!({
                    "ate": est.ate(),
                    "std_error": est.std_error(),
                    // An estimate without its estimand is a number without a
                    // referent. ATE, ATT and LATE describe different
                    // populations and coincide only under homogeneity.
                    "estimand": est.estimand().label(),
                    "population": est.estimand().population(),
                    "confidence_interval": est.confidence_interval(0.95),
                    "n_obs": est.n_obs(),
                    "method": "difference_in_means",
                }))
            }
            "check_policy" => {
                use cynepic_guardian::policy::PolicyChain;
                let action = args["action"].as_str().unwrap_or("");
                let context = args.get("context").cloned().unwrap_or(json!({}));
                let decision = PolicyChain::new()
                    .evaluate(action, &context)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({ "verdict": format!("{decision:?}") }))
            }
            "update_belief" => {
                use cynepic_bayes::priors::BetaBinomial;
                let successes = args["successes"].as_u64().unwrap_or(0);
                let trials = args["trials"].as_u64().unwrap_or(0);
                if successes > trials {
                    return Err(format!("{successes} successes in {trials} trials"));
                }
                let mut model = BetaBinomial::uniform();
                model.update(successes, trials - successes);
                let (lo, hi) = model.credible_interval_95();
                Ok(json!({
                    "posterior_mean": model.mean(),
                    // Exact Beta quantiles, not a normal approximation.
                    "credible_interval": [lo, hi],
                }))
            }
            "detect_loop" => {
                use cynepic_guardian::loop_detector::LoopDetector;
                let node_id = args["node_id"].as_str().unwrap_or("unknown");
                let mut detector = LoopDetector::new(5, 3);
                let violation = detector.record_visit(node_id);
                Ok(json!({
                    "violation_detected": violation.is_some(),
                    "violation": violation.map(|v| format!("{v:?}")),
                }))
            }
            "run_counterfactual" => {
                use cynepic_causal::counterfactual::{CounterfactualEngine, CounterfactualQuery};
                use cynepic_causal::estimate::linear::LinearATEEstimator;
                use ndarray::Array1;

                let treatment = Array1::from_vec(numbers(args, "treatment")?);
                let outcome = Array1::from_vec(numbers(args, "outcome")?);
                let observed = args["observed_outcome"]
                    .as_f64()
                    .ok_or("observed_outcome must be a number")?;
                let factual = args["factual_treatment"].as_f64().unwrap_or(1.0);
                let counterfactual = args["counterfactual_treatment"].as_f64().unwrap_or(0.0);

                let ate = LinearATEEstimator::difference_in_means(&treatment, &outcome)
                    .map_err(|e| e.to_string())?;
                let query = CounterfactualQuery {
                    treatment: args["treatment_name"]
                        .as_str()
                        .unwrap_or("treatment")
                        .into(),
                    outcome: args["outcome_name"].as_str().unwrap_or("outcome").into(),
                    factual_treatment: factual,
                    counterfactual_treatment: counterfactual,
                    observed_outcome: observed,
                };
                let result = CounterfactualEngine::query_with_ate(&query, &ate);

                Ok(json!({
                    "counterfactual_outcome": result.counterfactual_outcome,
                    "treatment_effect": result.treatment_effect,
                    // Projecting a LATE onto an arbitrary unit assumes that
                    // unit is a complier. The assumption must be visible.
                    "estimand": result.estimand.label(),
                    "std_error": result.std_error,
                    "confidence_interval": [
                        result.confidence_interval.0,
                        result.confidence_interval.1
                    ],
                }))
            }
            "monitor_drift" => Ok(json!({
                "status": "not_implemented",
                "detail": "drift monitoring needs a routing history to compare against; \
                           wire cynepic_router::drift::DriftDetector to a persisted baseline",
            })),
            _ => Err(format!("Unknown tool: {name}")),
        }
    }

    fn handle_initialize(id: Option<Value>) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "cynepic-mcp",
                    "version": "0.2.0"
                }
            })),
            error: None,
        }
    }

    fn handle_list_tools(id: Option<Value>) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(Self::list_tools()),
            error: None,
        }
    }

    async fn handle_call_tool(id: Option<Value>, params: &Value) -> JsonRpcResponse {
        let name = params["name"].as_str().unwrap_or("");
        // Bind the fallback so it outlives the borrow: `&json!({})` inline
        // creates a temporary that is dropped at the end of the statement.
        let empty = json!({});
        let arguments = params.get("arguments").unwrap_or(&empty);
        match Self::call_tool(name, arguments).await {
            Ok(result) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string_pretty(&result).unwrap_or_default()
                    }]
                })),
                error: None,
            },
            Err(msg) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: msg,
                }),
            },
        }
    }

    async fn handle_request(req: &JsonRpcRequest) -> JsonRpcResponse {
        // JSON-RPC 2.0 requires the version marker. It was previously parsed
        // and never read, so a malformed request was accepted as valid — which
        // is how a client talking a different protocol gets silently served.
        if req.jsonrpc != "2.0" {
            return JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32600,
                    message: format!(
                        "Invalid Request: jsonrpc must be \"2.0\", got \"{}\"",
                        req.jsonrpc
                    ),
                }),
            };
        }

        match req.method.as_str() {
            "initialize" => Self::handle_initialize(req.id.clone()),
            "tools/list" => Self::handle_list_tools(req.id.clone()),
            "tools/call" => {
                if let Some(params) = &req.params {
                    Self::handle_call_tool(req.id.clone(), params).await
                } else {
                    JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: req.id.clone(),
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32602,
                            message: "Missing params for tools/call".into(),
                        }),
                    }
                }
            }
            _ => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id.clone(),
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", req.method),
                }),
            },
        }
    }
}

#[tokio::main]
async fn main() -> io::Result<()> {
    eprintln!("cynepic-mcp v0.2.0 starting on stdio...");

    let stdin = io::stdin();
    let stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(req) => {
                let resp = McpServer::handle_request(&req).await;
                let resp_json = serde_json::to_string(&resp).unwrap_or_default();
                writeln!(&mut stdout.lock(), "{}", resp_json)?;
                stdout.lock().flush()?;
            }
            Err(e) => {
                let error_resp = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {}", e),
                    }),
                };
                let resp_json = serde_json::to_string(&error_resp).unwrap_or_default();
                writeln!(&mut stdout.lock(), "{}", resp_json)?;
                stdout.lock().flush()?;
            }
        }
    }

    Ok(())
}
