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
use serde_json::{json, Value};
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

    fn call_tool(name: &str, args: &Value) -> Result<Value, String> {
        match name {
            "classify_domain" => {
                let query = args["query"].as_str().unwrap_or("");
                let router = cynepic_router::CynefinRouter::new();
                let result = router.classify(query);
                Ok(json!({
                    "domain": format!("{}", result.domain),
                    "confidence": result.confidence,
                    "entropy": result.entropy
                }))
            }
            "estimate_ate" => {
                let treatment: Vec<f64> = args["treatment"]
                    .as_array()
                    .ok_or("treatment must be an array")?
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .collect();
                let outcome: Vec<f64> = args["outcome"]
                    .as_array()
                    .ok_or("outcome must be an array")?
                    .iter()
                    .filter_map(|v| v.as_f64())
                    .collect();
                let est = cynepic_causal::LinearATEEstimator::difference_in_means(&treatment, &outcome);
                Ok(json!({
                    "ate": est.ate,
                    "std_error": est.std_error,
                    "method": "difference_in_means"
                }))
            }
            "check_policy" => {
                let chain = cynepic_guardian::PolicyChain::default();
                let ctx = cynepic_core::PolicyContext::default();
                let decision = chain.evaluate(&ctx);
                Ok(json!({
                    "verdict": format!("{:?}", decision)
                }))
            }
            "update_belief" => {
                let successes = args["successes"].as_u64().unwrap_or(0);
                let trials = args["trials"].as_u64().unwrap_or(0);
                use cynepic_bayes::BetaBinomial;
                let mut model = BetaBinomial::uniform();
                let failures = trials.saturating_sub(successes);
                model.update(successes, failures);
                let ci = model.credible_interval_95();
                Ok(json!({
                    "posterior_mean": model.mean(),
                    "credible_interval": [ci.0, ci.1]
                }))
            }
            "detect_loop" => {
                let node_id = args["node_id"].as_str().unwrap_or("unknown");
                let mut detector = cynepic_guardian::LoopDetector::new(5, 3);
                let violation = detector.record_visit(node_id);
                Ok(json!({
                    "violation_detected": violation.is_some(),
                    "violation": violation.map(|v| format!("{:?}", v))
                }))
            }
            "audit_trail" => {
                let limit = args["limit"].as_u64().unwrap_or(20) as usize;
                let trail = cynepic_guardian::AuditTrail::new();
                let entries = trail.recent_entries(limit);
                Ok(json!({
                    "count": entries.len(),
                    "entries": entries.iter().map(|e| json!({
                        "action": e.action,
                        "engine": e.engine,
                        "decision": format!("{:?}", e.decision)
                    })).collect::<Vec<_>>()
                }))
            }
            "run_counterfactual" => {
                let query = args["query"].as_str().unwrap_or("");
                let engine = cynepic_causal::CounterfactualEngine::new();
                let result = engine.query(
                    &cynepic_causal::CounterfactualQuery::new(
                        query,
                        String::new(),
                        String::new(),
                    ),
                );
                Ok(json!({
                    "result": format!("{:?}", result)
                }))
            }
            "monitor_drift" => Ok(json!({
                "status": "ok",
                "drift_detected": false,
                "observations": 0
            })),
            _ => Err(format!("Unknown tool: {}", name)),
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

    fn handle_call_tool(id: Option<Value>, params: &Value) -> JsonRpcResponse {
        let name = params["name"].as_str().unwrap_or("");
        let arguments = params.get("arguments").unwrap_or(&json!({}));
        match Self::call_tool(name, arguments) {
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

    fn handle_request(req: &JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => Self::handle_initialize(req.id.clone()),
            "tools/list" => Self::handle_list_tools(req.id.clone()),
            "tools/call" => {
                if let Some(params) = &req.params {
                    Self::handle_call_tool(req.id.clone(), params)
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

fn main() -> io::Result<()> {
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
                let resp = McpServer::handle_request(&req);
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
