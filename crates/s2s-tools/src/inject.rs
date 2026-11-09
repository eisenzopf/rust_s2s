//! Formatting tool outcomes as `<tool_result>` injection segments (plan §3.5).
//!
//! Format (one line, compact JSON, keys in this order):
//!
//! ```text
//! <tool_result>{"call_id":"c_17","name":"get_weather","status":"ok","result":{"temp":18}}</tool_result>
//! <tool_result>{"call_id":"c_18","name":"book","status":"error","error":"no seats left"}</tool_result>
//! <tool_result>{"call_id":"c_19","name":"search","status":"timeout"}</tool_result>
//! <tool_result>{"call_id":"c_20","name":"search","status":"cancelled"}</tool_result>
//! ```
//!
//! A result too large for the budget becomes a JSON string holding a prefix
//! of its compact JSON, flagged with `"truncated":true` (the segment stays
//! valid JSON):
//!
//! ```text
//! <tool_result>{"call_id":"c_21","name":"search","status":"ok","truncated":true,"result":"[{\"title\":\"Fir"}</tool_result>
//! ```

use std::sync::Arc;

use s2s_core::streams::special;
use serde_json::{Map, Value};

use crate::executor::{ToolOutcome, ToolStatus};
use crate::parser::CallId;
use crate::tokenizer::TokenizerInfo;

/// Size limit for one injected segment, markers included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResultBudget {
    /// No limit.
    Unlimited,
    /// At most this many Unicode scalar values.
    Chars(usize),
    /// At most this many tokens: the two markers count one each, the JSON is
    /// counted with [`InjectionFormatter::with_tokenizer`]'s tokenizer (or
    /// estimated at 4 characters per token without one).
    Tokens(usize),
}

/// What to do with the outcome of a cancelled call (plan §4.2, "late results").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CancelledPolicy {
    /// Inject a short `cancelled` stub so the model knows.
    #[default]
    Stub,
    /// Inject nothing.
    Drop,
}

/// [`InjectionFormatter`] settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InjectionConfig {
    /// Size limit per segment. The plan budgets ≤ 512 tokens (one prefill
    /// between two frames); larger results should be summarized by the tool.
    pub budget: ResultBudget,
    /// Cancelled-call policy.
    pub cancelled: CancelledPolicy,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            budget: ResultBudget::Tokens(512),
            cancelled: CancelledPolicy::Stub,
        }
    }
}

/// Text to inject before the next frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InjectionSegment {
    /// The call the result belongs to.
    pub call_id: CallId,
    /// `<tool_result>{…}</tool_result>`.
    pub text: String,
    /// Whether the result or error message was cut to fit the budget.
    pub truncated: bool,
}

/// Formats [`ToolOutcome`]s as `<tool_result>` segments within a budget.
#[derive(Clone, Default)]
pub struct InjectionFormatter {
    config: InjectionConfig,
    tokenizer: Option<Arc<dyn TokenizerInfo>>,
}

impl std::fmt::Debug for InjectionFormatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InjectionFormatter")
            .field("config", &self.config)
            .field("has_tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

impl InjectionFormatter {
    /// A formatter with `config`.
    pub fn new(config: InjectionConfig) -> Self {
        Self {
            config,
            tokenizer: None,
        }
    }

    /// Count [`ResultBudget::Tokens`] with `tokenizer`.
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn TokenizerInfo>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// The settings.
    pub fn config(&self) -> &InjectionConfig {
        &self.config
    }

    /// Format an executor outcome. `None` when the policy drops it.
    pub fn format(&self, outcome: &ToolOutcome) -> Option<InjectionSegment> {
        self.format_status(&outcome.call_id, &outcome.name, &outcome.status)
    }

    /// Format a status for `call_id`. `None` when the policy drops it.
    ///
    /// If even the shortest form exceeds the budget (a tiny budget or a huge
    /// tool name), the shortest form is returned anyway.
    pub fn format_status(
        &self,
        call_id: &CallId,
        name: &str,
        status: &ToolStatus,
    ) -> Option<InjectionSegment> {
        if *status == ToolStatus::Cancelled && self.config.cancelled == CancelledPolicy::Drop {
            return None;
        }
        // The payload that may need truncating, as text.
        let payload: Option<(&str, String)> = match status {
            ToolStatus::Ok(v) => Some(("result", v.to_string())),
            ToolStatus::Error(e) => Some(("error", e.clone())),
            ToolStatus::Timeout | ToolStatus::Cancelled => None,
        };
        let build = |payload: Option<(&str, Value)>, truncated: bool| -> String {
            let mut m = Map::new();
            m.insert("call_id".into(), Value::String(call_id.0.clone()));
            m.insert("name".into(), Value::String(name.to_string()));
            m.insert("status".into(), Value::String(status.as_str().into()));
            if truncated {
                m.insert("truncated".into(), Value::Bool(true));
            }
            if let Some((k, v)) = payload {
                m.insert(k.into(), v);
            }
            format!(
                "{}{}{}",
                special::TOOL_RESULT,
                Value::Object(m),
                special::TOOL_RESULT_END
            )
        };
        let segment = |text, truncated| {
            Some(InjectionSegment {
                call_id: call_id.clone(),
                text,
                truncated,
            })
        };
        let Some((key, text)) = payload else {
            return segment(build(None, false), false);
        };
        let full_value = match status {
            ToolStatus::Ok(v) => v.clone(),
            _ => Value::String(text.clone()),
        };
        let full = build(Some((key, full_value)), false);
        if self.fits(&full) {
            return segment(full, false);
        }
        // Largest prefix (on a char boundary) that fits; `fits` is monotone
        // in the prefix length, so binary search over char counts.
        let bounds: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain([text.len()])
            .collect();
        let with_prefix = |n: usize| {
            build(
                Some((key, Value::String(text[..bounds[n]].to_string()))),
                true,
            )
        };
        let (mut lo, mut hi) = (0, bounds.len() - 1);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.fits(&with_prefix(mid)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        segment(with_prefix(lo), true)
    }

    fn fits(&self, segment: &str) -> bool {
        match self.config.budget {
            ResultBudget::Unlimited => true,
            ResultBudget::Chars(n) => segment.chars().count() <= n,
            ResultBudget::Tokens(n) => self.count_tokens(segment) <= n,
        }
    }

    /// Tokens `segment` takes (markers count one each).
    fn count_tokens(&self, segment: &str) -> usize {
        let body = segment
            .strip_prefix(special::TOOL_RESULT)
            .and_then(|s| s.strip_suffix(special::TOOL_RESULT_END))
            .unwrap_or(segment);
        2 + match &self.tokenizer {
            Some(t) => t.count_tokens(body),
            None => body.chars().count().div_ceil(4),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ByteTokenizer;
    use serde_json::json;
    use std::time::Duration;

    fn outcome(status: ToolStatus) -> ToolOutcome {
        ToolOutcome {
            call_id: "c_17".into(),
            name: "get_weather".into(),
            status,
            elapsed: Duration::ZERO,
        }
    }

    fn body(seg: &InjectionSegment) -> Value {
        let inner = seg
            .text
            .strip_prefix("<tool_result>")
            .and_then(|s| s.strip_suffix("</tool_result>"))
            .unwrap();
        serde_json::from_str(inner).unwrap()
    }

    #[test]
    fn formats_each_status() {
        let f = InjectionFormatter::default();
        let ok = f
            .format(&outcome(ToolStatus::Ok(
                json!({"temp": 18, "conditions": "sunny"}),
            )))
            .unwrap();
        assert_eq!(
            ok.text,
            "<tool_result>{\"call_id\":\"c_17\",\"name\":\"get_weather\",\"status\":\"ok\",\
             \"result\":{\"temp\":18,\"conditions\":\"sunny\"}}</tool_result>"
        );
        assert!(!ok.truncated);
        assert_eq!(ok.call_id, "c_17".into());
        let err = f
            .format(&outcome(ToolStatus::Error("city not found".into())))
            .unwrap();
        assert_eq!(
            err.text,
            "<tool_result>{\"call_id\":\"c_17\",\"name\":\"get_weather\",\"status\":\"error\",\
             \"error\":\"city not found\"}</tool_result>"
        );
        let to = f.format(&outcome(ToolStatus::Timeout)).unwrap();
        assert_eq!(
            to.text,
            "<tool_result>{\"call_id\":\"c_17\",\"name\":\"get_weather\",\"status\":\"timeout\"}</tool_result>"
        );
        let c = f.format(&outcome(ToolStatus::Cancelled)).unwrap();
        assert_eq!(body(&c)["status"], "cancelled");
        let drop = InjectionFormatter::new(InjectionConfig {
            cancelled: CancelledPolicy::Drop,
            ..InjectionConfig::default()
        });
        assert_eq!(drop.format(&outcome(ToolStatus::Cancelled)), None);
        assert!(drop.format(&outcome(ToolStatus::Timeout)).is_some());
    }

    #[test]
    fn truncates_to_char_budget() {
        let big = json!({"items": (0..200).map(|i| format!("item ñ {i}")).collect::<Vec<_>>()});
        for n in [120usize, 150, 333, 1000] {
            let f = InjectionFormatter::new(InjectionConfig {
                budget: ResultBudget::Chars(n),
                ..InjectionConfig::default()
            });
            let seg = f.format(&outcome(ToolStatus::Ok(big.clone()))).unwrap();
            assert!(seg.truncated);
            let len = seg.text.chars().count();
            assert!(len <= n, "{len} > {n}");
            // Maximal: within a few chars of the budget (one more char of
            // payload may need an escape).
            assert!(len + 8 > n, "{len} too short for {n}");
            let b = body(&seg);
            assert_eq!(b["truncated"], true);
            let prefix = b["result"].as_str().unwrap();
            assert!(big.to_string().starts_with(prefix));
        }
    }

    #[test]
    fn truncates_errors_and_respects_token_budget() {
        let tok: Arc<dyn TokenizerInfo> = Arc::new(ByteTokenizer::with_json_merges());
        let f = InjectionFormatter::new(InjectionConfig {
            budget: ResultBudget::Tokens(100),
            ..InjectionConfig::default()
        })
        .with_tokenizer(tok.clone());
        let msg = "the server returned an error ".repeat(20);
        let seg = f.format(&outcome(ToolStatus::Error(msg.clone()))).unwrap();
        assert!(seg.truncated);
        assert!(f.count_tokens(&seg.text) <= 100);
        let b = body(&seg);
        assert!(msg.starts_with(b["error"].as_str().unwrap()));
        assert!(!b["error"].as_str().unwrap().is_empty());
        // Small results are untouched.
        let seg = f.format(&outcome(ToolStatus::Ok(json!(1)))).unwrap();
        assert!(!seg.truncated);
        // Count = 2 markers + body tokens.
        let inner = seg
            .text
            .trim_start_matches("<tool_result>")
            .trim_end_matches("</tool_result>");
        assert_eq!(f.count_tokens(&seg.text), 2 + tok.count_tokens(inner));
        // Without a tokenizer, ~4 chars per token.
        let est = InjectionFormatter::new(InjectionConfig::default());
        assert_eq!(est.count_tokens("<tool_result>12345678</tool_result>"), 4);
    }

    #[test]
    fn tiny_budget_returns_shortest_form() {
        let f = InjectionFormatter::new(InjectionConfig {
            budget: ResultBudget::Chars(10),
            ..InjectionConfig::default()
        });
        let seg = f
            .format(&outcome(ToolStatus::Ok(json!([1, 2, 3]))))
            .unwrap();
        assert!(seg.truncated);
        assert_eq!(body(&seg)["result"], "");
        let seg = f.format(&outcome(ToolStatus::Timeout)).unwrap();
        assert!(!seg.truncated);
        let unlimited = InjectionFormatter::new(InjectionConfig {
            budget: ResultBudget::Unlimited,
            ..InjectionConfig::default()
        });
        let huge = json!("x".repeat(100_000));
        assert!(
            !unlimited
                .format(&outcome(ToolStatus::Ok(huge)))
                .unwrap()
                .truncated
        );
        assert!(format!("{unlimited:?}").contains("Unlimited"));
    }
}
