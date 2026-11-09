//! Tool-calling accuracy (plan §11: selection ≥ 90%, argument exact match
//! ≥ 85%, JSON validity 100% with the grammar).
//!
//! The unit of evaluation is an **example** (one point in a dialogue where
//! the model may call tools): its reference calls (possibly none) and the
//! calls the model emitted there (possibly none, possibly invalid).
//!
//! # Matching
//!
//! Within an example, reference calls are matched in order to unused
//! **valid** predicted calls: first to one with the same name and equal
//! arguments, then (for those still unmatched) to one with the same name.
//! Remaining reference calls are *missed*; remaining predicted calls
//! (valid or not) are *spurious*.
//!
//! Arguments are compared as JSON values: object key order is irrelevant,
//! numbers are equal when numerically equal (`1` = `1.0`), strings must be
//! identical.
//!
//! # Metrics
//!
//! | Metric | Definition |
//! |---|---|
//! | tool selection accuracy | reference calls matched by name / reference calls |
//! | argument exact match | reference calls matched by name **and** arguments / reference calls |
//! | argument exact match given name | exact matches / name matches |
//! | per-argument accuracy | over name-matched pairs: top-level reference arguments present in the prediction with an equal value / top-level reference arguments |
//! | per-argument precision | the same matches / top-level predicted arguments |
//! | JSON validity rate | valid predicted calls / predicted calls |
//! | no-call precision / recall | "no call" as the positive class per example: predicted no call and none needed / predicted no call; / no call needed |
//! | example exact match | examples whose predicted calls are exactly the reference calls (all matched with equal arguments, nothing spurious, nothing invalid) / examples |
//!
//! A predicted call is *valid* when it is a JSON object
//! `{"name": <string>, "arguments": <object>}` (the span shape the runtime
//! parser accepts); schema validation is the runtime's job and is not
//! repeated here.
//!
//! # JSONL format (CLI)
//!
//! One example per line, `{"id": "…", "calls": [ … ]}`. A call is
//! `{"name": "get_weather", "arguments": {"city": "Paris"}}`; `arguments`
//! may also be a JSON-encoded string (OpenAI style). In hypothesis files a
//! call may instead be `{"raw": "<span text>"}`, the text the model wrote
//! between `<tool_call>` and `</tool_call>`, which is parsed here (and
//! counts as invalid if it is not a well-formed call).

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::ratio;

/// A tool call: name and JSON arguments.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallRecord {
    /// Tool name.
    pub name: String,
    /// Arguments (normally an object).
    pub arguments: Value,
}

impl ToolCallRecord {
    /// A call record.
    pub fn new(name: impl Into<String>, arguments: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

impl From<&s2s_tools::ToolCall> for ToolCallRecord {
    fn from(c: &s2s_tools::ToolCall) -> Self {
        Self::new(c.name.clone(), c.arguments.clone())
    }
}

/// A call the model emitted.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictedCall {
    /// A well-formed call.
    Valid(ToolCallRecord),
    /// Text that is not a well-formed call.
    Invalid {
        /// What was emitted.
        raw: String,
    },
}

impl PredictedCall {
    /// Parse the span text the model wrote between `<tool_call>` and
    /// `</tool_call>`.
    pub fn parse_span(raw: &str) -> Self {
        match serde_json::from_str::<Value>(raw) {
            Ok(Value::Object(obj)) => match (obj.get("name"), obj.get("arguments")) {
                (Some(Value::String(name)), Some(args @ Value::Object(_))) => {
                    Self::Valid(ToolCallRecord::new(name.clone(), args.clone()))
                }
                _ => Self::invalid(raw),
            },
            _ => Self::invalid(raw),
        }
    }

    /// A call from its JSONL form (see the module docs).
    pub fn from_json(v: &Value) -> Self {
        if let Some(raw) = v.get("raw").and_then(Value::as_str) {
            return Self::parse_span(raw);
        }
        let Some(name) = v.get("name").and_then(Value::as_str) else {
            return Self::invalid(&v.to_string());
        };
        let args = match v.get("arguments") {
            None | Some(Value::Null) => Some(Value::Object(Default::default())),
            Some(a @ Value::Object(_)) => Some(a.clone()),
            Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                .ok()
                .filter(Value::is_object),
            Some(_) => None,
        };
        match args {
            Some(arguments) => Self::Valid(ToolCallRecord::new(name, arguments)),
            None => Self::invalid(&v.to_string()),
        }
    }

    fn invalid(raw: &str) -> Self {
        Self::Invalid {
            raw: raw.to_string(),
        }
    }

    /// The call, if valid.
    pub fn as_valid(&self) -> Option<&ToolCallRecord> {
        match self {
            Self::Valid(c) => Some(c),
            Self::Invalid { .. } => None,
        }
    }
}

impl From<&s2s_tools::ToolCall> for PredictedCall {
    fn from(c: &s2s_tools::ToolCall) -> Self {
        Self::Valid(c.into())
    }
}

/// One evaluation point.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ToolExample {
    /// Optional identifier.
    pub id: Option<String>,
    /// Calls that should have been made (empty: no call needed).
    pub reference: Vec<ToolCallRecord>,
    /// Calls the model made.
    pub predicted: Vec<PredictedCall>,
}

/// JSON equality with numeric comparison of numbers (see the module docs).
pub fn json_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            if let (Some(x), Some(y)) = (x.as_i64(), y.as_i64()) {
                x == y
            } else if let (Some(x), Some(y)) = (x.as_u64(), y.as_u64()) {
                x == y
            } else {
                x.as_f64() == y.as_f64()
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| json_equal(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_equal(v, w)))
        }
        _ => a == b,
    }
}

/// Matches of one argument name (`tool.argument`) across name-matched pairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ArgumentStat {
    /// Times the prediction had the argument with an equal value.
    pub matched: usize,
    /// Times the reference had the argument.
    pub total: usize,
}

/// Raw counts behind a [`ToolReport`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ToolCounts {
    /// Examples.
    pub examples: usize,
    /// Reference calls.
    pub reference_calls: usize,
    /// Predicted calls (valid or not).
    pub predicted_calls: usize,
    /// Valid predicted calls.
    pub valid_calls: usize,
    /// Reference calls matched by name (including exact matches).
    pub name_matches: usize,
    /// Reference calls matched by name and arguments.
    pub exact_matches: usize,
    /// Reference calls without a match.
    pub missed_calls: usize,
    /// Predicted calls without a match (valid or not).
    pub spurious_calls: usize,
    /// Top-level reference arguments in name-matched pairs.
    pub reference_arguments: usize,
    /// Top-level predicted arguments in name-matched pairs.
    pub predicted_arguments: usize,
    /// Of those, present in both with equal values.
    pub matched_arguments: usize,
    /// Examples needing no call where the model made none.
    pub no_call_true_positives: usize,
    /// Examples needing calls where the model made none.
    pub no_call_false_positives: usize,
    /// Examples needing no call where the model made one.
    pub no_call_false_negatives: usize,
    /// Examples predicted exactly.
    pub exact_examples: usize,
}

/// Tool-calling metrics (see the [module docs](self)). Rates are `None`
/// when their denominator is zero.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolReport {
    /// The counts.
    pub counts: ToolCounts,
    /// Name matches / reference calls.
    pub tool_selection_accuracy: Option<f64>,
    /// Exact matches / reference calls.
    pub argument_exact_match: Option<f64>,
    /// Exact matches / name matches.
    pub argument_exact_match_given_name: Option<f64>,
    /// Matched arguments / reference arguments (name-matched pairs).
    pub per_argument_accuracy: Option<f64>,
    /// Matched arguments / predicted arguments (name-matched pairs).
    pub per_argument_precision: Option<f64>,
    /// Valid calls / predicted calls.
    pub json_validity_rate: Option<f64>,
    /// Precision of predicting "no call".
    pub no_call_precision: Option<f64>,
    /// Recall of "no call needed".
    pub no_call_recall: Option<f64>,
    /// Exactly predicted examples / examples.
    pub example_exact_match: Option<f64>,
    /// Per `tool.argument` statistics.
    pub per_argument: BTreeMap<String, ArgumentStat>,
}

/// Accumulates tool-calling metrics over examples.
#[derive(Clone, Debug, Default)]
pub struct ToolEvaluator {
    counts: ToolCounts,
    per_argument: BTreeMap<String, ArgumentStat>,
}

fn top_level(v: &Value) -> Vec<(String, &Value)> {
    match v {
        Value::Object(m) => m.iter().map(|(k, v)| (k.clone(), v)).collect(),
        other => vec![(String::new(), other)],
    }
}

impl ToolEvaluator {
    /// An empty evaluator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Score one example.
    pub fn add(&mut self, ex: &ToolExample) {
        let c = &mut self.counts;
        c.examples += 1;
        c.reference_calls += ex.reference.len();
        c.predicted_calls += ex.predicted.len();
        c.valid_calls += ex
            .predicted
            .iter()
            .filter(|p| p.as_valid().is_some())
            .count();
        match (ex.reference.is_empty(), ex.predicted.is_empty()) {
            (true, true) => c.no_call_true_positives += 1,
            (false, true) => c.no_call_false_positives += 1,
            (true, false) => c.no_call_false_negatives += 1,
            (false, false) => {}
        }

        // Matching: exact first, then by name.
        let mut used = vec![false; ex.predicted.len()];
        let mut matched: Vec<Option<(usize, bool)>> = vec![None; ex.reference.len()];
        for (ri, r) in ex.reference.iter().enumerate() {
            let hit = ex.predicted.iter().enumerate().position(|(pi, p)| {
                !used[pi]
                    && p.as_valid()
                        .is_some_and(|p| p.name == r.name && json_equal(&p.arguments, &r.arguments))
            });
            if let Some(pi) = hit {
                used[pi] = true;
                matched[ri] = Some((pi, true));
            }
        }
        for (ri, r) in ex.reference.iter().enumerate() {
            if matched[ri].is_some() {
                continue;
            }
            let hit =
                ex.predicted.iter().enumerate().position(|(pi, p)| {
                    !used[pi] && p.as_valid().is_some_and(|p| p.name == r.name)
                });
            if let Some(pi) = hit {
                used[pi] = true;
                matched[ri] = Some((pi, false));
            }
        }

        for (r, m) in ex.reference.iter().zip(&matched) {
            let Some((pi, exact)) = *m else {
                c.missed_calls += 1;
                continue;
            };
            c.name_matches += 1;
            c.exact_matches += usize::from(exact);
            let p = ex.predicted[pi]
                .as_valid()
                .expect("matched calls are valid");
            let pred_args = top_level(&p.arguments);
            c.predicted_arguments += pred_args.len();
            for (k, v) in top_level(&r.arguments) {
                c.reference_arguments += 1;
                let ok = pred_args
                    .iter()
                    .any(|(pk, pv)| *pk == k && json_equal(pv, v));
                c.matched_arguments += usize::from(ok);
                let stat = self
                    .per_argument
                    .entry(format!("{}.{k}", r.name))
                    .or_default();
                stat.total += 1;
                stat.matched += usize::from(ok);
            }
        }
        let spurious = used.iter().filter(|u| !**u).count();
        c.spurious_calls += spurious;
        let exact = matched.iter().all(|m| matches!(m, Some((_, true)))) && spurious == 0;
        c.exact_examples += usize::from(exact);
    }

    /// The metrics so far.
    pub fn report(&self) -> ToolReport {
        let c = self.counts;
        ToolReport {
            counts: c,
            tool_selection_accuracy: ratio(c.name_matches, c.reference_calls),
            argument_exact_match: ratio(c.exact_matches, c.reference_calls),
            argument_exact_match_given_name: ratio(c.exact_matches, c.name_matches),
            per_argument_accuracy: ratio(c.matched_arguments, c.reference_arguments),
            per_argument_precision: ratio(c.matched_arguments, c.predicted_arguments),
            json_validity_rate: ratio(c.valid_calls, c.predicted_calls),
            no_call_precision: ratio(
                c.no_call_true_positives,
                c.no_call_true_positives + c.no_call_false_positives,
            ),
            no_call_recall: ratio(
                c.no_call_true_positives,
                c.no_call_true_positives + c.no_call_false_negatives,
            ),
            example_exact_match: ratio(c.exact_examples, c.examples),
            per_argument: self.per_argument.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, args: Value) -> ToolCallRecord {
        ToolCallRecord::new(name, args)
    }

    #[test]
    fn json_equality() {
        assert!(json_equal(&json!(1), &json!(1.0)));
        assert!(json_equal(&json!(-3), &json!(-3.0)));
        assert!(!json_equal(&json!(1), &json!(1.5)));
        assert!(json_equal(
            &json!({"a": 1, "b": [1, {"c": 2.0}]}),
            &json!({"b": [1.0, {"c": 2}], "a": 1})
        ));
        assert!(!json_equal(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
        assert!(!json_equal(&json!([1, 2]), &json!([2, 1])));
        assert!(!json_equal(&json!("1"), &json!(1)));
        assert!(json_equal(&json!(u64::MAX), &json!(u64::MAX)));
    }

    #[test]
    fn parsing_predictions() {
        let v = PredictedCall::parse_span(r#"{"name": "f", "arguments": {"x": 1}}"#);
        assert_eq!(v, PredictedCall::Valid(call("f", json!({"x": 1}))));
        for bad in [
            r#"{"name": "f", "arguments": {"x": 1}"#,
            r#"{"name": "f"}"#,
            r#"{"name": 3, "arguments": {}}"#,
            r#"{"name": "f", "arguments": [1]}"#,
            "[]",
            "",
        ] {
            assert!(PredictedCall::parse_span(bad).as_valid().is_none(), "{bad}");
        }
        assert_eq!(
            PredictedCall::from_json(&json!({"name": "f", "arguments": "{\"x\": 2}"})),
            PredictedCall::Valid(call("f", json!({"x": 2})))
        );
        assert_eq!(
            PredictedCall::from_json(&json!({"name": "f"})),
            PredictedCall::Valid(call("f", json!({})))
        );
        assert!(
            PredictedCall::from_json(&json!({"name": "f", "arguments": "{oops"}))
                .as_valid()
                .is_none()
        );
        assert!(PredictedCall::from_json(&json!({"arguments": {}}))
            .as_valid()
            .is_none());
        assert_eq!(
            PredictedCall::from_json(&json!({"raw": "{\"name\": \"g\", \"arguments\": {}}"})),
            PredictedCall::Valid(call("g", json!({})))
        );
        let tc = s2s_tools::ToolCall {
            call_id: s2s_tools::CallId::from("c_1"),
            name: "f".into(),
            arguments: json!({"a": true}),
        };
        assert_eq!(
            PredictedCall::from(&tc),
            PredictedCall::Valid(call("f", json!({"a": true})))
        );
    }

    #[test]
    fn hand_computed_metrics() {
        let w = |city: &str, unit: &str| call("get_weather", json!({"city": city, "unit": unit}));
        let examples = vec![
            // 1. Exact.
            ToolExample {
                id: Some("1".into()),
                reference: vec![w("Paris", "celsius")],
                predicted: vec![PredictedCall::Valid(w("Paris", "celsius"))],
            },
            // 2. Right tool, one of two arguments wrong.
            ToolExample {
                id: Some("2".into()),
                reference: vec![w("Paris", "celsius")],
                predicted: vec![PredictedCall::Valid(w("Paris", "fahrenheit"))],
            },
            // 3. Wrong tool.
            ToolExample {
                id: None,
                reference: vec![w("Oslo", "celsius")],
                predicted: vec![PredictedCall::Valid(call(
                    "set_timer",
                    json!({"seconds": 5}),
                ))],
            },
            // 4. Invalid JSON where a call was needed.
            ToolExample {
                id: None,
                reference: vec![call("set_timer", json!({"seconds": 60}))],
                predicted: vec![PredictedCall::parse_span(
                    "{\"name\": \"set_timer\", \"argu",
                )],
            },
            // 5. No call needed, none made.
            ToolExample::default(),
            // 6. No call needed, one made.
            ToolExample {
                id: None,
                reference: vec![],
                predicted: vec![PredictedCall::Valid(call(
                    "set_timer",
                    json!({"seconds": 1}),
                ))],
            },
            // 7. Call needed, none made.
            ToolExample {
                id: None,
                reference: vec![call("set_timer", json!({"seconds": 10}))],
                predicted: vec![],
            },
            // 8. Two calls, out of order, one with an extra argument and
            //    1 vs 1.0, plus a spurious duplicate.
            ToolExample {
                id: None,
                reference: vec![
                    call("set_timer", json!({"seconds": 1})),
                    w("Rome", "celsius"),
                ],
                predicted: vec![
                    PredictedCall::Valid(w("Rome", "celsius")),
                    PredictedCall::Valid(call("set_timer", json!({"seconds": 1.0, "loud": true}))),
                    PredictedCall::Valid(w("Rome", "celsius")),
                ],
            },
        ];
        let mut ev = ToolEvaluator::new();
        for ex in &examples {
            ev.add(ex);
        }
        let r = ev.report();
        let c = r.counts;
        assert_eq!(c.examples, 8);
        // Reference calls: 1+1+1+1+0+0+1+2 = 7.
        assert_eq!(c.reference_calls, 7);
        // Predicted: 1+1+1+1+0+1+0+3 = 8, of which 7 valid.
        assert_eq!((c.predicted_calls, c.valid_calls), (8, 7));
        // Name matches: ex1, ex2, ex8 (both) = 4. Exact: ex1, ex8 Rome = 2.
        assert_eq!((c.name_matches, c.exact_matches), (4, 2));
        // Missed: ex3, ex4, ex7 = 3. Spurious: ex3 set_timer, ex4 invalid,
        // ex6 set_timer, ex8 duplicate = 4.
        assert_eq!((c.missed_calls, c.spurious_calls), (3, 4));
        assert_eq!(r.tool_selection_accuracy, Some(4.0 / 7.0));
        assert_eq!(r.argument_exact_match, Some(2.0 / 7.0));
        assert_eq!(r.argument_exact_match_given_name, Some(0.5));
        // Arguments over name-matched pairs: ex1 2/2, ex2 1/2, ex8 timer 1/1
        // (pred has 2), ex8 Rome 2/2 → matched 6 of 7 reference, 8 predicted.
        assert_eq!(
            (
                c.matched_arguments,
                c.reference_arguments,
                c.predicted_arguments
            ),
            (6, 7, 8)
        );
        assert_eq!(r.per_argument_accuracy, Some(6.0 / 7.0));
        assert_eq!(r.per_argument_precision, Some(6.0 / 8.0));
        assert_eq!(r.json_validity_rate, Some(7.0 / 8.0));
        // No call: TP ex5, FP ex7, FN ex6.
        assert_eq!(r.no_call_precision, Some(0.5));
        assert_eq!(r.no_call_recall, Some(0.5));
        // Exact examples: ex1 and ex5.
        assert_eq!(r.example_exact_match, Some(2.0 / 8.0));
        assert_eq!(
            r.per_argument["get_weather.unit"],
            ArgumentStat {
                matched: 2,
                total: 3
            }
        );
        assert_eq!(
            r.per_argument["set_timer.seconds"],
            ArgumentStat {
                matched: 1,
                total: 1
            }
        );
        let empty = ToolEvaluator::new().report();
        assert_eq!(empty.tool_selection_accuracy, None);
        assert_eq!(empty.json_validity_rate, None);
    }
}
