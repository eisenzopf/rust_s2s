//! Tool declarations: [`ToolSpec`], the [`ToolRegistry`] and system-prompt
//! rendering ([`ToolPromptRenderer`]).

use std::collections::HashMap;

use s2s_core::streams::special;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Where a tool runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolLocation {
    /// A handler registered with the [`Executor`](crate::Executor).
    #[default]
    Server,
    /// The client executes it: the server sends `tool.call` and waits for
    /// `tool.result {call_id, json}` (plan §4.2, §10.5).
    Client,
}

/// One declared tool, as sent in `session.start {tools[]}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name: 1–64 characters from `[A-Za-z0-9_.-]`.
    pub name: String,
    /// What the tool does, for the model.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the `arguments` object. Must be an object schema
    /// (`"type": "object"`). Defaults to an object with no properties.
    #[serde(default = "empty_object_schema")]
    pub parameters: Value,
    /// Server- or client-side execution.
    #[serde(default)]
    pub location: ToolLocation,
}

fn empty_object_schema() -> Value {
    serde_json::json!({"type": "object", "properties": {}})
}

impl ToolSpec {
    /// A server-side tool.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            location: ToolLocation::Server,
        }
    }

    /// A client-side tool.
    pub fn client(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            location: ToolLocation::Client,
            ..Self::new(name, description, parameters)
        }
    }
}

/// Errors registering a tool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// The name is empty, too long or has characters outside `[A-Za-z0-9_.-]`.
    InvalidName(String),
    /// A tool with this name is already registered.
    Duplicate(String),
    /// `parameters` is not a valid JSON Schema for an object.
    InvalidSchema {
        /// Tool name.
        tool: String,
        /// Why.
        reason: String,
    },
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidName(n) => write!(f, "invalid tool name {n:?}"),
            Self::Duplicate(n) => write!(f, "tool {n:?} is already registered"),
            Self::InvalidSchema { tool, reason } => {
                write!(f, "invalid parameter schema for tool {tool:?}: {reason}")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// Why a tool call's arguments were rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgumentError {
    /// No tool with this name is registered.
    UnknownTool(String),
    /// The arguments do not match the tool's schema; one message per
    /// violation (`"<message> at <json pointer>"`).
    Invalid(Vec<String>),
}

impl std::fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTool(n) => write!(f, "unknown tool {n:?}"),
            Self::Invalid(errs) => write!(f, "invalid arguments: {}", errs.join("; ")),
        }
    }
}

impl std::error::Error for ArgumentError {}

struct Entry {
    spec: ToolSpec,
    validator: jsonschema::Validator,
}

/// The set of tools declared for a session, with compiled argument
/// validators. Tools keep their registration order (which is also the order
/// they are rendered into the prompt and the grammar).
#[derive(Default)]
pub struct ToolRegistry {
    entries: Vec<Entry>,
    index: HashMap<String, usize>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.entries.iter().map(|e| &e.spec.name))
            .finish()
    }
}

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry holding `specs`, in order.
    pub fn from_specs(specs: impl IntoIterator<Item = ToolSpec>) -> Result<Self, RegistryError> {
        let mut r = Self::new();
        for s in specs {
            r.register(s)?;
        }
        Ok(r)
    }

    /// Add a tool, compiling its parameter schema.
    pub fn register(&mut self, spec: ToolSpec) -> Result<(), RegistryError> {
        if !valid_name(&spec.name) {
            return Err(RegistryError::InvalidName(spec.name));
        }
        if self.index.contains_key(&spec.name) {
            return Err(RegistryError::Duplicate(spec.name));
        }
        let invalid = |reason: String| RegistryError::InvalidSchema {
            tool: spec.name.clone(),
            reason,
        };
        if spec.parameters.get("type").and_then(Value::as_str) != Some("object") {
            return Err(invalid(
                "parameters must be an object schema with \"type\": \"object\"".into(),
            ));
        }
        let validator =
            jsonschema::validator_for(&spec.parameters).map_err(|e| invalid(e.to_string()))?;
        self.index.insert(spec.name.clone(), self.entries.len());
        self.entries.push(Entry { spec, validator });
        Ok(())
    }

    /// The tool called `name`.
    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.index.get(name).map(|&i| &self.entries[i].spec)
    }

    /// All tools, in registration order.
    pub fn specs(&self) -> impl Iterator<Item = &ToolSpec> + '_ {
        self.entries.iter().map(|e| &e.spec)
    }

    /// Number of tools.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check `arguments` against the schema of tool `name`.
    pub fn validate_arguments(&self, name: &str, arguments: &Value) -> Result<(), ArgumentError> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| ArgumentError::UnknownTool(name.to_string()))?;
        let errors: Vec<String> = self.entries[i]
            .validator
            .iter_errors(arguments)
            .map(|e| {
                let path = e.instance_path().to_string();
                if path.is_empty() {
                    e.to_string()
                } else {
                    format!("{e} at {path}")
                }
            })
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ArgumentError::Invalid(errors))
        }
    }

    /// Render the system prompt and tool list with `renderer`, wrapped in
    /// `<sys>` … `</sys>` as an injection segment (plan §3.5, §4.2).
    pub fn system_segment(&self, renderer: &dyn ToolPromptRenderer, system_prompt: &str) -> String {
        let specs: Vec<&ToolSpec> = self.specs().collect();
        format!(
            "{}{}{}",
            special::SYS,
            renderer.render(system_prompt, &specs),
            special::SYS_END
        )
    }
}

/// Renders the system prompt plus tool declarations into the text of the
/// session's `<sys>` injection segment.
///
/// The plan wants this in the base LLM's native function-calling format, to
/// reuse its prior. The base LLM is not chosen yet, so this is a trait;
/// [`DefaultPromptRenderer`] is a simple model-agnostic format.
pub trait ToolPromptRenderer {
    /// The segment body (without the `<sys>` markers).
    fn render(&self, system_prompt: &str, tools: &[&ToolSpec]) -> String;
}

/// The default, model-agnostic tool prompt format.
///
/// With no tools the output is just the system prompt. Otherwise:
///
/// ```text
/// {system prompt}
///
/// # Tools
///
/// You can call these tools. To call one, write
/// <tool_call>{"name": "<tool name>", "arguments": {<arguments>}}</tool_call>
/// and keep talking. Each result arrives later as
/// <tool_result>{"call_id": "...", "name": "...", "status": "...", ...}</tool_result>.
/// Write <tool_cancel> to abandon calls that are no longer needed.
///
/// ## get_weather
/// Current weather for a city.
/// Parameters: {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
/// ```
///
/// One `## name` section per tool, in registration order; `Parameters` is
/// the compact JSON of the schema. Client-side tools are rendered the same way
/// (the model need not know where a tool runs).
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultPromptRenderer;

impl ToolPromptRenderer for DefaultPromptRenderer {
    fn render(&self, system_prompt: &str, tools: &[&ToolSpec]) -> String {
        let mut out = system_prompt.trim_end().to_string();
        if tools.is_empty() {
            return out;
        }
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "# Tools\n\n\
             You can call these tools. To call one, write\n\
             {call}{{\"name\": \"<tool name>\", \"arguments\": {{<arguments>}}}}{end}\n\
             and keep talking. Each result arrives later as\n\
             {res}{{\"call_id\": \"...\", \"name\": \"...\", \"status\": \"...\", ...}}{res_end}.\n\
             Write {cancel} to abandon calls that are no longer needed.\n",
            call = special::TOOL_CALL,
            end = special::TOOL_CALL_END,
            res = special::TOOL_RESULT,
            res_end = special::TOOL_RESULT_END,
            cancel = special::TOOL_CANCEL,
        ));
        for t in tools {
            out.push_str(&format!("\n## {}\n", t.name));
            let desc = t.description.trim();
            if !desc.is_empty() {
                out.push_str(desc);
                out.push('\n');
            }
            out.push_str(&format!("Parameters: {}\n", t.parameters));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn weather() -> ToolSpec {
        ToolSpec::new(
            "get_weather",
            "Current weather for a city.",
            json!({
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}},
                "required": ["city"],
                "additionalProperties": false
            }),
        )
    }

    #[test]
    fn register_and_lookup() {
        let mut r = ToolRegistry::new();
        assert!(r.is_empty());
        r.register(weather()).unwrap();
        r.register(ToolSpec::client("open_url", "", json!({"type": "object"})))
            .unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.get("get_weather").unwrap().location, ToolLocation::Server);
        assert_eq!(r.get("open_url").unwrap().location, ToolLocation::Client);
        assert!(r.get("nope").is_none());
        let names: Vec<_> = r.specs().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["get_weather", "open_url"]);
        assert_eq!(format!("{r:?}"), r#"["get_weather", "open_url"]"#);
    }

    #[test]
    fn register_rejects_bad_input() {
        let mut r = ToolRegistry::new();
        r.register(weather()).unwrap();
        assert_eq!(
            r.register(weather()),
            Err(RegistryError::Duplicate("get_weather".into()))
        );
        for bad in ["", "has space", "ünï", &"x".repeat(65)] {
            let e = r
                .register(ToolSpec::new(bad, "", json!({"type": "object"})))
                .unwrap_err();
            assert_eq!(e, RegistryError::InvalidName(bad.to_string()));
        }
        let e = r
            .register(ToolSpec::new("t1", "", json!({"type": "string"})))
            .unwrap_err();
        assert!(matches!(e, RegistryError::InvalidSchema { .. }), "{e}");
        let e = r
            .register(ToolSpec::new(
                "t2",
                "",
                json!({"type": "object", "properties": {"a": {"type": 7}}}),
            ))
            .unwrap_err();
        assert!(matches!(e, RegistryError::InvalidSchema { .. }), "{e}");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn validate_arguments() {
        let r = ToolRegistry::from_specs([weather()]).unwrap();
        assert_eq!(
            r.validate_arguments("get_weather", &json!({"city": "Paris"})),
            Ok(())
        );
        assert_eq!(
            r.validate_arguments("nope", &json!({})),
            Err(ArgumentError::UnknownTool("nope".into()))
        );
        let Err(ArgumentError::Invalid(errs)) =
            r.validate_arguments("get_weather", &json!({"unit": "kelvin", "x": 1}))
        else {
            panic!("expected validation errors");
        };
        // Missing `city`, bad enum, and an extra property.
        assert_eq!(errs.len(), 3, "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("/unit")), "{errs:?}");
    }

    #[test]
    fn spec_serde_defaults() {
        let s: ToolSpec = serde_json::from_value(json!({"name": "ping"})).unwrap();
        assert_eq!(s.location, ToolLocation::Server);
        assert_eq!(s.parameters, json!({"type": "object", "properties": {}}));
        let s: ToolSpec =
            serde_json::from_value(json!({"name": "beep", "location": "client"})).unwrap();
        assert_eq!(s.location, ToolLocation::Client);
        let v = serde_json::to_value(weather()).unwrap();
        assert_eq!(v["location"], "server");
    }

    #[test]
    fn default_renderer_format() {
        let r = ToolRegistry::from_specs([weather()]).unwrap();
        let seg = r.system_segment(
            &DefaultPromptRenderer,
            "You are a helpful voice assistant.\n",
        );
        let expected = concat!(
            "<sys>You are a helpful voice assistant.\n\n",
            "# Tools\n\n",
            "You can call these tools. To call one, write\n",
            "<tool_call>{\"name\": \"<tool name>\", \"arguments\": {<arguments>}}</tool_call>\n",
            "and keep talking. Each result arrives later as\n",
            "<tool_result>{\"call_id\": \"...\", \"name\": \"...\", \"status\": \"...\", ...}</tool_result>.\n",
            "Write <tool_cancel> to abandon calls that are no longer needed.\n",
            "\n## get_weather\n",
            "Current weather for a city.\n",
            "Parameters: {\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},",
            "\"unit\":{\"enum\":[\"celsius\",\"fahrenheit\"]}},\"required\":[\"city\"],",
            "\"additionalProperties\":false}\n",
            "</sys>"
        );
        assert_eq!(seg, expected);
        // No tools: just the prompt.
        let empty = ToolRegistry::new();
        assert_eq!(
            empty.system_segment(&DefaultPromptRenderer, "Hi."),
            "<sys>Hi.</sys>"
        );
    }
}
