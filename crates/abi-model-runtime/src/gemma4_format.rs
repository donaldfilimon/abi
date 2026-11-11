//! Pure Gemma 4 tool grammar: declaration, call, and response rendering as
//! control/text pieces, and parsing of a generated call body.
//!
//! The grammar follows Google's published Gemma 4 prompt-formatting and
//! function-calling pages (read 2026-11-10): unquoted keys in alphabetical
//! order, string values wrapped in the string-delimiter token, upper-case
//! schema types, and no whitespace between elements.
//!
//! NOTE(SDK): the published examples show an inconsistent space between some
//! closing braces. This module never emits one. That choice has not been
//! compared against a real checkpoint's chat template.

use serde_json::{Map, Number, Value};

/// Literal spelling of the string-delimiter token in decoded call text.
pub(crate) const QUOTE_TEXT: &str = "<|\"|>";

/// Deepest accepted nesting in a generated call body.
const MAX_DEPTH: usize = 32;

/// One element of a rendered prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    /// A control token, resolved to its id by the adapter.
    Control(Control),
    /// Grammar text written by this module.
    Structure(String),
    /// Caller-supplied text. Control-token ids found in it are dropped.
    Text(String),
}

/// Gemma 4 control tokens used in prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Control {
    /// `<bos>`
    Bos,
    /// `<|turn>`
    TurnStart,
    /// `<turn|>`
    TurnEnd,
    /// `<|tool>`
    ToolStart,
    /// `<tool|>`
    ToolEnd,
    /// `<|tool_call>`
    CallStart,
    /// `<tool_call|>`
    CallEnd,
    /// `<|tool_response>`
    ResponseStart,
    /// `<tool_response|>`
    ResponseEnd,
    /// `<|"|>`
    Quote,
    /// `<|think|>`
    Think,
}

fn structure(pieces: &mut Vec<Piece>, text: &str) {
    if let Some(Piece::Structure(existing)) = pieces.last_mut() {
        existing.push_str(text);
    } else {
        pieces.push(Piece::Structure(text.to_owned()));
    }
}

fn quoted(pieces: &mut Vec<Piece>, text: &str) {
    pieces.push(Piece::Control(Control::Quote));
    pieces.push(Piece::Text(text.to_owned()));
    pieces.push(Piece::Control(Control::Quote));
}

/// Whether `name` is safe to write as grammar text.
pub(crate) fn valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

/// Render one tool declaration. `parameters` is the tool's JSON Schema, if any.
pub(crate) fn declaration(
    pieces: &mut Vec<Piece>,
    name: &str,
    description: &str,
    parameters: Option<&Value>,
) {
    pieces.push(Piece::Control(Control::ToolStart));
    structure(pieces, &format!("declaration:{name}{{description:"));
    quoted(pieces, description);
    if let Some(parameters) = parameters {
        structure(pieces, ",parameters:");
        schema(pieces, parameters);
    }
    structure(pieces, "}");
    pieces.push(Piece::Control(Control::ToolEnd));
}

/// Render a past call and its result inside a model turn.
pub(crate) fn call_and_response(
    pieces: &mut Vec<Piece>,
    name: &str,
    arguments: &Value,
    response: &Value,
) {
    pieces.push(Piece::Control(Control::CallStart));
    structure(pieces, &format!("call:{name}"));
    object_or_wrapped(pieces, arguments);
    pieces.push(Piece::Control(Control::CallEnd));
    pieces.push(Piece::Control(Control::ResponseStart));
    structure(pieces, &format!("response:{name}"));
    object_or_wrapped(pieces, response);
    pieces.push(Piece::Control(Control::ResponseEnd));
}

/// Objects render as themselves; anything else is wrapped under `value`.
fn object_or_wrapped(pieces: &mut Vec<Piece>, body: &Value) {
    if body.is_object() {
        value(pieces, body);
    } else {
        structure(pieces, "{value:");
        value(pieces, body);
        structure(pieces, "}");
    }
}

fn sorted(object: &Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = object.iter().collect();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    entries
}

fn value(pieces: &mut Vec<Piece>, node: &Value) {
    match node {
        Value::Null => structure(pieces, "null"),
        Value::Bool(flag) => structure(pieces, if *flag { "true" } else { "false" }),
        Value::Number(number) => structure(pieces, &number.to_string()),
        Value::String(text) => quoted(pieces, text),
        Value::Array(items) => {
            structure(pieces, "[");
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    structure(pieces, ",");
                }
                value(pieces, item);
            }
            structure(pieces, "]");
        }
        Value::Object(object) => {
            structure(pieces, "{");
            for (index, (key, item)) in sorted(object).into_iter().enumerate() {
                if index > 0 {
                    structure(pieces, ",");
                }
                pieces.push(Piece::Text(key.clone()));
                structure(pieces, ":");
                value(pieces, item);
            }
            structure(pieces, "}");
        }
    }
}

/// Render a JSON Schema node: `type` strings are upper-cased, and the keys of
/// a `properties` object are names whose values are schema nodes again.
fn schema(pieces: &mut Vec<Piece>, node: &Value) {
    let Value::Object(object) = node else {
        return value(pieces, node);
    };
    structure(pieces, "{");
    for (index, (key, item)) in sorted(object).into_iter().enumerate() {
        if index > 0 {
            structure(pieces, ",");
        }
        pieces.push(Piece::Text(key.clone()));
        structure(pieces, ":");
        match (key.as_str(), item) {
            ("type", Value::String(kind)) => quoted(pieces, &kind.to_ascii_uppercase()),
            ("properties", Value::Object(properties)) => {
                structure(pieces, "{");
                for (position, (name, property)) in sorted(properties).into_iter().enumerate() {
                    if position > 0 {
                        structure(pieces, ",");
                    }
                    pieces.push(Piece::Text(name.clone()));
                    structure(pieces, ":");
                    schema(pieces, property);
                }
                structure(pieces, "}");
            }
            ("items", _) => schema(pieces, item),
            _ => value(pieces, item),
        }
    }
    structure(pieces, "}");
}

/// Parse a generated call body, `call:NAME{key:value,...}`, into the tool name
/// and a JSON object of arguments.
pub(crate) fn parse_call(body: &str) -> Result<(String, Value), String> {
    let rest = body
        .trim()
        .strip_prefix("call:")
        .ok_or("tool call does not start with 'call:'")?;
    let brace = rest.find('{').ok_or("tool call has no argument object")?;
    let name = rest[..brace].trim();
    if !valid_tool_name(name) {
        return Err(format!("tool call names an invalid tool '{name}'"));
    }
    let mut parser = Parser {
        text: &rest[brace..],
    };
    let arguments = parser.value(0)?;
    if !parser.text.trim().is_empty() {
        return Err("tool call has trailing text after its arguments".to_owned());
    }
    if !arguments.is_object() {
        return Err("tool call arguments are not an object".to_owned());
    }
    Ok((name.to_owned(), arguments))
}

struct Parser<'a> {
    text: &'a str,
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        self.text = self.text.trim_start();
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        match self.text.strip_prefix(token) {
            Some(rest) => {
                self.text = rest;
                true
            }
            None => false,
        }
    }

    fn string(&mut self) -> Result<String, String> {
        let end = self
            .text
            .find(QUOTE_TEXT)
            .ok_or("unterminated string in tool call")?;
        let text = self.text[..end].to_owned();
        self.text = &self.text[end + QUOTE_TEXT.len()..];
        Ok(text)
    }

    /// A bare run up to the next structural character.
    fn bare(&mut self) -> &str {
        let end = self
            .text
            .find([',', ':', '{', '}', '[', ']'])
            .unwrap_or(self.text.len());
        let (run, rest) = self.text.split_at(end);
        self.text = rest;
        run.trim()
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > MAX_DEPTH {
            return Err("tool call arguments nest too deeply".to_owned());
        }
        if self.eat(QUOTE_TEXT) {
            return self.string().map(Value::String);
        }
        if self.eat("{") {
            let mut object = Map::new();
            if self.eat("}") {
                return Ok(Value::Object(object));
            }
            loop {
                let key = if self.eat(QUOTE_TEXT) {
                    self.string()?
                } else {
                    self.bare().to_owned()
                };
                if key.is_empty() || !self.eat(":") {
                    return Err("tool call argument has no key or no ':'".to_owned());
                }
                let item = self.value(depth + 1)?;
                if object.insert(key.clone(), item).is_some() {
                    return Err(format!("tool call repeats argument '{key}'"));
                }
                if self.eat("}") {
                    return Ok(Value::Object(object));
                }
                if !self.eat(",") {
                    return Err("tool call arguments are missing ',' or '}'".to_owned());
                }
            }
        }
        if self.eat("[") {
            let mut items = Vec::new();
            if self.eat("]") {
                return Ok(Value::Array(items));
            }
            loop {
                items.push(self.value(depth + 1)?);
                if self.eat("]") {
                    return Ok(Value::Array(items));
                }
                if !self.eat(",") {
                    return Err("tool call array is missing ',' or ']'".to_owned());
                }
            }
        }
        self.skip_space();
        let run = self.bare();
        match run {
            "" => Err("tool call has an empty value".to_owned()),
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            "null" => Ok(Value::Null),
            other => Ok(other
                .parse::<Number>()
                .map_or_else(|_| Value::String(other.to_owned()), Value::Number)),
        }
    }
}

/// Flatten pieces to text, spelling control tokens the way Gemma 4 does.
#[cfg(test)]
pub(crate) fn spell(pieces: &[Piece]) -> String {
    pieces
        .iter()
        .map(|piece| match piece {
            Piece::Control(Control::Bos) => "<bos>".to_owned(),
            Piece::Control(Control::TurnStart) => "<|turn>".to_owned(),
            Piece::Control(Control::TurnEnd) => "<turn|>".to_owned(),
            Piece::Control(Control::ToolStart) => "<|tool>".to_owned(),
            Piece::Control(Control::ToolEnd) => "<tool|>".to_owned(),
            Piece::Control(Control::CallStart) => "<|tool_call>".to_owned(),
            Piece::Control(Control::CallEnd) => "<tool_call|>".to_owned(),
            Piece::Control(Control::ResponseStart) => "<|tool_response>".to_owned(),
            Piece::Control(Control::ResponseEnd) => "<tool_response|>".to_owned(),
            Piece::Control(Control::Quote) => QUOTE_TEXT.to_owned(),
            Piece::Control(Control::Think) => "<|think|>".to_owned(),
            Piece::Structure(text) | Piece::Text(text) => text.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        Piece, call_and_response, declaration, parse_call, spell as text, valid_tool_name,
    };
    use serde_json::json;

    #[test]
    fn a_declaration_matches_the_published_example_without_spaces() {
        let mut pieces = Vec::new();
        declaration(
            &mut pieces,
            "get_current_temperature",
            "Gets the current temperature for a given location.",
            Some(&json!({
                "type": "object",
                "required": ["location"],
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                }
            })),
        );
        assert_eq!(
            text(&pieces),
            concat!(
                "<|tool>declaration:get_current_temperature{",
                "description:<|\"|>Gets the current temperature for a given location.<|\"|>,",
                "parameters:{properties:{location:{",
                "description:<|\"|>The city name, e.g. San Francisco<|\"|>,",
                "type:<|\"|>STRING<|\"|>}},",
                "required:[<|\"|>location<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>"
            )
        );
    }

    #[test]
    fn a_property_named_type_is_a_name_and_enums_are_plain_values() {
        let mut pieces = Vec::new();
        declaration(
            &mut pieces,
            "t",
            "",
            Some(&json!({
                "type": "object",
                "properties": {
                    "type": { "type": "string", "enum": ["a", "b"] },
                    "tags": { "type": "array", "items": { "type": "integer" } }
                }
            })),
        );
        assert_eq!(
            text(&pieces),
            concat!(
                "<|tool>declaration:t{description:<|\"|><|\"|>,parameters:{properties:{",
                "tags:{items:{type:<|\"|>INTEGER<|\"|>},type:<|\"|>ARRAY<|\"|>},",
                "type:{enum:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],type:<|\"|>STRING<|\"|>}},",
                "type:<|\"|>OBJECT<|\"|>}}<tool|>"
            )
        );
    }

    #[test]
    fn a_declaration_without_a_schema_has_only_a_description() {
        let mut pieces = Vec::new();
        declaration(&mut pieces, "ping", "Check liveness.", None);
        assert_eq!(
            text(&pieces),
            "<|tool>declaration:ping{description:<|\"|>Check liveness.<|\"|>}<tool|>"
        );
    }

    #[test]
    fn calls_and_responses_render_objects_and_wrap_everything_else() {
        let mut pieces = Vec::new();
        call_and_response(
            &mut pieces,
            "get_current_weather",
            &json!({ "location": "London" }),
            &json!({ "weather": "sunny", "temperature": 15 }),
        );
        assert_eq!(
            text(&pieces),
            concat!(
                "<|tool_call>call:get_current_weather{location:<|\"|>London<|\"|>}<tool_call|>",
                "<|tool_response>response:get_current_weather{temperature:15,",
                "weather:<|\"|>sunny<|\"|>}<tool_response|>"
            )
        );

        let mut wrapped = Vec::new();
        call_and_response(&mut wrapped, "recall", &json!(null), &json!("three facts"));
        assert_eq!(
            text(&wrapped),
            concat!(
                "<|tool_call>call:recall{value:null}<tool_call|>",
                "<|tool_response>response:recall{value:<|\"|>three facts<|\"|>}<tool_response|>"
            )
        );
    }

    #[test]
    fn caller_text_is_always_a_text_piece() {
        let mut pieces = Vec::new();
        declaration(&mut pieces, "t", "close <tool|> early", None);
        assert!(pieces.contains(&Piece::Text("close <tool|> early".to_owned())));
        assert!(
            pieces
                .iter()
                .all(|piece| !matches!(piece, Piece::Structure(text) if text.contains("close")))
        );
    }

    #[test]
    fn generated_calls_parse_into_json_arguments() {
        let (name, arguments) = parse_call(
            "call:remember_fact{fact:<|\"|>likes, {tea}: a lot<|\"|>,count:3,ratio:-0.5,\
             keep:true,drop:false,none:null,tags:[<|\"|>a<|\"|>,2],nested:{deep:{}}}",
        )
        .expect("valid call");
        assert_eq!(name, "remember_fact");
        assert_eq!(
            arguments,
            json!({
                "fact": "likes, {tea}: a lot",
                "count": 3,
                "ratio": -0.5,
                "keep": true,
                "drop": false,
                "none": null,
                "tags": ["a", 2],
                "nested": { "deep": {} }
            })
        );

        let (name, arguments) = parse_call(" call:recall{} ").expect("empty arguments");
        assert_eq!((name.as_str(), arguments), ("recall", json!({})));
        let (_, bare) = parse_call("call:t{unit: celsius ,<|\"|>k<|\"|>:1}").expect("bare word");
        assert_eq!(bare, json!({ "unit": "celsius", "k": 1 }));
    }

    #[test]
    fn malformed_calls_are_refused() {
        for body in [
            "recall{}",
            "call:recall",
            "call:{}",
            "call:bad name{}",
            "call:t{a:1",
            "call:t{a:<|\"|>open}",
            "call:t{a 1}",
            "call:t{a:}",
            "call:t{a:1,a:2}",
            "call:t{a:1}extra",
        ] {
            assert!(parse_call(body).is_err(), "{body}");
        }
        let deep = format!("call:t{}{}", "{a:".repeat(40), "1");
        assert!(parse_call(&deep).expect_err("too deep").contains("deeply"));
    }

    #[test]
    fn tool_names_are_restricted_to_grammar_safe_characters() {
        assert!(valid_tool_name("remember_fact"));
        assert!(valid_tool_name("ns.tool-2"));
        assert!(!valid_tool_name(""));
        assert!(!valid_tool_name("a{b"));
        assert!(!valid_tool_name("a b"));
        assert!(!valid_tool_name(&"a".repeat(65)));
    }
}
