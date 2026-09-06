//! JSON Canonicalization Scheme (`abbey-jcs-v1`) encoding.
//!
//! Bounded, domain-prefixed canonical bytes for authority objects. The profile
//! deliberately rejects any floating-point value that is not a safe integer, so
//! a canonical form never depends on binary64 rounding.

use serde_json::{Number, Value};

use crate::ContractError;

/// Largest integer representable exactly in IEEE-754 binary64.
const JCS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Canonicalize a bounded authority object under the `abbey-jcs-v1` profile.
///
/// The returned bytes include the schema-family/major domain prefix. This
/// profile accepts strings, booleans, null, arrays, objects, safe integers, and
/// negative zero (normalized to zero). Other floating-point values are rejected
/// before canonicalization.
pub fn canonicalize_jcs(
    schema_family: &str,
    major: u32,
    value: &Value,
) -> Result<Vec<u8>, ContractError> {
    if schema_family.is_empty()
        || schema_family.len() > 64
        || !schema_family.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
    {
        return Err(ContractError::PathInvalid {
            path: "schema_family".to_owned(),
        });
    }
    let mut output = format!("abbey-jcs-v1:{schema_family}:{major}\0").into_bytes();
    canonical_value(value, &mut output)?;
    Ok(output)
}

fn canonical_value(value: &Value, output: &mut Vec<u8>) -> Result<(), ContractError> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(flag) => output.extend_from_slice(if *flag { b"true" } else { b"false" }),
        Value::String(text) => output.extend_from_slice(
            serde_json::to_string(text)
                .expect("serializing a string is infallible")
                .as_bytes(),
        ),
        Value::Number(number) => canonical_number(number, output)?,
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                canonical_value(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(map) => {
            output.push(b'{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by_key(|key| key.encode_utf16().collect::<Vec<_>>());
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend_from_slice(
                    serde_json::to_string(key)
                        .expect("serializing a key is infallible")
                        .as_bytes(),
                );
                output.push(b':');
                canonical_value(&map[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn canonical_number(number: &Number, output: &mut Vec<u8>) -> Result<(), ContractError> {
    if let Some(integer) = number.as_i64() {
        if integer.unsigned_abs() > JCS_SAFE_INTEGER {
            return Err(ContractError::NumericDomain {
                path: "number".to_owned(),
            });
        }
        output.extend_from_slice(integer.to_string().as_bytes());
        return Ok(());
    }
    if let Some(integer) = number.as_u64() {
        if integer > JCS_SAFE_INTEGER {
            return Err(ContractError::NumericDomain {
                path: "number".to_owned(),
            });
        }
        output.extend_from_slice(integer.to_string().as_bytes());
        return Ok(());
    }
    if number.as_f64().is_some_and(|float| float == 0.0) {
        output.push(b'0');
        return Ok(());
    }
    Err(ContractError::NumericDomain {
        path: "number".to_owned(),
    })
}
