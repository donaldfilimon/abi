//! Pre-schema and semantic refusal codes for the Abbey corpus.
//!
//! These run before and alongside JSON Schema validation so a fixture is
//! refused for the specific reason the corpus names, rather than a generic
//! schema failure.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::semantic_change;

pub(crate) fn privacy_code(value: &Value) -> Option<&'static str> {
    const FORBIDDEN: &[&str] = &[
        "audio",
        "transcript",
        "message",
        "prompt",
        "response_text",
        "credential",
        "token",
        "password",
        "username",
        "display_name",
        "filesystem_path",
        "participant_identity",
    ];
    match value {
        Value::Object(map) => {
            for (key, nested) in map {
                if FORBIDDEN.contains(&key.as_str()) || privacy_code(nested).is_some() {
                    return Some("forbidden_content");
                }
            }
        }
        Value::Array(items) => {
            if items.iter().any(|item| privacy_code(item).is_some()) {
                return Some("forbidden_content");
            }
        }
        Value::String(text) => {
            if (17..=20).contains(&text.len()) && text.bytes().all(|byte| byte.is_ascii_digit()) {
                return Some("forbidden_content");
            }
            if ["/Users/", "/home/", "C:\\", "sk-", "ghp_"]
                .iter()
                .any(|prefix| text.starts_with(prefix))
            {
                return Some("forbidden_content");
            }
        }
        _ => {}
    }
    None
}

pub(crate) fn pre_schema_code(schema: &str, document: &Value) -> Option<&'static str> {
    let map = document.as_object()?;
    if schema.ends_with("/learning/promotion-candidate.schema.json")
        && [
            "grant",
            "approval",
            "safety_policy_mutation",
            "command_registration",
            "platform_write",
            "direct_platform_write",
        ]
        .iter()
        .any(|key| map.contains_key(*key))
    {
        return Some("learning_authority_forbidden");
    }
    if schema.ends_with("/episode/proposal.schema.json")
        && map.get("priority_class").and_then(Value::as_str) == Some("MandatoryIncident")
        && (map.get("minimized").and_then(Value::as_bool) != Some(true)
            || map.get("redacted").and_then(Value::as_bool) != Some(true)
            || map.get("deletion_required").and_then(Value::as_bool) != Some(true)
            || map.get("deletion_key").and_then(Value::as_str).is_none()
            || map.get("retention_class").and_then(Value::as_str) != Some("mandatory_incident")
            || !matches!(
                map.get("hold_state").and_then(Value::as_str),
                Some("active" | "released")
            ))
    {
        return Some("mandatory_controls_missing");
    }
    None
}

pub(crate) fn semantic_code(schema: &str, document: &Value) -> Option<&'static str> {
    let map = document.as_object()?;
    if schema.ends_with("/identity/delegation-chain.schema.json") {
        let hops = map.get("hops")?.as_array()?;
        let mut seen = BTreeSet::new();
        if let Some(first) = hops.first().and_then(Value::as_object) {
            seen.insert(first.get("delegator_principal_id")?.as_str()?);
        }
        for pair in hops.windows(2) {
            let left = pair[0].as_object()?;
            let right = pair[1].as_object()?;
            if left.get("delegatee_principal_id") != right.get("delegator_principal_id") {
                return Some("delegation_chain_broken");
            }
        }
        for hop in hops {
            let delegatee = hop.as_object()?.get("delegatee_principal_id")?.as_str()?;
            if !seen.insert(delegatee) {
                return Some("delegation_cycle");
            }
        }
    }
    if schema.ends_with("/authorization/approval.schema.json")
        && map.get("approver_principal_id") == map.get("request_subject_principal_id")
    {
        return Some("self_approval");
    }
    if let Some(code) = semantic_change::code(schema, map) {
        return Some(code);
    }
    if schema.ends_with("/authorization/policy-decision.schema.json")
        && map.get("reason_code").and_then(Value::as_str) == Some("dependency_unavailable")
        && map.get("decision").and_then(Value::as_str) != Some("deny")
    {
        return Some("degraded_authority");
    }
    if schema.ends_with("/cognition/request.schema.json")
        && matches!(
            map.get("effect_class").and_then(Value::as_str),
            Some("durable_write" | "platform_effect")
        )
        && !map.contains_key("idempotency_key")
    {
        return Some("idempotency_required");
    }
    if schema.ends_with("/event/cancellation.schema.json")
        && map.get("cancellation_reference") != map.get("target_cancellation_reference")
    {
        return Some("cancellation_mismatch");
    }
    consent_semantic(schema, map).or_else(|| memory_learning_semantic(schema, map))
}

fn consent_semantic(schema: &str, map: &Map<String, Value>) -> Option<&'static str> {
    if schema.ends_with("/consent/transition.schema.json") {
        let transition = (
            map.get("from_state").and_then(Value::as_str),
            map.get("to_state").and_then(Value::as_str),
        );
        let valid = matches!(
            transition,
            (Some("Closed"), Some("PendingAttestation"))
                | (Some("PendingAttestation"), Some("Open"))
                | (Some("Open"), Some("Closing"))
                | (Some("Closing"), Some("Closed"))
        );
        if !valid {
            return Some(
                if matches!(
                    map.get("reason_code").and_then(Value::as_str),
                    Some(
                        "participant_change"
                            | "unidentified_participant"
                            | "attestation_lost"
                            | "manager_deauthorized"
                            | "connection_lost"
                            | "explicit_stop"
                    )
                ) {
                    "consent_close_required"
                } else {
                    "consent_transition_invalid"
                },
            );
        }
        if transition.1 == Some("Open")
            && (map.get("manager_authorized").and_then(Value::as_bool) != Some(true)
                || map
                    .get("all_current_participants_consented")
                    .and_then(Value::as_bool)
                    != Some(true)
                || map.get("participant_count").and_then(Value::as_u64) == Some(0))
        {
            return Some("consent_open_denied");
        }
        if transition.1 == Some("Closing") {
            let actual: BTreeSet<&str> = map
                .get("cancelled_stages")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .collect();
            let expected = BTreeSet::from([
                "decoded_receive",
                "stt",
                "reasoning",
                "synthesis",
                "provider",
                "playback",
            ]);
            if actual != expected {
                return Some("consent_cancellation_incomplete");
            }
        }
    }
    None
}

fn memory_learning_semantic(schema: &str, map: &Map<String, Value>) -> Option<&'static str> {
    if schema.ends_with("/episode/claim.schema.json") {
        let level = |field: &str| {
            map.get(field)
                .and_then(Value::as_str)
                .and_then(|text| text.strip_prefix('C'))
                .and_then(|text| text.parse::<u8>().ok())
        };
        if level("display_evidence_level") > level("evidence_level") {
            return Some("evidence_overclaim");
        }
    }
    if schema.ends_with("/learning/guild-learning-policy.schema.json") {
        if matches!(
            map.get("state").and_then(Value::as_str),
            Some("Unset" | "ExplicitDisabled")
        ) && map.get("adaptive_update_allowed").and_then(Value::as_bool) == Some(true)
        {
            return Some("learning_disabled");
        }
        if map.get("quiet_override").and_then(Value::as_bool) == Some(true)
            && map
                .get("unsolicited_action_allowed")
                .and_then(Value::as_bool)
                == Some(true)
        {
            return Some("quiet_override");
        }
    }
    None
}
