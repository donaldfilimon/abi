use abi_capability::{Digest, source::PackageSource};
use serde_json::json;
pub fn source() -> PackageSource {
    serde_json::from_value(json!({
        "id":"memory.read", "version":"1.0.0",
        "input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false},
        "output_schema":{"type":"boolean"},"error_schema":{"type":"null"},
        "streaming":"None","cancellation":"BeforeEffect", "required_platform_permissions":[],
        "resource_scope":{"organization_id":"org","deployment_id":"deploy","tenant_id":"tenant","platform":"local","guild_ref":null,"resource_digest":Digest::from_bytes([1;32]),"subject_digest":Digest::from_bytes([2;32])},
        "side_effect":"None","reversibility":"ReversibleWithLoss","risk_class":"Low",
        "data_classes":["Operational"],"residency":"LocalOnly","retention":"Session",
        "preconditions":["fresh"],"invariants":["scoped"],"deadline_ms":1000,"rate_class":"bounded",
        "budgets":{"max_input_bytes":512,"max_output_bytes":512,"max_events":4,"max_calls":1},
        "confirmation_policy":"A0None","delegation_policy":"Forbidden","expiry_policy_ms":1000,"revocation_policy":"RecheckBeforeEffect","approval_policy":"A0None",
        "idempotency":"NaturallyIdempotent","compensation":null,"rollback":"None","postconditions":[],
        "adapter_bindings":[{"adapter_id":"recording","api_version":"1"}],
        "receipt_requirements":{"redacted":true,"require_postcondition_evidence":true},
        "fixtures":{"deterministic":Digest::from_bytes([3;32]),"adversarial":Digest::from_bytes([4;32]),"failure":Digest::from_bytes([5;32])},
        "promotion_thresholds":{"minimum_passed":1,"maximum_failed":0},
        "expiration_criteria":{"expires_at_ms":10000,"disable_on_schema_drift":true,"disable_on_missing_evidence":true}
    })).unwrap()
}
