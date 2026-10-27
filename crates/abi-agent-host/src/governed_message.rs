//! Governed, single-shot outbound message capability (Discord channel, Twilio SMS).
//!
//! The approval binds the exact channel, account, recipient and body: the
//! compiled source's resource digest must commit to the [`ApprovedMessage`], and
//! the executed call input must equal it byte-for-byte after JSON parsing. The
//! executor calls its [`MessageSender`] at most once. Any sender failure,
//! including a refusal, consumes the executor; an uncertain send is never
//! retried, and the governed lifecycle records it as unresolved.
//!
//! This adapter performs delivery only. Provider read-back (postcondition
//! evidence that a message exists) and live-account acceptance are separate.
use crate::{GovernedLiveExecutor, ToolExecutionContext, ToolExecutionError, ToolOutput};
use abi_agent_runtime::ToolCall;
use abi_capability::{
    ApprovalLevel, Clock, Digest, Request, Reversibility, SideEffect,
    lifecycle::PreparedExecution,
    source::{AdapterBinding, CompiledSource, PackageSource, ResidencyPolicy},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::sync::Arc;

const TOOL: &str = "message_send";
const SOURCE_ID: &str = "abi.message.send";
const SOURCE_VERSION: &str = "1.0.0";
const MAX_FIELD_BYTES: usize = 2048;
const MAX_PROVIDER_ID_BYTES: usize = 128;

/// Delivery channel a message is approved for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageChannel {
    /// A Discord channel message (recipient is a channel snowflake).
    DiscordChannel,
    /// A Twilio SMS (recipient is an E.164 number).
    TwilioSms,
}

/// The exact message a human approved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovedMessage {
    /// Delivery channel.
    pub channel: MessageChannel,
    /// Sending account identity (Discord bot label, Twilio sending number).
    pub account: String,
    /// Recipient (Discord channel id, E.164 number).
    pub recipient: String,
    /// Message body.
    pub body: String,
}

impl ApprovedMessage {
    /// The canonical tool input for this message.
    #[must_use]
    pub fn input_value(&self) -> serde_json::Value {
        json!({
            "channel": self.channel,
            "account": self.account,
            "recipient": self.recipient,
            "body": self.body,
        })
    }
    /// Versioned resource commitment the compiled source must carry.
    #[must_use]
    pub fn resource_digest(&self) -> Digest {
        commitment(&("abi.message.approved.v1", self.input_value()))
    }
    fn well_formed(&self) -> bool {
        [&self.account, &self.recipient, &self.body]
            .iter()
            .all(|f| !f.is_empty() && f.len() <= MAX_FIELD_BYTES)
    }
}

/// Provider acknowledgement of a created message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageReceipt {
    /// Provider-assigned message id (Discord message id, Twilio SID).
    pub provider_id: String,
}

/// Why a send did not produce a receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SendError {
    /// Refused before any request left the host.
    Refused(&'static str),
    /// The request may have reached the provider; the outcome is unknown.
    Uncertain(&'static str),
}

/// Delivers one approved message.
pub trait MessageSender {
    /// Send `message` exactly as given. Never retries internally.
    fn send(&mut self, message: &ApprovedMessage) -> Result<MessageReceipt, SendError>;
}

/// Sealed message executor bound to one approved message and compiled source.
pub struct GovernedMessageExecutor<S: MessageSender> {
    approved: ApprovedMessage,
    sender: S,
    source: PackageSource,
    package: Digest,
    schema: Digest,
    clock: Arc<dyn Clock>,
    attempted: bool,
    sends: u32,
}

impl<S: MessageSender> GovernedMessageExecutor<S> {
    /// Bind `approved` to one exact compiled source and trusted clock.
    ///
    /// # Errors
    ///
    /// `message_source_mismatch` when the source does not declare this
    /// capability exactly or does not commit to `approved`.
    pub fn bind(
        source: &CompiledSource,
        approved: ApprovedMessage,
        sender: S,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, ToolExecutionError> {
        let src = source.source();
        let payload_budget = approved.input_value().to_string().len();
        if src.id != SOURCE_ID
            || src.version != SOURCE_VERSION
            || !approved.well_formed()
            || src.resource_scope.resource_digest != approved.resource_digest()
            || src.adapter_bindings != [adapter()]
            || src.input_schema != input_schema()
            || src.output_schema != output_schema()
            || src.side_effect != SideEffect::PlatformWrite
            || src.reversibility != Reversibility::Irreversible
            || src.residency != ResidencyPolicy::TenantRegion
            || src.approval_policy < ApprovalLevel::A1Actor
            || src.confirmation_policy < ApprovalLevel::A1Actor
            || src.budgets.max_calls != 1
            || (src.budgets.max_input_bytes as usize) < payload_budget
            || (src.budgets.max_output_bytes as usize) < MAX_PROVIDER_ID_BYTES + 32
            || src.postconditions.is_empty()
            || !src.receipt_requirements.require_postcondition_evidence
        {
            return Err(error("message_source_mismatch"));
        }
        Ok(Self {
            approved,
            sender,
            source: src.clone(),
            package: source.package().digest,
            schema: source.schema_digest(),
            clock,
            attempted: false,
            sends: 0,
        })
    }

    /// How many times the sender has been invoked (0 or 1).
    #[must_use]
    pub const fn sends(&self) -> u32 {
        self.sends
    }

    fn bound(
        &self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
    ) -> bool {
        let input = self.approved.input_value();
        let id = commitment(&(
            "abi.host.execution.v2",
            &request.attempt_id,
            &request.principal,
            &request.scope,
            &request.capability_id,
            &request.capability_version,
            &call.id,
            &call.name,
        ));
        let binding = prepared.binding();
        request.computed_digest() == request.request_digest
            && request.package_digest == self.package
            && request.scope == self.source.resource_scope
            && request.capability_id == self.source.id
            && request.capability_version == self.source.version
            && request.parameter_digest == commitment(&input)
            && source.source() == &self.source
            && source.package().digest == self.package
            && source.schema_digest() == self.schema
            && call.name == TOOL
            && !call.id.is_empty()
            && call.id.len() <= 128
            && call
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
            && call.input.len() <= self.source.budgets.max_input_bytes as usize
            && serde_json::from_str::<serde_json::Value>(&call.input).ok() == Some(input)
            && prepared.execution_id() == id
            && request.call_id == id.to_string().trim_start_matches("sha256:")
            && binding.request_digest == request.request_digest
            && binding.package_digest == self.package
            && binding.schema_digest == self.schema
            && binding.principal_digest == commitment(&request.principal)
            && binding.scope_digest == commitment(&request.scope)
            && binding.compensation_origin.is_none()
            && binding.required_approval >= ApprovalLevel::A1Actor
            && binding.required_approval
                >= self
                    .source
                    .approval_policy
                    .max(self.source.confirmation_policy)
            && binding.policy_digest
                == commitment(&(
                    "abi.policy.v1",
                    &request.policy_version,
                    &request.guild_constitution_version,
                    &request.safety_policy_version,
                    request.revocation_epoch,
                ))
    }
}

impl<S: MessageSender> GovernedLiveExecutor for GovernedMessageExecutor<S> {
    fn binding(&self) -> AdapterBinding {
        adapter()
    }
    fn execute(
        &mut self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError> {
        if self.attempted {
            return Err(error("message_already_attempted"));
        }
        if !self.bound(call, request, source, prepared) || context.should_stop() {
            return Err(error("message_binding_refused"));
        }
        if self.clock.now_ms() >= self.source.expiration_criteria.expires_at_ms
            || context.should_stop()
        {
            return Err(error("message_stopped"));
        }
        // Consume before the effect: whatever the sender returns, never again.
        self.attempted = true;
        self.sends += 1;
        let receipt = self
            .sender
            .send(&self.approved)
            .map_err(|failure| match failure {
                SendError::Refused(code) | SendError::Uncertain(code) => error(code),
            })?;
        if receipt.provider_id.is_empty()
            || receipt.provider_id.len() > MAX_PROVIDER_ID_BYTES
            || receipt.provider_id.chars().any(char::is_control)
        {
            return Err(error("message_receipt_unverified"));
        }
        let payload = json!({ "provider_id": receipt.provider_id }).to_string();
        if payload.len() > self.source.budgets.max_output_bytes as usize {
            return Err(error("message_receipt_unverified"));
        }
        Ok(ToolOutput::ok(payload))
    }
}

fn adapter() -> AdapterBinding {
    AdapterBinding {
        adapter_id: "abi.message".into(),
        api_version: "1".into(),
    }
}
fn input_schema() -> serde_json::Value {
    json!({
        "type":"object",
        "properties":{
            "channel":{"enum":["discord_channel","twilio_sms"]},
            "account":{"type":"string"},
            "recipient":{"type":"string"},
            "body":{"type":"string"}
        },
        "required":["channel","account","recipient","body"],
        "additionalProperties":false
    })
}
fn output_schema() -> serde_json::Value {
    json!({
        "type":"object",
        "properties":{"provider_id":{"type":"string"}},
        "required":["provider_id"],
        "additionalProperties":false
    })
}
fn error(code: &'static str) -> ToolExecutionError {
    ToolExecutionError::new(code)
}
fn commitment<T: Serialize>(value: &T) -> Digest {
    Digest::from_bytes(
        Sha256::digest(serde_json::to_vec(value).expect("closed message commitment")).into(),
    )
}
