//! Discord/Twilio delivery for the governed message executor.
//!
//! [`ConnectorMessageSender`] implements
//! [`abi_agent_host::governed_message::MessageSender`] over the existing live
//! `send_message_live` paths. Each channel is configured at startup with one
//! account identity; an approved message for any other account or channel is
//! refused before a request is built. Discord's account is the configured bot
//! label; Twilio's account is the configured sending number (`From`).
//!
//! Failure split: anything that fails while building the request (local
//! transport, insecure URL, missing credentials, invalid recipient or body)
//! never left the host and is [`SendError::Refused`]. Anything after the
//! request was handed to the transport — connection, timeout, non-2xx status,
//! or a 2xx response without a provider id — is [`SendError::Uncertain`],
//! because the provider may have created the message.

use crate::connector::{ConnectorError, Response};
use crate::discord_gateway::{validate_discord_id, validate_message_content};
use crate::providers::{DiscordClient, TwilioClient};
use crate::transport::Transport;
use abi_agent_host::governed_message::{
    ApprovedMessage, MessageChannel, MessageReceipt, MessageSender, SendError,
};

/// Twilio's documented maximum message body, in bytes of this ceiling.
const MAX_SMS_BODY_BYTES: usize = 1600;
const MAX_PROVIDER_ID_BYTES: usize = 128;

/// Live Discord/Twilio sender bound to startup-configured accounts.
pub struct ConnectorMessageSender<'t> {
    transport: &'t dyn Transport,
    discord: Option<(String, DiscordClient)>,
    twilio: Option<(String, TwilioClient)>,
}

impl<'t> ConnectorMessageSender<'t> {
    /// A sender with no channel configured; every send is refused until one is.
    #[must_use]
    pub fn new(transport: &'t dyn Transport) -> Self {
        Self {
            transport,
            discord: None,
            twilio: None,
        }
    }

    /// Enable Discord channel messages for the bot account labelled `account`.
    #[must_use]
    pub fn with_discord(mut self, account: impl Into<String>, client: DiscordClient) -> Self {
        self.discord = Some((account.into(), client));
        self
    }

    /// Enable Twilio SMS sent from the number `from` (the bound account).
    #[must_use]
    pub fn with_twilio(mut self, from: impl Into<String>, client: TwilioClient) -> Self {
        self.twilio = Some((from.into(), client));
        self
    }
}

impl MessageSender for ConnectorMessageSender<'_> {
    fn send(&mut self, message: &ApprovedMessage) -> Result<MessageReceipt, SendError> {
        match message.channel {
            MessageChannel::DiscordChannel => {
                let (account, client) = self
                    .discord
                    .as_ref()
                    .ok_or(SendError::Refused("message_channel_unconfigured"))?;
                if *account != message.account {
                    return Err(SendError::Refused("message_account_mismatch"));
                }
                validate_discord_id(&message.recipient)
                    .and_then(|()| validate_message_content(&message.body))
                    .map_err(|_| SendError::Refused("message_invalid"))?;
                client
                    .build_send_request(&message.recipient, &message.body)
                    .map_err(|e| refused(&e))?;
                let response = client
                    .send_message_live(self.transport, &message.recipient, &message.body)
                    .map_err(|e| uncertain(&e))?;
                receipt(&response, "id")
            }
            MessageChannel::TwilioSms => {
                let (from, client) = self
                    .twilio
                    .as_ref()
                    .ok_or(SendError::Refused("message_channel_unconfigured"))?;
                if *from != message.account {
                    return Err(SendError::Refused("message_account_mismatch"));
                }
                if !e164(&message.recipient)
                    || message.body.is_empty()
                    || message.body.len() > MAX_SMS_BODY_BYTES
                {
                    return Err(SendError::Refused("message_invalid"));
                }
                client
                    .build_message_request(&message.recipient, from, &message.body)
                    .map_err(|e| refused(&e))?;
                let response = client
                    .send_message_live(self.transport, &message.recipient, from, &message.body)
                    .map_err(|e| uncertain(&e))?;
                receipt(&response, "sid")
            }
        }
    }
}

fn refused(_: &ConnectorError) -> SendError {
    SendError::Refused("message_request_refused")
}
fn uncertain(error: &ConnectorError) -> SendError {
    match error {
        // Only the unavailable transport reports this, and it never sends.
        ConnectorError::LiveTransportUnavailable => {
            SendError::Refused("message_transport_unavailable")
        }
        _ => SendError::Uncertain("message_send_uncertain"),
    }
}
fn receipt(response: &Response, key: &str) -> Result<MessageReceipt, SendError> {
    let id = response
        .json()
        .ok()
        .and_then(|value| value.get(key)?.as_str().map(str::to_owned))
        .filter(|id| {
            !id.is_empty()
                && id.len() <= MAX_PROVIDER_ID_BYTES
                && id.bytes().all(|b| b.is_ascii_alphanumeric())
        })
        .ok_or(SendError::Uncertain("message_receipt_unverified"))?;
    Ok(MessageReceipt { provider_id: id })
}
fn e164(number: &str) -> bool {
    number.strip_prefix('+').is_some_and(|digits| {
        (2..=15).contains(&digits.len()) && digits.bytes().all(|b| b.is_ascii_digit())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connector::{ConnectorConfig, ConnectorError, Response};
    use crate::providers::{DiscordClient, TwilioClient};
    use crate::transport::{Method, RecordingTransport};
    use abi_agent_host::governed_message::{
        ApprovedMessage, MessageChannel, MessageSender, SendError,
    };

    fn discord() -> DiscordClient {
        DiscordClient::new(ConnectorConfig::new("tok", "https://discord.com").live())
    }
    fn twilio() -> TwilioClient {
        TwilioClient::new(
            "AC123",
            "secret",
            ConnectorConfig::new("unused", "https://api.twilio.com").live(),
        )
    }
    fn discord_message() -> ApprovedMessage {
        ApprovedMessage {
            channel: MessageChannel::DiscordChannel,
            account: "abbey-bot".into(),
            recipient: "123456789012345678".into(),
            body: "deploy finished".into(),
        }
    }
    fn sms_message() -> ApprovedMessage {
        ApprovedMessage {
            channel: MessageChannel::TwilioSms,
            account: "+15559876543".into(),
            recipient: "+15551234567".into(),
            body: "hi there".into(),
        }
    }

    #[test]
    fn connector_sender_builds_exact_discord_and_twilio_requests() {
        let transport =
            RecordingTransport::new().with_response(Ok(Response::ok(r#"{"id":"1100"}"#)));
        let mut sender =
            ConnectorMessageSender::new(&transport).with_discord("abbey-bot", discord());
        let receipt = sender.send(&discord_message()).unwrap();
        assert_eq!(receipt.provider_id, "1100");
        let request = transport.sole_request();
        assert_eq!(request.method, Method::Post);
        assert_eq!(
            request.url,
            "https://discord.com/api/v10/channels/123456789012345678/messages"
        );
        assert_eq!(
            request.body.as_deref(),
            Some(r#"{"content":"deploy finished"}"#)
        );

        let transport =
            RecordingTransport::new().with_response(Ok(Response::ok(r#"{"sid":"SM42"}"#)));
        let mut sender =
            ConnectorMessageSender::new(&transport).with_twilio("+15559876543", twilio());
        let receipt = sender.send(&sms_message()).unwrap();
        assert_eq!(receipt.provider_id, "SM42");
        let request = transport.sole_request();
        assert_eq!(request.method, Method::Post);
        assert_eq!(
            request.url,
            "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages.json"
        );
        assert_eq!(
            request.body.as_deref(),
            Some("To=%2B15551234567&From=%2B15559876543&Body=hi%20there")
        );
    }

    #[test]
    fn unapproved_account_channel_or_invalid_recipient_refuses_without_transport() {
        let transport = RecordingTransport::new();
        let mut sender = ConnectorMessageSender::new(&transport)
            .with_discord("abbey-bot", discord())
            .with_twilio("+15559876543", twilio());
        let mut wrong_account = discord_message();
        wrong_account.account = "other-bot".into();
        let mut wrong_sms_account = sms_message();
        wrong_sms_account.account = "+15550000000".into();
        let mut bad_recipient = discord_message();
        bad_recipient.recipient = "../../users/@me".into();
        let mut bad_number = sms_message();
        bad_number.recipient = "5551234567".into();
        let mut empty_body = discord_message();
        empty_body.body = String::new();
        for message in [
            wrong_account,
            wrong_sms_account,
            bad_recipient,
            bad_number,
            empty_body,
        ] {
            assert!(matches!(sender.send(&message), Err(SendError::Refused(_))));
        }
        let mut unconfigured = ConnectorMessageSender::new(&transport);
        assert!(matches!(
            unconfigured.send(&discord_message()),
            Err(SendError::Refused(_))
        ));
        assert_eq!(transport.requests(), Vec::new());
    }

    #[test]
    fn transport_error_maps_to_uncertain() {
        for outcome in [
            Err(ConnectorError::ConnectionFailed),
            Err(ConnectorError::Timeout),
            Ok(Response {
                status: 500,
                body: String::new(),
            }),
            // Accepted but no provider id to prove which message was created.
            Ok(Response::ok("{}")),
        ] {
            let transport = RecordingTransport::new().with_response(outcome);
            let mut sender =
                ConnectorMessageSender::new(&transport).with_discord("abbey-bot", discord());
            assert!(matches!(
                sender.send(&discord_message()),
                Err(SendError::Uncertain(_))
            ));
            assert_eq!(transport.requests().len(), 1);
        }
    }

    #[test]
    fn local_transport_config_is_refused_before_any_request() {
        let transport = RecordingTransport::new();
        let local = DiscordClient::new(ConnectorConfig::new("tok", "https://discord.com"));
        let mut sender = ConnectorMessageSender::new(&transport).with_discord("abbey-bot", local);
        assert!(matches!(
            sender.send(&discord_message()),
            Err(SendError::Refused(_))
        ));
        assert_eq!(transport.requests(), Vec::new());
    }
}
