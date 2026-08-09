//! Provider-neutral request description.

use crate::tool::ToolSpec;

/// Author of one message in a conversation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Role {
    /// Operator instructions.
    System,
    /// The person or host driving the conversation.
    #[default]
    User,
    /// The model.
    Assistant,
    /// A tool result fed back to the model.
    Tool,
}

impl Role {
    /// Stable lowercase label, suitable for logs and audit records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One conversation turn.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Message {
    /// Who authored the turn.
    pub role: Role,
    /// Turn text, carried verbatim.
    pub content: String,
}

impl Message {
    /// Build a message for `role`.
    #[must_use]
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Build a [`Role::User`] message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Build a [`Role::Assistant`] message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

/// Everything a provider needs to start one run.
///
/// The `model` field is an opaque identifier. This crate keeps no model
/// catalogue, applies no default, and makes no claim about which models exist
/// or what any of them can do — that belongs to whichever adapter implements
/// [`ModelProvider`].
///
/// [`ModelProvider`]: crate::ModelProvider
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelRequest {
    model: String,
    system: Option<String>,
    messages: Vec<Message>,
    tools: Vec<ToolSpec>,
    max_output_tokens: Option<u64>,
}

impl ModelRequest {
    /// Start a request for the opaque model identifier `model`.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Self::default()
        }
    }

    /// Set the system instruction.
    #[must_use]
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Append one message, preserving order.
    #[must_use]
    pub fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Append a [`Role::User`] message.
    #[must_use]
    pub fn with_user(self, content: impl Into<String>) -> Self {
        self.with_message(Message::user(content))
    }

    /// Offer one tool to the model.
    #[must_use]
    pub fn with_tool(mut self, spec: ToolSpec) -> Self {
        self.tools.push(spec);
        self
    }

    /// Request a ceiling on generated tokens.
    ///
    /// This is a hint forwarded to the provider. It is not enforced here —
    /// [`RunBudget`] is what this crate enforces.
    ///
    /// [`RunBudget`]: crate::RunBudget
    #[must_use]
    pub fn with_max_output_tokens(mut self, tokens: u64) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Opaque model identifier.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// System instruction, if one was set.
    #[must_use]
    pub fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    /// Conversation messages, in order.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Tools offered to the model, in order.
    #[must_use]
    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    /// Requested output-token hint, if one was set.
    #[must_use]
    pub fn max_output_tokens(&self) -> Option<u64> {
        self.max_output_tokens
    }

    /// Last message authored by [`Role::User`], if any.
    #[must_use]
    pub fn last_user_message(&self) -> Option<&Message> {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
    }
}

#[cfg(test)]
mod tests {
    use super::{Message, ModelRequest, Role};
    use crate::tool::ToolSpec;

    #[test]
    fn builder_sets_every_field_and_keeps_order() {
        let request = ModelRequest::new("test-model")
            .with_system("be terse")
            .with_user("first")
            .with_message(Message::assistant("reply"))
            .with_user("second")
            .with_tool(ToolSpec::new("read_file"))
            .with_max_output_tokens(128);

        assert_eq!(request.model(), "test-model");
        assert_eq!(request.system(), Some("be terse"));
        assert_eq!(request.messages().len(), 3);
        assert_eq!(request.messages()[1].role, Role::Assistant);
        assert_eq!(request.tools().len(), 1);
        assert_eq!(request.max_output_tokens(), Some(128));
        assert_eq!(
            request
                .last_user_message()
                .map(|message| message.content.as_str()),
            Some("second")
        );
    }

    #[test]
    fn a_bare_request_has_no_defaults_invented_for_it() {
        let request = ModelRequest::new("m");
        assert_eq!(request.system(), None);
        assert_eq!(request.messages(), []);
        assert_eq!(request.tools(), []);
        assert_eq!(request.max_output_tokens(), None);
        assert_eq!(request.last_user_message(), None);
    }

    #[test]
    fn role_labels_are_stable() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(Role::Tool.as_str(), "tool");
        assert_eq!(Role::default(), Role::User);
    }
}
