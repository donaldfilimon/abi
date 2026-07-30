//! Pure Discord gateway command routing.
//!
//! Ported from `src/connectors/discord_routing.zig`. No transport dependency —
//! command parse, reply text, and UTF-8-safe truncation only.

/// Replies are truncated to this many bytes (Discord's message limit is 2000).
pub const MAX_MESSAGE_CONTENT_BYTES: usize = 1900;

/// Parsed Discord command from a message body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscordCommand {
    /// `!help`
    Help,
    /// `!status`
    Status,
    /// `!prompt`
    Prompt,
    /// `!governance`
    Governance,
    /// Prefixed but unrecognized command.
    Unknown,
    /// Message is not for the bot (missing/empty prefix).
    NotForAbbey,
}

/// Deterministic local prompt summary (reference routing reply).
#[must_use]
pub const fn prompt_summary() -> &'static str {
    "prompt summary: local persona routing (Abbey/Aviva/Abi) via keyword sentiment; completions recorded to WDBX."
}

/// Deterministic local governance summary (reference routing reply).
#[must_use]
pub const fn governance_summary() -> &'static str {
    "governance: six-principle constitutional audit (truthfulness, safety, helpfulness, fairness, privacy, transparency) with a weighted E-score and a hard safety-class veto."
}

const HELP_TEXT: &str = "Available commands: !help, !status, !prompt, !governance";

fn status_text(token_configured: bool) -> &'static str {
    if token_configured {
        "status: connected (token configured)"
    } else {
        "status: offline (no token configured)"
    }
}

/// Parse a command from a message body given a command prefix.
///
/// An empty prefix captures nothing (so the bot never replies to every message);
/// a message without the prefix is [`DiscordCommand::NotForAbbey`].
#[must_use]
pub fn parse_discord_command(content: &str, prefix: &str) -> DiscordCommand {
    if prefix.is_empty() {
        return DiscordCommand::NotForAbbey;
    }
    let Some(rest) = content.strip_prefix(prefix) else {
        return DiscordCommand::NotForAbbey;
    };
    let cmd = rest.split(' ').next().unwrap_or("");
    if cmd.eq_ignore_ascii_case("help") {
        DiscordCommand::Help
    } else if cmd.eq_ignore_ascii_case("status") {
        DiscordCommand::Status
    } else if cmd.eq_ignore_ascii_case("prompt") {
        DiscordCommand::Prompt
    } else if cmd.eq_ignore_ascii_case("governance") {
        DiscordCommand::Governance
    } else {
        DiscordCommand::Unknown
    }
}

/// Route a message to a reply string.
///
/// Returns `None` when no reply should be sent (message not for Abbey).
#[must_use]
pub fn route_discord_message(
    content: &str,
    prefix: &str,
    token_configured: bool,
) -> Option<String> {
    match parse_discord_command(content, prefix) {
        DiscordCommand::NotForAbbey => None,
        DiscordCommand::Unknown => {
            Some("Unknown command. Type `!help` for available commands.".into())
        }
        DiscordCommand::Help => Some(HELP_TEXT.into()),
        DiscordCommand::Status => Some(status_text(token_configured).into()),
        DiscordCommand::Prompt => Some(prompt_summary().into()),
        DiscordCommand::Governance => Some(governance_summary().into()),
    }
}

/// Truncate `text` to at most `max` bytes without splitting a UTF-8 sequence.
#[must_use]
pub fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut end = max;
    while end > 0 && (bytes[end] & 0xC0) == 0x80 {
        end -= 1;
    }
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_honors_prefix_and_empty_prefix_safety() {
        assert_eq!(parse_discord_command("!help", "!"), DiscordCommand::Help);
        assert_eq!(
            parse_discord_command("!status", "!"),
            DiscordCommand::Status
        );
        assert_eq!(
            parse_discord_command("!prompt", "!"),
            DiscordCommand::Prompt
        );
        assert_eq!(
            parse_discord_command("!governance", "!"),
            DiscordCommand::Governance
        );
        assert_eq!(
            parse_discord_command("!nonsense", "!"),
            DiscordCommand::Unknown
        );
        assert_eq!(
            parse_discord_command("help", ""),
            DiscordCommand::NotForAbbey
        );
        assert_eq!(
            parse_discord_command("help me", "!"),
            DiscordCommand::NotForAbbey
        );
    }

    #[test]
    fn route_produces_replies_and_null_for_non_abbey() {
        assert!(route_discord_message("hello there", "!", true).is_none());
        let status = route_discord_message("!status", "!", true).unwrap();
        assert!(status.contains("connected"));
        let status_off = route_discord_message("!status", "!", false).unwrap();
        assert!(status_off.contains("offline"));
        let gov = route_discord_message("!governance", "!", true).unwrap();
        assert!(gov.contains("constitutional"));
    }

    #[test]
    fn truncate_respects_byte_and_utf8_boundaries() {
        assert_eq!(truncate("hello", 1900), "hello");
        // é is 2 bytes in UTF-8; 5 bytes → two complete characters (4 bytes).
        let cut = truncate("ééééé", 5);
        assert_eq!(cut.len(), 4);
        assert_eq!(cut, "éé");
    }
}
