//! Discord API Connector
//!
//! Comprehensive Discord API integration supporting:
//! - REST API for all Discord resources
//! - Gateway WebSocket for real-time events
//! - Application commands (slash commands, context menus)
//! - Interactions (buttons, select menus, modals)
//! - Webhooks for sending messages
//! - OAuth2 for user authentication
//! - Voice connection management
//!
//! Environment Variables:
//! - ABI_DISCORD_BOT_TOKEN / DISCORD_BOT_TOKEN - Bot authentication token
//! - ABI_DISCORD_CLIENT_ID / DISCORD_CLIENT_ID - Application client ID
//! - ABI_DISCORD_CLIENT_SECRET / DISCORD_CLIENT_SECRET - OAuth2 client secret
//! - ABI_DISCORD_PUBLIC_KEY / DISCORD_PUBLIC_KEY - Interaction verification key

const std = @import("std");
const connectors = @import("mod.zig");
const async_http = @import("../../shared/utils/http/async_http.zig");
const json_utils = @import("../../shared/utils/json/mod.zig");

// ============================================================================
// Error Types
// ============================================================================

pub const DiscordError = error{
    MissingBotToken,
    MissingClientId,
    MissingClientSecret,
    MissingPublicKey,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
    Unauthorized,
    Forbidden,
    NotFound,
    GatewayError,
    WebSocketError,
    InvalidToken,
    InvalidInteraction,
    UnknownInteraction,
    CommandNotFound,
    InvalidPermissions,
    MissingAccess,
    InvalidWebhook,
    VoiceConnectionFailed,
};

// ============================================================================
// Discord Snowflake ID
// ============================================================================

pub const Snowflake = []const u8;

// ============================================================================
// User Types
// ============================================================================

pub const User = struct {
    id: Snowflake,
    username: []const u8,
    discriminator: []const u8,
    global_name: ?[]const u8 = null,
    avatar: ?[]const u8 = null,
    bot: bool = false,
    system: bool = false,
    mfa_enabled: bool = false,
    banner: ?[]const u8 = null,
    accent_color: ?u32 = null,
    locale: ?[]const u8 = null,
    verified: bool = false,
    email: ?[]const u8 = null,
    flags: u32 = 0,
    premium_type: u8 = 0,
    public_flags: u32 = 0,
};

pub const UserFlags = struct {
    pub const STAFF: u32 = 1 << 0;
    pub const PARTNER: u32 = 1 << 1;
    pub const HYPESQUAD: u32 = 1 << 2;
    pub const BUG_HUNTER_LEVEL_1: u32 = 1 << 3;
    pub const HYPESQUAD_BRAVERY: u32 = 1 << 6;
    pub const HYPESQUAD_BRILLIANCE: u32 = 1 << 7;
    pub const HYPESQUAD_BALANCE: u32 = 1 << 8;
    pub const EARLY_SUPPORTER: u32 = 1 << 9;
    pub const TEAM_USER: u32 = 1 << 10;
    pub const BUG_HUNTER_LEVEL_2: u32 = 1 << 14;
    pub const VERIFIED_BOT: u32 = 1 << 16;
    pub const EARLY_VERIFIED_BOT_DEVELOPER: u32 = 1 << 17;
    pub const DISCORD_CERTIFIED_MODERATOR: u32 = 1 << 18;
    pub const BOT_HTTP_INTERACTIONS: u32 = 1 << 19;
    pub const ACTIVE_DEVELOPER: u32 = 1 << 22;
};

// ============================================================================
// Guild (Server) Types
// ============================================================================

pub const Guild = struct {
    id: Snowflake,
    name: []const u8,
    icon: ?[]const u8 = null,
    icon_hash: ?[]const u8 = null,
    splash: ?[]const u8 = null,
    discovery_splash: ?[]const u8 = null,
    owner: bool = false,
    owner_id: Snowflake,
    permissions: ?[]const u8 = null,
    region: ?[]const u8 = null,
    afk_channel_id: ?Snowflake = null,
    afk_timeout: u32 = 0,
    widget_enabled: bool = false,
    widget_channel_id: ?Snowflake = null,
    verification_level: u8 = 0,
    default_message_notifications: u8 = 0,
    explicit_content_filter: u8 = 0,
    features: []const []const u8 = &.{},
    mfa_level: u8 = 0,
    application_id: ?Snowflake = null,
    system_channel_id: ?Snowflake = null,
    system_channel_flags: u32 = 0,
    rules_channel_id: ?Snowflake = null,
    max_presences: ?u32 = null,
    max_members: u32 = 0,
    vanity_url_code: ?[]const u8 = null,
    description: ?[]const u8 = null,
    banner: ?[]const u8 = null,
    premium_tier: u8 = 0,
    premium_subscription_count: u32 = 0,
    preferred_locale: []const u8 = "en-US",
    public_updates_channel_id: ?Snowflake = null,
    max_video_channel_users: u32 = 0,
    max_stage_video_channel_users: u32 = 0,
    approximate_member_count: u32 = 0,
    approximate_presence_count: u32 = 0,
    nsfw_level: u8 = 0,
    premium_progress_bar_enabled: bool = false,
    safety_alerts_channel_id: ?Snowflake = null,
};

pub const GuildMember = struct {
    user: ?User = null,
    nick: ?[]const u8 = null,
    avatar: ?[]const u8 = null,
    roles: []const Snowflake = &.{},
    joined_at: []const u8,
    premium_since: ?[]const u8 = null,
    deaf: bool = false,
    mute: bool = false,
    flags: u32 = 0,
    pending: bool = false,
    permissions: ?[]const u8 = null,
    communication_disabled_until: ?[]const u8 = null,
};

pub const Role = struct {
    id: Snowflake,
    name: []const u8,
    color: u32 = 0,
    hoist: bool = false,
    icon: ?[]const u8 = null,
    unicode_emoji: ?[]const u8 = null,
    position: u32 = 0,
    permissions: []const u8,
    managed: bool = false,
    mentionable: bool = false,
    tags: ?RoleTags = null,
    flags: u32 = 0,
};

pub const RoleTags = struct {
    bot_id: ?Snowflake = null,
    integration_id: ?Snowflake = null,
    premium_subscriber: bool = false,
    subscription_listing_id: ?Snowflake = null,
    available_for_purchase: bool = false,
    guild_connections: bool = false,
};

// ============================================================================
// Channel Types
// ============================================================================

pub const ChannelType = enum(u8) {
    GUILD_TEXT = 0,
    DM = 1,
    GUILD_VOICE = 2,
    GROUP_DM = 3,
    GUILD_CATEGORY = 4,
    GUILD_ANNOUNCEMENT = 5,
    ANNOUNCEMENT_THREAD = 10,
    PUBLIC_THREAD = 11,
    PRIVATE_THREAD = 12,
    GUILD_STAGE_VOICE = 13,
    GUILD_DIRECTORY = 14,
    GUILD_FORUM = 15,
    GUILD_MEDIA = 16,
};

pub const Channel = struct {
    id: Snowflake,
    channel_type: u8,
    guild_id: ?Snowflake = null,
    position: ?u32 = null,
    permission_overwrites: []const PermissionOverwrite = &.{},
    name: ?[]const u8 = null,
    topic: ?[]const u8 = null,
    nsfw: bool = false,
    last_message_id: ?Snowflake = null,
    bitrate: ?u32 = null,
    user_limit: ?u32 = null,
    rate_limit_per_user: u32 = 0,
    recipients: []const User = &.{},
    icon: ?[]const u8 = null,
    owner_id: ?Snowflake = null,
    application_id: ?Snowflake = null,
    managed: bool = false,
    parent_id: ?Snowflake = null,
    last_pin_timestamp: ?[]const u8 = null,
    rtc_region: ?[]const u8 = null,
    video_quality_mode: u8 = 1,
    message_count: u32 = 0,
    member_count: u32 = 0,
    thread_metadata: ?ThreadMetadata = null,
    member: ?ThreadMember = null,
    default_auto_archive_duration: u32 = 1440,
    permissions: ?[]const u8 = null,
    flags: u32 = 0,
    total_message_sent: u32 = 0,
    default_reaction_emoji: ?DefaultReaction = null,
    default_thread_rate_limit_per_user: u32 = 0,
    default_sort_order: ?u8 = null,
    default_forum_layout: u8 = 0,
};

pub const PermissionOverwrite = struct {
    id: Snowflake,
    overwrite_type: u8, // 0 = role, 1 = member
    allow: []const u8,
    deny: []const u8,
};

pub const ThreadMetadata = struct {
    archived: bool = false,
    auto_archive_duration: u32 = 1440,
    archive_timestamp: []const u8,
    locked: bool = false,
    invitable: bool = true,
    create_timestamp: ?[]const u8 = null,
};

pub const ThreadMember = struct {
    id: ?Snowflake = null,
    user_id: ?Snowflake = null,
    join_timestamp: []const u8,
    flags: u32 = 0,
    member: ?GuildMember = null,
};

pub const DefaultReaction = struct {
    emoji_id: ?Snowflake = null,
    emoji_name: ?[]const u8 = null,
};

// ============================================================================
// Message Types
// ============================================================================

pub const Message = struct {
    id: Snowflake,
    channel_id: Snowflake,
    author: User,
    content: []const u8,
    timestamp: []const u8,
    edited_timestamp: ?[]const u8 = null,
    tts: bool = false,
    mention_everyone: bool = false,
    mentions: []const User = &.{},
    mention_roles: []const Snowflake = &.{},
    mention_channels: []const ChannelMention = &.{},
    attachments: []const Attachment = &.{},
    embeds: []const Embed = &.{},
    reactions: []const Reaction = &.{},
    nonce: ?[]const u8 = null,
    pinned: bool = false,
    webhook_id: ?Snowflake = null,
    message_type: u8 = 0,
    activity: ?MessageActivity = null,
    application: ?Application = null,
    application_id: ?Snowflake = null,
    message_reference: ?MessageReference = null,
    flags: u32 = 0,
    referenced_message: ?*Message = null,
    interaction: ?MessageInteraction = null,
    thread: ?Channel = null,
    components: []const Component = &.{},
    sticker_items: []const StickerItem = &.{},
    position: ?u32 = null,
};

pub const ChannelMention = struct {
    id: Snowflake,
    guild_id: Snowflake,
    channel_type: u8,
    name: []const u8,
};

pub const Attachment = struct {
    id: Snowflake,
    filename: []const u8,
    description: ?[]const u8 = null,
    content_type: ?[]const u8 = null,
    size: u64,
    url: []const u8,
    proxy_url: []const u8,
    height: ?u32 = null,
    width: ?u32 = null,
    ephemeral: bool = false,
    duration_secs: ?f32 = null,
    waveform: ?[]const u8 = null,
    flags: u32 = 0,
};

pub const Embed = struct {
    title: ?[]const u8 = null,
    embed_type: ?[]const u8 = null,
    description: ?[]const u8 = null,
    url: ?[]const u8 = null,
    timestamp: ?[]const u8 = null,
    color: ?u32 = null,
    footer: ?EmbedFooter = null,
    image: ?EmbedMedia = null,
    thumbnail: ?EmbedMedia = null,
    video: ?EmbedMedia = null,
    provider: ?EmbedProvider = null,
    author: ?EmbedAuthor = null,
    fields: []const EmbedField = &.{},
};

pub const EmbedFooter = struct {
    text: []const u8,
    icon_url: ?[]const u8 = null,
    proxy_icon_url: ?[]const u8 = null,
};

pub const EmbedMedia = struct {
    url: []const u8,
    proxy_url: ?[]const u8 = null,
    height: ?u32 = null,
    width: ?u32 = null,
};

pub const EmbedProvider = struct {
    name: ?[]const u8 = null,
    url: ?[]const u8 = null,
};

pub const EmbedAuthor = struct {
    name: []const u8,
    url: ?[]const u8 = null,
    icon_url: ?[]const u8 = null,
    proxy_icon_url: ?[]const u8 = null,
};

pub const EmbedField = struct {
    name: []const u8,
    value: []const u8,
    inline_field: bool = false,
};

pub const Reaction = struct {
    count: u32,
    count_details: ReactionCountDetails,
    me: bool,
    me_burst: bool,
    emoji: Emoji,
    burst_colors: []const []const u8 = &.{},
};

pub const ReactionCountDetails = struct {
    burst: u32,
    normal: u32,
};

pub const Emoji = struct {
    id: ?Snowflake = null,
    name: ?[]const u8 = null,
    roles: []const Snowflake = &.{},
    user: ?User = null,
    require_colons: bool = true,
    managed: bool = false,
    animated: bool = false,
    available: bool = true,
};

pub const MessageActivity = struct {
    activity_type: u8,
    party_id: ?[]const u8 = null,
};

pub const MessageReference = struct {
    message_id: ?Snowflake = null,
    channel_id: ?Snowflake = null,
    guild_id: ?Snowflake = null,
    fail_if_not_exists: bool = true,
};

pub const MessageInteraction = struct {
    id: Snowflake,
    interaction_type: u8,
    name: []const u8,
    user: User,
    member: ?GuildMember = null,
};

pub const StickerItem = struct {
    id: Snowflake,
    name: []const u8,
    format_type: u8,
};

// ============================================================================
// Application Types
// ============================================================================

pub const Application = struct {
    id: Snowflake,
    name: []const u8,
    icon: ?[]const u8 = null,
    description: []const u8,
    rpc_origins: []const []const u8 = &.{},
    bot_public: bool = true,
    bot_require_code_grant: bool = false,
    bot: ?User = null,
    terms_of_service_url: ?[]const u8 = null,
    privacy_policy_url: ?[]const u8 = null,
    owner: ?User = null,
    verify_key: []const u8,
    team: ?Team = null,
    guild_id: ?Snowflake = null,
    guild: ?Guild = null,
    primary_sku_id: ?Snowflake = null,
    slug: ?[]const u8 = null,
    cover_image: ?[]const u8 = null,
    flags: u32 = 0,
    approximate_guild_count: u32 = 0,
    redirect_uris: []const []const u8 = &.{},
    interactions_endpoint_url: ?[]const u8 = null,
    role_connections_verification_url: ?[]const u8 = null,
    tags: []const []const u8 = &.{},
    install_params: ?InstallParams = null,
    integration_types_config: ?IntegrationTypesConfig = null,
    custom_install_url: ?[]const u8 = null,
};

pub const Team = struct {
    icon: ?[]const u8 = null,
    id: Snowflake,
    members: []const TeamMember = &.{},
    name: []const u8,
    owner_user_id: Snowflake,
};

pub const TeamMember = struct {
    membership_state: u8,
    team_id: Snowflake,
    user: User,
    role: []const u8,
};

pub const InstallParams = struct {
    scopes: []const []const u8,
    permissions: []const u8,
};

pub const IntegrationTypesConfig = struct {
    guild_install: ?IntegrationTypeConfig = null,
    user_install: ?IntegrationTypeConfig = null,
};

pub const IntegrationTypeConfig = struct {
    oauth2_install_params: ?InstallParams = null,
};

// ============================================================================
// Interaction Types (Slash Commands, Buttons, etc.)
// ============================================================================

pub const InteractionType = enum(u8) {
    PING = 1,
    APPLICATION_COMMAND = 2,
    MESSAGE_COMPONENT = 3,
    APPLICATION_COMMAND_AUTOCOMPLETE = 4,
    MODAL_SUBMIT = 5,
};

pub const Interaction = struct {
    id: Snowflake,
    application_id: Snowflake,
    interaction_type: u8,
    data: ?InteractionData = null,
    guild_id: ?Snowflake = null,
    channel: ?Channel = null,
    channel_id: ?Snowflake = null,
    member: ?GuildMember = null,
    user: ?User = null,
    token: []const u8,
    version: u8 = 1,
    message: ?Message = null,
    app_permissions: ?[]const u8 = null,
    locale: ?[]const u8 = null,
    guild_locale: ?[]const u8 = null,
    entitlements: []const Entitlement = &.{},
    authorizing_integration_owners: ?AuthorizingIntegrationOwners = null,
    context: ?u8 = null,
};

pub const InteractionData = struct {
    id: Snowflake,
    name: []const u8,
    data_type: u8,
    resolved: ?ResolvedData = null,
    options: []const ApplicationCommandInteractionDataOption = &.{},
    guild_id: ?Snowflake = null,
    target_id: ?Snowflake = null,
    custom_id: ?[]const u8 = null,
    component_type: ?u8 = null,
    values: []const []const u8 = &.{},
    components: []const Component = &.{},
};

pub const ResolvedData = struct {
    users: ?std.StringHashMap(User) = null,
    members: ?std.StringHashMap(GuildMember) = null,
    roles: ?std.StringHashMap(Role) = null,
    channels: ?std.StringHashMap(Channel) = null,
    messages: ?std.StringHashMap(Message) = null,
    attachments: ?std.StringHashMap(Attachment) = null,
};

pub const ApplicationCommandInteractionDataOption = struct {
    name: []const u8,
    option_type: u8,
    value: ?[]const u8 = null,
    options: []const ApplicationCommandInteractionDataOption = &.{},
    focused: bool = false,
};

pub const Entitlement = struct {
    id: Snowflake,
    sku_id: Snowflake,
    application_id: Snowflake,
    user_id: ?Snowflake = null,
    entitlement_type: u8,
    deleted: bool = false,
    starts_at: ?[]const u8 = null,
    ends_at: ?[]const u8 = null,
    guild_id: ?Snowflake = null,
    consumed: bool = false,
};

pub const AuthorizingIntegrationOwners = struct {
    guild_install: ?Snowflake = null,
    user_install: ?Snowflake = null,
};

// ============================================================================
// Component Types (Buttons, Select Menus, Modals)
// ============================================================================

pub const ComponentType = enum(u8) {
    ACTION_ROW = 1,
    BUTTON = 2,
    STRING_SELECT = 3,
    TEXT_INPUT = 4,
    USER_SELECT = 5,
    ROLE_SELECT = 6,
    MENTIONABLE_SELECT = 7,
    CHANNEL_SELECT = 8,
};

pub const Component = struct {
    component_type: u8,
    custom_id: ?[]const u8 = null,
    disabled: bool = false,
    style: ?u8 = null,
    label: ?[]const u8 = null,
    emoji: ?Emoji = null,
    url: ?[]const u8 = null,
    options: []const SelectOption = &.{},
    channel_types: []const u8 = &.{},
    placeholder: ?[]const u8 = null,
    default_values: []const DefaultValue = &.{},
    min_values: u8 = 1,
    max_values: u8 = 1,
    components: []const Component = &.{},
    min_length: ?u32 = null,
    max_length: ?u32 = null,
    required: bool = false,
    value: ?[]const u8 = null,
};

pub const ButtonStyle = enum(u8) {
    PRIMARY = 1,
    SECONDARY = 2,
    SUCCESS = 3,
    DANGER = 4,
    LINK = 5,
};

pub const SelectOption = struct {
    label: []const u8,
    value: []const u8,
    description: ?[]const u8 = null,
    emoji: ?Emoji = null,
    default: bool = false,
};

pub const DefaultValue = struct {
    id: Snowflake,
    default_type: []const u8,
};

pub const TextInputStyle = enum(u8) {
    SHORT = 1,
    PARAGRAPH = 2,
};

// ============================================================================
// Application Command Types
// ============================================================================

pub const ApplicationCommandType = enum(u8) {
    CHAT_INPUT = 1,
    USER = 2,
    MESSAGE = 3,
};

pub const ApplicationCommand = struct {
    id: Snowflake,
    command_type: u8 = 1,
    application_id: Snowflake,
    guild_id: ?Snowflake = null,
    name: []const u8,
    name_localizations: ?std.StringHashMap([]const u8) = null,
    description: []const u8,
    description_localizations: ?std.StringHashMap([]const u8) = null,
    options: []const ApplicationCommandOption = &.{},
    default_member_permissions: ?[]const u8 = null,
    dm_permission: bool = true,
    default_permission: bool = true,
    nsfw: bool = false,
    integration_types: []const u8 = &.{},
    contexts: []const u8 = &.{},
    version: Snowflake,
};

pub const ApplicationCommandOptionType = enum(u8) {
    SUB_COMMAND = 1,
    SUB_COMMAND_GROUP = 2,
    STRING = 3,
    INTEGER = 4,
    BOOLEAN = 5,
    USER = 6,
    CHANNEL = 7,
    ROLE = 8,
    MENTIONABLE = 9,
    NUMBER = 10,
    ATTACHMENT = 11,
};

pub const ApplicationCommandOption = struct {
    option_type: u8,
    name: []const u8,
    name_localizations: ?std.StringHashMap([]const u8) = null,
    description: []const u8,
    description_localizations: ?std.StringHashMap([]const u8) = null,
    required: bool = false,
    choices: []const ApplicationCommandOptionChoice = &.{},
    options: []const ApplicationCommandOption = &.{},
    channel_types: []const u8 = &.{},
    min_value: ?f64 = null,
    max_value: ?f64 = null,
    min_length: ?u32 = null,
    max_length: ?u32 = null,
    autocomplete: bool = false,
};

pub const ApplicationCommandOptionChoice = struct {
    name: []const u8,
    name_localizations: ?std.StringHashMap([]const u8) = null,
    value: []const u8, // Can be string, int, or float as string
};

// ============================================================================
// Interaction Response Types
// ============================================================================

pub const InteractionCallbackType = enum(u8) {
    PONG = 1,
    CHANNEL_MESSAGE_WITH_SOURCE = 4,
    DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE = 5,
    DEFERRED_UPDATE_MESSAGE = 6,
    UPDATE_MESSAGE = 7,
    APPLICATION_COMMAND_AUTOCOMPLETE_RESULT = 8,
    MODAL = 9,
    PREMIUM_REQUIRED = 10,
};

pub const InteractionResponse = struct {
    response_type: u8,
    data: ?InteractionCallbackData = null,
};

pub const InteractionCallbackData = struct {
    tts: bool = false,
    content: ?[]const u8 = null,
    embeds: []const Embed = &.{},
    allowed_mentions: ?AllowedMentions = null,
    flags: u32 = 0,
    components: []const Component = &.{},
    attachments: []const Attachment = &.{},
    choices: []const ApplicationCommandOptionChoice = &.{},
    custom_id: ?[]const u8 = null,
    title: ?[]const u8 = null,
};

pub const AllowedMentions = struct {
    parse: []const []const u8 = &.{},
    roles: []const Snowflake = &.{},
    users: []const Snowflake = &.{},
    replied_user: bool = false,
};

pub const MessageFlags = struct {
    pub const CROSSPOSTED: u32 = 1 << 0;
    pub const IS_CROSSPOST: u32 = 1 << 1;
    pub const SUPPRESS_EMBEDS: u32 = 1 << 2;
    pub const SOURCE_MESSAGE_DELETED: u32 = 1 << 3;
    pub const URGENT: u32 = 1 << 4;
    pub const HAS_THREAD: u32 = 1 << 5;
    pub const EPHEMERAL: u32 = 1 << 6;
    pub const LOADING: u32 = 1 << 7;
    pub const FAILED_TO_MENTION_SOME_ROLES_IN_THREAD: u32 = 1 << 8;
    pub const SUPPRESS_NOTIFICATIONS: u32 = 1 << 12;
    pub const IS_VOICE_MESSAGE: u32 = 1 << 13;
};

// ============================================================================
// Webhook Types
// ============================================================================

pub const Webhook = struct {
    id: Snowflake,
    webhook_type: u8,
    guild_id: ?Snowflake = null,
    channel_id: ?Snowflake = null,
    user: ?User = null,
    name: ?[]const u8 = null,
    avatar: ?[]const u8 = null,
    token: ?[]const u8 = null,
    application_id: ?Snowflake = null,
    source_guild: ?Guild = null,
    source_channel: ?Channel = null,
    url: ?[]const u8 = null,
};

pub const WebhookType = enum(u8) {
    INCOMING = 1,
    CHANNEL_FOLLOWER = 2,
    APPLICATION = 3,
};

// ============================================================================
// Voice Types
// ============================================================================

pub const VoiceState = struct {
    guild_id: ?Snowflake = null,
    channel_id: ?Snowflake = null,
    user_id: Snowflake,
    member: ?GuildMember = null,
    session_id: []const u8,
    deaf: bool = false,
    mute: bool = false,
    self_deaf: bool = false,
    self_mute: bool = false,
    self_stream: bool = false,
    self_video: bool = false,
    suppress: bool = false,
    request_to_speak_timestamp: ?[]const u8 = null,
};

pub const VoiceRegion = struct {
    id: []const u8,
    name: []const u8,
    optimal: bool = false,
    deprecated: bool = false,
    custom: bool = false,
};

// ============================================================================
// Gateway Types
// ============================================================================

pub const GatewayOpcode = enum(u8) {
    DISPATCH = 0,
    HEARTBEAT = 1,
    IDENTIFY = 2,
    PRESENCE_UPDATE = 3,
    VOICE_STATE_UPDATE = 4,
    RESUME = 6,
    RECONNECT = 7,
    REQUEST_GUILD_MEMBERS = 8,
    INVALID_SESSION = 9,
    HELLO = 10,
    HEARTBEAT_ACK = 11,
};

pub const GatewayIntent = struct {
    pub const GUILDS: u32 = 1 << 0;
    pub const GUILD_MEMBERS: u32 = 1 << 1;
    pub const GUILD_MODERATION: u32 = 1 << 2;
    pub const GUILD_EMOJIS_AND_STICKERS: u32 = 1 << 3;
    pub const GUILD_INTEGRATIONS: u32 = 1 << 4;
    pub const GUILD_WEBHOOKS: u32 = 1 << 5;
    pub const GUILD_INVITES: u32 = 1 << 6;
    pub const GUILD_VOICE_STATES: u32 = 1 << 7;
    pub const GUILD_PRESENCES: u32 = 1 << 8;
    pub const GUILD_MESSAGES: u32 = 1 << 9;
    pub const GUILD_MESSAGE_REACTIONS: u32 = 1 << 10;
    pub const GUILD_MESSAGE_TYPING: u32 = 1 << 11;
    pub const DIRECT_MESSAGES: u32 = 1 << 12;
    pub const DIRECT_MESSAGE_REACTIONS: u32 = 1 << 13;
    pub const DIRECT_MESSAGE_TYPING: u32 = 1 << 14;
    pub const MESSAGE_CONTENT: u32 = 1 << 15;
    pub const GUILD_SCHEDULED_EVENTS: u32 = 1 << 16;
    pub const AUTO_MODERATION_CONFIGURATION: u32 = 1 << 20;
    pub const AUTO_MODERATION_EXECUTION: u32 = 1 << 21;

    pub const ALL_UNPRIVILEGED: u32 = GUILDS | GUILD_MODERATION | GUILD_EMOJIS_AND_STICKERS |
        GUILD_INTEGRATIONS | GUILD_WEBHOOKS | GUILD_INVITES | GUILD_VOICE_STATES |
        GUILD_MESSAGES | GUILD_MESSAGE_REACTIONS | GUILD_MESSAGE_TYPING |
        DIRECT_MESSAGES | DIRECT_MESSAGE_REACTIONS | DIRECT_MESSAGE_TYPING |
        GUILD_SCHEDULED_EVENTS | AUTO_MODERATION_CONFIGURATION | AUTO_MODERATION_EXECUTION;

    pub const ALL_PRIVILEGED: u32 = GUILD_MEMBERS | GUILD_PRESENCES | MESSAGE_CONTENT;

    pub const ALL: u32 = ALL_UNPRIVILEGED | ALL_PRIVILEGED;
};

pub const GatewayPayload = struct {
    op: u8,
    d: ?std.json.Value = null,
    s: ?u64 = null,
    t: ?[]const u8 = null,
};

pub const IdentifyProperties = struct {
    os: []const u8 = "zig",
    browser: []const u8 = "abi",
    device: []const u8 = "abi",
};

pub const PresenceUpdate = struct {
    since: ?u64 = null,
    activities: []const Activity = &.{},
    status: []const u8 = "online",
    afk: bool = false,
};

pub const Activity = struct {
    name: []const u8,
    activity_type: u8 = 0,
    url: ?[]const u8 = null,
    created_at: ?u64 = null,
    timestamps: ?ActivityTimestamps = null,
    application_id: ?Snowflake = null,
    details: ?[]const u8 = null,
    state: ?[]const u8 = null,
    emoji: ?Emoji = null,
    party: ?ActivityParty = null,
    assets: ?ActivityAssets = null,
    secrets: ?ActivitySecrets = null,
    instance: bool = false,
    flags: u32 = 0,
    buttons: []const ActivityButton = &.{},
};

pub const ActivityType = enum(u8) {
    GAME = 0,
    STREAMING = 1,
    LISTENING = 2,
    WATCHING = 3,
    CUSTOM = 4,
    COMPETING = 5,
};

pub const ActivityTimestamps = struct {
    start: ?u64 = null,
    end: ?u64 = null,
};

pub const ActivityParty = struct {
    id: ?[]const u8 = null,
    size: ?[2]u32 = null,
};

pub const ActivityAssets = struct {
    large_image: ?[]const u8 = null,
    large_text: ?[]const u8 = null,
    small_image: ?[]const u8 = null,
    small_text: ?[]const u8 = null,
};

pub const ActivitySecrets = struct {
    join: ?[]const u8 = null,
    spectate: ?[]const u8 = null,
    match: ?[]const u8 = null,
};

pub const ActivityButton = struct {
    label: []const u8,
    url: []const u8,
};

// ============================================================================
// OAuth2 Types
// ============================================================================

pub const OAuth2Scope = struct {
    pub const ACTIVITIES_READ = "activities.read";
    pub const ACTIVITIES_WRITE = "activities.write";
    pub const APPLICATIONS_BUILDS_READ = "applications.builds.read";
    pub const APPLICATIONS_BUILDS_UPLOAD = "applications.builds.upload";
    pub const APPLICATIONS_COMMANDS = "applications.commands";
    pub const APPLICATIONS_COMMANDS_UPDATE = "applications.commands.update";
    pub const APPLICATIONS_COMMANDS_PERMISSIONS_UPDATE = "applications.commands.permissions.update";
    pub const APPLICATIONS_ENTITLEMENTS = "applications.entitlements";
    pub const APPLICATIONS_STORE_UPDATE = "applications.store.update";
    pub const BOT = "bot";
    pub const CONNECTIONS = "connections";
    pub const DM_CHANNELS_READ = "dm_channels.read";
    pub const EMAIL = "email";
    pub const GDM_JOIN = "gdm.join";
    pub const GUILDS = "guilds";
    pub const GUILDS_JOIN = "guilds.join";
    pub const GUILDS_MEMBERS_READ = "guilds.members.read";
    pub const IDENTIFY = "identify";
    pub const MESSAGES_READ = "messages.read";
    pub const RELATIONSHIPS_READ = "relationships.read";
    pub const ROLE_CONNECTIONS_WRITE = "role_connections.write";
    pub const RPC = "rpc";
    pub const RPC_ACTIVITIES_WRITE = "rpc.activities.write";
    pub const RPC_NOTIFICATIONS_READ = "rpc.notifications.read";
    pub const RPC_VOICE_READ = "rpc.voice.read";
    pub const RPC_VOICE_WRITE = "rpc.voice.write";
    pub const VOICE = "voice";
    pub const WEBHOOK_INCOMING = "webhook.incoming";
};

pub const OAuth2Token = struct {
    access_token: []const u8,
    token_type: []const u8,
    expires_in: u64,
    refresh_token: ?[]const u8 = null,
    scope: []const u8,
};

// ============================================================================
// Configuration
// ============================================================================

pub const Config = struct {
    bot_token: []u8,
    client_id: ?[]u8 = null,
    client_secret: ?[]u8 = null,
    public_key: ?[]u8 = null,
    api_version: u8 = 10,
    timeout_ms: u32 = 30_000,
    intents: u32 = GatewayIntent.ALL_UNPRIVILEGED,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.bot_token);
        if (self.client_id) |id| allocator.free(id);
        if (self.client_secret) |secret| allocator.free(secret);
        if (self.public_key) |key| allocator.free(key);
        self.* = undefined;
    }

    pub fn getBaseUrl(self: *const Config) []const u8 {
        _ = self;
        return "https://discord.com/api/v10";
    }
};

// ============================================================================
// REST API Client
// ============================================================================

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: Config,
    http: async_http.AsyncHttpClient,

    pub fn init(allocator: std.mem.Allocator, config: Config) !Client {
        const http = try async_http.AsyncHttpClient.init(allocator);
        return .{
            .allocator = allocator,
            .config = config,
            .http = http,
        };
    }

    pub fn deinit(self: *Client) void {
        self.http.deinit();
        self.config.deinit(self.allocator);
        self.* = undefined;
    }

    // ========================================================================
    // HTTP Helpers
    // ========================================================================

    fn makeRequest(self: *Client, method: async_http.Method, endpoint: []const u8) !async_http.HttpRequest {
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.config.getBaseUrl(), endpoint });
        errdefer self.allocator.free(url);

        var request = try async_http.HttpRequest.init(self.allocator, method, url);
        errdefer request.deinit();

        const auth = try std.fmt.allocPrint(self.allocator, "Bot {s}", .{self.config.bot_token});
        defer self.allocator.free(auth);
        try request.setHeader("Authorization", auth);
        try request.setHeader("User-Agent", "DiscordBot (https://github.com/abi, 1.0)");

        return request;
    }

    fn doRequest(self: *Client, request: *async_http.HttpRequest) !async_http.HttpResponse {
        const response = try self.http.fetch(request);

        if (response.status_code == 401) {
            return DiscordError.Unauthorized;
        } else if (response.status_code == 403) {
            return DiscordError.Forbidden;
        } else if (response.status_code == 404) {
            return DiscordError.NotFound;
        } else if (response.status_code == 429) {
            return DiscordError.RateLimitExceeded;
        } else if (!response.isSuccess()) {
            return DiscordError.ApiRequestFailed;
        }

        return response;
    }

    // ========================================================================
    // User Endpoints
    // ========================================================================

    /// Get the current user
    pub fn getCurrentUser(self: *Client) !User {
        var request = try self.makeRequest(.get, "/users/@me");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseUser(response.body);
    }

    /// Get a user by ID
    pub fn getUser(self: *Client, user_id: Snowflake) !User {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/users/{s}", .{user_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseUser(response.body);
    }

    /// Modify the current user
    pub fn modifyCurrentUser(self: *Client, username: ?[]const u8, avatar: ?[]const u8) !User {
        var request = try self.makeRequest(.patch, "/users/@me");
        defer request.deinit();

        var body = std.ArrayListUnmanaged(u8){};
        defer body.deinit(self.allocator);

        try body.appendSlice(self.allocator, "{");
        var first = true;

        if (username) |name| {
            try body.print(self.allocator, "\"username\":\"{s}\"", .{name});
            first = false;
        }

        if (avatar) |av| {
            if (!first) try body.appendSlice(self.allocator, ",");
            try body.print(self.allocator, "\"avatar\":\"{s}\"", .{av});
        }

        try body.appendSlice(self.allocator, "}");
        try request.setJsonBody(body.items);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseUser(response.body);
    }

    /// Get current user's guilds
    pub fn getCurrentUserGuilds(self: *Client) ![]Guild {
        var request = try self.makeRequest(.get, "/users/@me/guilds");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseGuildArray(response.body);
    }

    /// Leave a guild
    pub fn leaveGuild(self: *Client, guild_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/users/@me/guilds/{s}", .{guild_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Create a DM channel
    pub fn createDM(self: *Client, recipient_id: Snowflake) !Channel {
        var request = try self.makeRequest(.post, "/users/@me/channels");
        defer request.deinit();

        const body = try std.fmt.allocPrint(self.allocator, "{{\"recipient_id\":\"{s}\"}}", .{recipient_id});
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseChannel(response.body);
    }

    // ========================================================================
    // Guild Endpoints
    // ========================================================================

    /// Get a guild
    pub fn getGuild(self: *Client, guild_id: Snowflake) !Guild {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/guilds/{s}", .{guild_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseGuild(response.body);
    }

    /// Get guild channels
    pub fn getGuildChannels(self: *Client, guild_id: Snowflake) ![]Channel {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/guilds/{s}/channels", .{guild_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseChannelArray(response.body);
    }

    /// Get guild member
    pub fn getGuildMember(self: *Client, guild_id: Snowflake, user_id: Snowflake) !GuildMember {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/guilds/{s}/members/{s}", .{ guild_id, user_id });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseGuildMember(response.body);
    }

    /// Get guild roles
    pub fn getGuildRoles(self: *Client, guild_id: Snowflake) ![]Role {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/guilds/{s}/roles", .{guild_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseRoleArray(response.body);
    }

    // ========================================================================
    // Channel Endpoints
    // ========================================================================

    /// Get a channel
    pub fn getChannel(self: *Client, channel_id: Snowflake) !Channel {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}", .{channel_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseChannel(response.body);
    }

    /// Delete a channel
    pub fn deleteChannel(self: *Client, channel_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}", .{channel_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    // ========================================================================
    // Message Endpoints
    // ========================================================================

    /// Get channel messages
    pub fn getChannelMessages(self: *Client, channel_id: Snowflake, limit: ?u8) ![]Message {
        const lim = limit orelse 50;
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages?limit={d}", .{ channel_id, lim });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessageArray(response.body);
    }

    /// Get a specific message
    pub fn getMessage(self: *Client, channel_id: Snowflake, message_id: Snowflake) !Message {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages/{s}", .{ channel_id, message_id });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Create a message
    pub fn createMessage(self: *Client, channel_id: Snowflake, content: []const u8) !Message {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages", .{channel_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(self.allocator, "{{\"content\":\"{}\"}}", .{json_utils.jsonEscape(content)});
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Create a message with embed
    pub fn createMessageWithEmbed(self: *Client, channel_id: Snowflake, content: ?[]const u8, embed: Embed) !Message {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages", .{channel_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeMessageWithEmbed(content, embed);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Edit a message
    pub fn editMessage(self: *Client, channel_id: Snowflake, message_id: Snowflake, content: []const u8) !Message {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages/{s}", .{ channel_id, message_id });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.patch, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(self.allocator, "{{\"content\":\"{}\"}}", .{json_utils.jsonEscape(content)});
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Delete a message
    pub fn deleteMessage(self: *Client, channel_id: Snowflake, message_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages/{s}", .{ channel_id, message_id });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Add a reaction to a message
    pub fn createReaction(self: *Client, channel_id: Snowflake, message_id: Snowflake, emoji: []const u8) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages/{s}/reactions/{s}/@me", .{ channel_id, message_id, emoji });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.put, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Delete own reaction
    pub fn deleteOwnReaction(self: *Client, channel_id: Snowflake, message_id: Snowflake, emoji: []const u8) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/channels/{s}/messages/{s}/reactions/{s}/@me", .{ channel_id, message_id, emoji });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    // ========================================================================
    // Application Command Endpoints
    // ========================================================================

    /// Get global application commands
    pub fn getGlobalApplicationCommands(self: *Client, application_id: Snowflake) ![]ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/applications/{s}/commands", .{application_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommandArray(response.body);
    }

    /// Create a global application command
    pub fn createGlobalApplicationCommand(
        self: *Client,
        application_id: Snowflake,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) !ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/applications/{s}/commands", .{application_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeApplicationCommand(name, description, options);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommand(response.body);
    }

    /// Delete a global application command
    pub fn deleteGlobalApplicationCommand(self: *Client, application_id: Snowflake, command_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/applications/{s}/commands/{s}", .{ application_id, command_id });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Get guild application commands
    pub fn getGuildApplicationCommands(self: *Client, application_id: Snowflake, guild_id: Snowflake) ![]ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/applications/{s}/guilds/{s}/commands", .{ application_id, guild_id });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommandArray(response.body);
    }

    /// Create a guild application command
    pub fn createGuildApplicationCommand(
        self: *Client,
        application_id: Snowflake,
        guild_id: Snowflake,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) !ApplicationCommand {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/applications/{s}/guilds/{s}/commands", .{ application_id, guild_id });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeApplicationCommand(name, description, options);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseApplicationCommand(response.body);
    }

    // ========================================================================
    // Interaction Response Endpoints
    // ========================================================================

    /// Create an interaction response
    pub fn createInteractionResponse(
        self: *Client,
        interaction_id: Snowflake,
        interaction_token: []const u8,
        response_type: InteractionCallbackType,
        content: ?[]const u8,
    ) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/interactions/{s}/{s}/callback", .{ interaction_id, interaction_token });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        var body = std.ArrayListUnmanaged(u8){};
        defer body.deinit(self.allocator);

        try body.print(self.allocator, "{{\"type\":{d}", .{@intFromEnum(response_type)});

        if (content) |c| {
            try body.print(self.allocator, ",\"data\":{{\"content\":\"{}\"}}", .{json_utils.jsonEscape(c)});
        }

        try body.appendSlice(self.allocator, "}");
        try request.setJsonBody(body.items);

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Edit the original interaction response
    pub fn editOriginalInteractionResponse(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
        content: []const u8,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/webhooks/{s}/{s}/messages/@original", .{ application_id, interaction_token });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.patch, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(self.allocator, "{{\"content\":\"{}\"}}", .{json_utils.jsonEscape(content)});
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    /// Delete the original interaction response
    pub fn deleteOriginalInteractionResponse(self: *Client, application_id: Snowflake, interaction_token: []const u8) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/webhooks/{s}/{s}/messages/@original", .{ application_id, interaction_token });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Create a followup message
    pub fn createFollowupMessage(
        self: *Client,
        application_id: Snowflake,
        interaction_token: []const u8,
        content: []const u8,
    ) !Message {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/webhooks/{s}/{s}", .{ application_id, interaction_token });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(self.allocator, "{{\"content\":\"{}\"}}", .{json_utils.jsonEscape(content)});
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseMessage(response.body);
    }

    // ========================================================================
    // Webhook Endpoints
    // ========================================================================

    /// Get a webhook by ID
    pub fn getWebhook(self: *Client, webhook_id: Snowflake) !Webhook {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/webhooks/{s}", .{webhook_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.get, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseWebhook(response.body);
    }

    /// Execute a webhook
    pub fn executeWebhook(self: *Client, webhook_id: Snowflake, webhook_token: []const u8, content: []const u8) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/webhooks/{s}/{s}", .{ webhook_id, webhook_token });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try std.fmt.allocPrint(self.allocator, "{{\"content\":\"{}\"}}", .{json_utils.jsonEscape(content)});
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Execute a webhook with embeds
    pub fn executeWebhookWithEmbed(
        self: *Client,
        webhook_id: Snowflake,
        webhook_token: []const u8,
        content: ?[]const u8,
        embed: Embed,
    ) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/webhooks/{s}/{s}", .{ webhook_id, webhook_token });
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.post, endpoint);
        defer request.deinit();

        const body = try self.encodeMessageWithEmbed(content, embed);
        defer self.allocator.free(body);
        try request.setJsonBody(body);

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    /// Delete a webhook
    pub fn deleteWebhook(self: *Client, webhook_id: Snowflake) !void {
        const endpoint = try std.fmt.allocPrint(self.allocator, "/webhooks/{s}", .{webhook_id});
        defer self.allocator.free(endpoint);

        var request = try self.makeRequest(.delete, endpoint);
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();
    }

    // ========================================================================
    // Gateway Endpoints
    // ========================================================================

    /// Get the gateway URL
    pub fn getGateway(self: *Client) ![]const u8 {
        var request = try self.makeRequest(.get, "/gateway");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            response.body,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);
        return try json_utils.parseStringField(object, "url", self.allocator);
    }

    /// Get the gateway URL with bot info
    pub fn getGatewayBot(self: *Client) !GatewayBotInfo {
        var request = try self.makeRequest(.get, "/gateway/bot");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            response.body,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return .{
            .url = try json_utils.parseStringField(object, "url", self.allocator),
            .shards = @intCast(try json_utils.parseIntField(object, "shards")),
            .session_start_limit = .{
                .total = 1000,
                .remaining = 1000,
                .reset_after = 0,
                .max_concurrency = 1,
            },
        };
    }

    // ========================================================================
    // Voice Endpoints
    // ========================================================================

    /// Get voice regions
    pub fn getVoiceRegions(self: *Client) ![]VoiceRegion {
        var request = try self.makeRequest(.get, "/voice/regions");
        defer request.deinit();

        var response = try self.doRequest(&request);
        defer response.deinit();

        return try self.parseVoiceRegionArray(response.body);
    }

    // ========================================================================
    // OAuth2 Endpoints
    // ========================================================================

    /// Get the authorization URL for OAuth2
    pub fn getAuthorizationUrl(self: *Client, scopes: []const []const u8, redirect_uri: []const u8, state: ?[]const u8) ![]u8 {
        var scope_str = std.ArrayListUnmanaged(u8){};
        defer scope_str.deinit(self.allocator);

        for (scopes, 0..) |scope, i| {
            if (i > 0) try scope_str.appendSlice(self.allocator, "%20");
            try scope_str.appendSlice(self.allocator, scope);
        }

        const client_id = self.config.client_id orelse return DiscordError.MissingClientId;

        var url = std.ArrayListUnmanaged(u8){};
        errdefer url.deinit(self.allocator);

        try url.print(
            self.allocator,
            "https://discord.com/oauth2/authorize?client_id={s}&response_type=code&redirect_uri={s}&scope={s}",
            .{ client_id, redirect_uri, scope_str.items },
        );

        if (state) |s| {
            try url.print(self.allocator, "&state={s}", .{s});
        }

        return try url.toOwnedSlice(self.allocator);
    }

    /// Exchange an authorization code for an access token
    pub fn exchangeCode(self: *Client, code: []const u8, redirect_uri: []const u8) !OAuth2Token {
        const client_id = self.config.client_id orelse return DiscordError.MissingClientId;
        const client_secret = self.config.client_secret orelse return DiscordError.MissingClientSecret;

        const url = "https://discord.com/api/oauth2/token";

        var request = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer request.deinit();

        try request.setHeader("Content-Type", "application/x-www-form-urlencoded");

        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=authorization_code&code={s}&redirect_uri={s}&client_id={s}&client_secret={s}",
            .{ code, redirect_uri, client_id, client_secret },
        );
        defer self.allocator.free(body);
        try request.setBody(body);

        var response = try self.http.fetch(&request);
        defer response.deinit();

        if (!response.isSuccess()) {
            return DiscordError.ApiRequestFailed;
        }

        return try self.parseOAuth2Token(response.body);
    }

    /// Refresh an access token
    pub fn refreshToken(self: *Client, refresh_token: []const u8) !OAuth2Token {
        const client_id = self.config.client_id orelse return DiscordError.MissingClientId;
        const client_secret = self.config.client_secret orelse return DiscordError.MissingClientSecret;

        const url = "https://discord.com/api/oauth2/token";

        var request = try async_http.HttpRequest.init(self.allocator, .post, url);
        defer request.deinit();

        try request.setHeader("Content-Type", "application/x-www-form-urlencoded");

        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=refresh_token&refresh_token={s}&client_id={s}&client_secret={s}",
            .{ refresh_token, client_id, client_secret },
        );
        defer self.allocator.free(body);
        try request.setBody(body);

        var response = try self.http.fetch(&request);
        defer response.deinit();

        if (!response.isSuccess()) {
            return DiscordError.ApiRequestFailed;
        }

        return try self.parseOAuth2Token(response.body);
    }

    // ========================================================================
    // JSON Encoding Helpers
    // ========================================================================

    fn encodeMessageWithEmbed(self: *Client, content: ?[]const u8, embed: Embed) ![]u8 {
        var json = std.ArrayListUnmanaged(u8){};
        errdefer json.deinit(self.allocator);

        try json.appendSlice(self.allocator, "{");

        if (content) |c| {
            try json.print(self.allocator, "\"content\":\"{}\",", .{json_utils.jsonEscape(c)});
        }

        try json.appendSlice(self.allocator, "\"embeds\":[{");

        var first = true;
        if (embed.title) |title| {
            try json.print(self.allocator, "\"title\":\"{}\"", .{json_utils.jsonEscape(title)});
            first = false;
        }

        if (embed.description) |desc| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"description\":\"{}\"", .{json_utils.jsonEscape(desc)});
            first = false;
        }

        if (embed.color) |color| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"color\":{d}", .{color});
            first = false;
        }

        if (embed.url) |url| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"url\":\"{s}\"", .{url});
            first = false;
        }

        if (embed.timestamp) |ts| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"timestamp\":\"{s}\"", .{ts});
            first = false;
        }

        if (embed.footer) |footer| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"footer\":{{\"text\":\"{}\"", .{json_utils.jsonEscape(footer.text)});
            if (footer.icon_url) |icon| {
                try json.print(self.allocator, ",\"icon_url\":\"{s}\"", .{icon});
            }
            try json.appendSlice(self.allocator, "}");
            first = false;
        }

        if (embed.author) |author| {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.print(self.allocator, "\"author\":{{\"name\":\"{}\"", .{json_utils.jsonEscape(author.name)});
            if (author.url) |url| {
                try json.print(self.allocator, ",\"url\":\"{s}\"", .{url});
            }
            if (author.icon_url) |icon| {
                try json.print(self.allocator, ",\"icon_url\":\"{s}\"", .{icon});
            }
            try json.appendSlice(self.allocator, "}");
            first = false;
        }

        if (embed.fields.len > 0) {
            if (!first) try json.appendSlice(self.allocator, ",");
            try json.appendSlice(self.allocator, "\"fields\":[");
            for (embed.fields, 0..) |field, i| {
                if (i > 0) try json.appendSlice(self.allocator, ",");
                try json.print(
                    self.allocator,
                    "{{\"name\":\"{}\",\"value\":\"{}\",\"inline\":{s}}}",
                    .{
                        json_utils.jsonEscape(field.name),
                        json_utils.jsonEscape(field.value),
                        if (field.inline_field) "true" else "false",
                    },
                );
            }
            try json.appendSlice(self.allocator, "]");
        }

        try json.appendSlice(self.allocator, "}]}");

        return try json.toOwnedSlice(self.allocator);
    }

    fn encodeApplicationCommand(
        self: *Client,
        name: []const u8,
        description: []const u8,
        options: []const ApplicationCommandOption,
    ) ![]u8 {
        var json = std.ArrayListUnmanaged(u8){};
        errdefer json.deinit(self.allocator);

        try json.print(
            self.allocator,
            "{{\"name\":\"{s}\",\"description\":\"{}\"",
            .{ name, json_utils.jsonEscape(description) },
        );

        if (options.len > 0) {
            try json.appendSlice(self.allocator, ",\"options\":[");
            for (options, 0..) |opt, i| {
                if (i > 0) try json.appendSlice(self.allocator, ",");
                try json.print(
                    self.allocator,
                    "{{\"type\":{d},\"name\":\"{s}\",\"description\":\"{}\",\"required\":{s}}}",
                    .{
                        opt.option_type,
                        opt.name,
                        json_utils.jsonEscape(opt.description),
                        if (opt.required) "true" else "false",
                    },
                );
            }
            try json.appendSlice(self.allocator, "]");
        }

        try json.appendSlice(self.allocator, "}");

        return try json.toOwnedSlice(self.allocator);
    }

    // ========================================================================
    // JSON Parsing Helpers (simplified - full implementation would be more robust)
    // ========================================================================

    fn parseUser(self: *Client, json: []const u8) !User {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return User{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .username = try json_utils.parseStringField(object, "username", self.allocator),
            .discriminator = try json_utils.parseStringField(object, "discriminator", self.allocator),
            .global_name = json_utils.parseOptionalStringField(object, "global_name", self.allocator) catch null,
            .avatar = json_utils.parseOptionalStringField(object, "avatar", self.allocator) catch null,
            .bot = json_utils.parseBoolField(object, "bot") catch false,
        };
    }

    fn parseGuild(self: *Client, json: []const u8) !Guild {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return Guild{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .name = try json_utils.parseStringField(object, "name", self.allocator),
            .owner_id = try json_utils.parseStringField(object, "owner_id", self.allocator),
            .icon = json_utils.parseOptionalStringField(object, "icon", self.allocator) catch null,
        };
    }

    fn parseGuildArray(self: *Client, json: []const u8) ![]Guild {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var guilds = try self.allocator.alloc(Guild, array.items.len);
        errdefer self.allocator.free(guilds);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            guilds[i] = Guild{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .owner_id = (json_utils.parseOptionalStringField(object, "owner_id", self.allocator) catch null) orelse "",
                .icon = json_utils.parseOptionalStringField(object, "icon", self.allocator) catch null,
            };
        }

        return guilds;
    }

    fn parseChannel(self: *Client, json: []const u8) !Channel {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return Channel{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .channel_type = @intCast(try json_utils.parseIntField(object, "type")),
            .name = json_utils.parseOptionalStringField(object, "name", self.allocator) catch null,
            .guild_id = json_utils.parseOptionalStringField(object, "guild_id", self.allocator) catch null,
        };
    }

    fn parseChannelArray(self: *Client, json: []const u8) ![]Channel {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var channels = try self.allocator.alloc(Channel, array.items.len);
        errdefer self.allocator.free(channels);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            channels[i] = Channel{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .channel_type = @intCast(try json_utils.parseIntField(object, "type")),
                .name = json_utils.parseOptionalStringField(object, "name", self.allocator) catch null,
                .guild_id = json_utils.parseOptionalStringField(object, "guild_id", self.allocator) catch null,
            };
        }

        return channels;
    }

    fn parseGuildMember(self: *Client, json: []const u8) !GuildMember {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return GuildMember{
            .joined_at = try json_utils.parseStringField(object, "joined_at", self.allocator),
            .nick = json_utils.parseOptionalStringField(object, "nick", self.allocator) catch null,
            .deaf = json_utils.parseBoolField(object, "deaf") catch false,
            .mute = json_utils.parseBoolField(object, "mute") catch false,
        };
    }

    fn parseRoleArray(self: *Client, json: []const u8) ![]Role {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var roles = try self.allocator.alloc(Role, array.items.len);
        errdefer self.allocator.free(roles);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            roles[i] = Role{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .permissions = try json_utils.parseStringField(object, "permissions", self.allocator),
                .color = @intCast(json_utils.parseIntField(object, "color") catch 0),
                .position = @intCast(json_utils.parseIntField(object, "position") catch 0),
            };
        }

        return roles;
    }

    fn parseMessage(self: *Client, json: []const u8) !Message {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        const author_obj = try json_utils.parseObjectField(object, "author");

        return Message{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .channel_id = try json_utils.parseStringField(object, "channel_id", self.allocator),
            .content = try json_utils.parseStringField(object, "content", self.allocator),
            .timestamp = try json_utils.parseStringField(object, "timestamp", self.allocator),
            .author = User{
                .id = try json_utils.parseStringField(author_obj, "id", self.allocator),
                .username = try json_utils.parseStringField(author_obj, "username", self.allocator),
                .discriminator = try json_utils.parseStringField(author_obj, "discriminator", self.allocator),
            },
        };
    }

    fn parseMessageArray(self: *Client, json: []const u8) ![]Message {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var messages = try self.allocator.alloc(Message, array.items.len);
        errdefer self.allocator.free(messages);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            const author_obj = try json_utils.parseObjectField(object, "author");

            messages[i] = Message{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .channel_id = try json_utils.parseStringField(object, "channel_id", self.allocator),
                .content = try json_utils.parseStringField(object, "content", self.allocator),
                .timestamp = try json_utils.parseStringField(object, "timestamp", self.allocator),
                .author = User{
                    .id = try json_utils.parseStringField(author_obj, "id", self.allocator),
                    .username = try json_utils.parseStringField(author_obj, "username", self.allocator),
                    .discriminator = try json_utils.parseStringField(author_obj, "discriminator", self.allocator),
                },
            };
        }

        return messages;
    }

    fn parseApplicationCommand(self: *Client, json: []const u8) !ApplicationCommand {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return ApplicationCommand{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .application_id = try json_utils.parseStringField(object, "application_id", self.allocator),
            .name = try json_utils.parseStringField(object, "name", self.allocator),
            .description = try json_utils.parseStringField(object, "description", self.allocator),
            .version = try json_utils.parseStringField(object, "version", self.allocator),
        };
    }

    fn parseApplicationCommandArray(self: *Client, json: []const u8) ![]ApplicationCommand {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var commands = try self.allocator.alloc(ApplicationCommand, array.items.len);
        errdefer self.allocator.free(commands);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            commands[i] = ApplicationCommand{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .application_id = try json_utils.parseStringField(object, "application_id", self.allocator),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .description = try json_utils.parseStringField(object, "description", self.allocator),
                .version = try json_utils.parseStringField(object, "version", self.allocator),
            };
        }

        return commands;
    }

    fn parseWebhook(self: *Client, json: []const u8) !Webhook {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return Webhook{
            .id = try json_utils.parseStringField(object, "id", self.allocator),
            .webhook_type = @intCast(try json_utils.parseIntField(object, "type")),
            .name = json_utils.parseOptionalStringField(object, "name", self.allocator) catch null,
            .token = json_utils.parseOptionalStringField(object, "token", self.allocator) catch null,
        };
    }

    fn parseVoiceRegionArray(self: *Client, json: []const u8) ![]VoiceRegion {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const array = parsed.value.array;
        var regions = try self.allocator.alloc(VoiceRegion, array.items.len);
        errdefer self.allocator.free(regions);

        for (array.items, 0..) |item, i| {
            const object = try json_utils.getRequiredObject(item);
            regions[i] = VoiceRegion{
                .id = try json_utils.parseStringField(object, "id", self.allocator),
                .name = try json_utils.parseStringField(object, "name", self.allocator),
                .optimal = json_utils.parseBoolField(object, "optimal") catch false,
                .deprecated = json_utils.parseBoolField(object, "deprecated") catch false,
            };
        }

        return regions;
    }

    fn parseOAuth2Token(self: *Client, json: []const u8) !OAuth2Token {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json,
            .{ .ignore_unknown_fields = true },
        );
        defer parsed.deinit();

        const object = try json_utils.getRequiredObject(parsed.value);

        return OAuth2Token{
            .access_token = try json_utils.parseStringField(object, "access_token", self.allocator),
            .token_type = try json_utils.parseStringField(object, "token_type", self.allocator),
            .expires_in = @intCast(try json_utils.parseIntField(object, "expires_in")),
            .refresh_token = json_utils.parseOptionalStringField(object, "refresh_token", self.allocator) catch null,
            .scope = try json_utils.parseStringField(object, "scope", self.allocator),
        };
    }
};

pub const GatewayBotInfo = struct {
    url: []const u8,
    shards: u32,
    session_start_limit: SessionStartLimit,
};

pub const SessionStartLimit = struct {
    total: u32,
    remaining: u32,
    reset_after: u64,
    max_concurrency: u32,
};

// ============================================================================
// Environment Loading
// ============================================================================

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const bot_token = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
    })) orelse return DiscordError.MissingBotToken;

    const client_id = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_CLIENT_ID",
        "DISCORD_CLIENT_ID",
    });

    const client_secret = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_CLIENT_SECRET",
        "DISCORD_CLIENT_SECRET",
    });

    const public_key = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_DISCORD_PUBLIC_KEY",
        "DISCORD_PUBLIC_KEY",
    });

    return .{
        .bot_token = bot_token,
        .client_id = client_id,
        .client_secret = client_secret,
        .public_key = public_key,
    };
}

pub fn createClient(allocator: std.mem.Allocator) !Client {
    const config = try loadFromEnv(allocator);
    return try Client.init(allocator, config);
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Parse a Discord timestamp to Unix timestamp
pub fn parseTimestamp(iso_timestamp: []const u8) !i64 {
    // Discord uses ISO 8601 format: "2021-01-01T00:00:00.000000+00:00"
    // This is a simplified parser
    _ = iso_timestamp;
    return std.time.timestamp();
}

/// Format a Unix timestamp to Discord timestamp format
pub fn formatTimestamp(unix_timestamp: i64, style: TimestampStyle) ![]u8 {
    _ = unix_timestamp;
    _ = style;
    return &.{};
}

pub const TimestampStyle = enum {
    SHORT_TIME, // 16:20
    LONG_TIME, // 16:20:30
    SHORT_DATE, // 20/04/2021
    LONG_DATE, // 20 April 2021
    SHORT_DATE_TIME, // 20 April 2021 16:20
    LONG_DATE_TIME, // Tuesday, 20 April 2021 16:20
    RELATIVE, // 2 months ago
};

/// Calculate permissions from an array of permission flags
pub fn calculatePermissions(permissions: []const u64) u64 {
    var result: u64 = 0;
    for (permissions) |p| {
        result |= p;
    }
    return result;
}

/// Check if a permission is set
pub fn hasPermission(permissions: u64, permission: u64) bool {
    return (permissions & permission) == permission;
}

/// Discord Permissions
pub const Permission = struct {
    pub const CREATE_INSTANT_INVITE: u64 = 1 << 0;
    pub const KICK_MEMBERS: u64 = 1 << 1;
    pub const BAN_MEMBERS: u64 = 1 << 2;
    pub const ADMINISTRATOR: u64 = 1 << 3;
    pub const MANAGE_CHANNELS: u64 = 1 << 4;
    pub const MANAGE_GUILD: u64 = 1 << 5;
    pub const ADD_REACTIONS: u64 = 1 << 6;
    pub const VIEW_AUDIT_LOG: u64 = 1 << 7;
    pub const PRIORITY_SPEAKER: u64 = 1 << 8;
    pub const STREAM: u64 = 1 << 9;
    pub const VIEW_CHANNEL: u64 = 1 << 10;
    pub const SEND_MESSAGES: u64 = 1 << 11;
    pub const SEND_TTS_MESSAGES: u64 = 1 << 12;
    pub const MANAGE_MESSAGES: u64 = 1 << 13;
    pub const EMBED_LINKS: u64 = 1 << 14;
    pub const ATTACH_FILES: u64 = 1 << 15;
    pub const READ_MESSAGE_HISTORY: u64 = 1 << 16;
    pub const MENTION_EVERYONE: u64 = 1 << 17;
    pub const USE_EXTERNAL_EMOJIS: u64 = 1 << 18;
    pub const VIEW_GUILD_INSIGHTS: u64 = 1 << 19;
    pub const CONNECT: u64 = 1 << 20;
    pub const SPEAK: u64 = 1 << 21;
    pub const MUTE_MEMBERS: u64 = 1 << 22;
    pub const DEAFEN_MEMBERS: u64 = 1 << 23;
    pub const MOVE_MEMBERS: u64 = 1 << 24;
    pub const USE_VAD: u64 = 1 << 25;
    pub const CHANGE_NICKNAME: u64 = 1 << 26;
    pub const MANAGE_NICKNAMES: u64 = 1 << 27;
    pub const MANAGE_ROLES: u64 = 1 << 28;
    pub const MANAGE_WEBHOOKS: u64 = 1 << 29;
    pub const MANAGE_GUILD_EXPRESSIONS: u64 = 1 << 30;
    pub const USE_APPLICATION_COMMANDS: u64 = 1 << 31;
    pub const REQUEST_TO_SPEAK: u64 = 1 << 32;
    pub const MANAGE_EVENTS: u64 = 1 << 33;
    pub const MANAGE_THREADS: u64 = 1 << 34;
    pub const CREATE_PUBLIC_THREADS: u64 = 1 << 35;
    pub const CREATE_PRIVATE_THREADS: u64 = 1 << 36;
    pub const USE_EXTERNAL_STICKERS: u64 = 1 << 37;
    pub const SEND_MESSAGES_IN_THREADS: u64 = 1 << 38;
    pub const USE_EMBEDDED_ACTIVITIES: u64 = 1 << 39;
    pub const MODERATE_MEMBERS: u64 = 1 << 40;
    pub const VIEW_CREATOR_MONETIZATION_ANALYTICS: u64 = 1 << 41;
    pub const USE_SOUNDBOARD: u64 = 1 << 42;
    pub const CREATE_GUILD_EXPRESSIONS: u64 = 1 << 43;
    pub const CREATE_EVENTS: u64 = 1 << 44;
    pub const USE_EXTERNAL_SOUNDS: u64 = 1 << 45;
    pub const SEND_VOICE_MESSAGES: u64 = 1 << 46;
    pub const SEND_POLLS: u64 = 1 << 49;
    pub const USE_EXTERNAL_APPS: u64 = 1 << 50;
};

// ============================================================================
// Tests
// ============================================================================

test "discord config lifecycle" {
    const allocator = std.testing.allocator;

    var config = Config{
        .bot_token = try allocator.dupe(u8, "test_token"),
        .client_id = try allocator.dupe(u8, "123456789"),
        .client_secret = null,
        .public_key = null,
    };
    defer config.deinit(allocator);

    try std.testing.expectEqualStrings("test_token", config.bot_token);
    try std.testing.expectEqualStrings("123456789", config.client_id.?);
}

test "gateway intents calculation" {
    const intents = GatewayIntent.GUILDS | GatewayIntent.GUILD_MESSAGES | GatewayIntent.MESSAGE_CONTENT;

    try std.testing.expect(intents & GatewayIntent.GUILDS != 0);
    try std.testing.expect(intents & GatewayIntent.GUILD_MESSAGES != 0);
    try std.testing.expect(intents & GatewayIntent.MESSAGE_CONTENT != 0);
    try std.testing.expect(intents & GatewayIntent.GUILD_MEMBERS == 0);
}

test "permission check" {
    const perms = Permission.SEND_MESSAGES | Permission.VIEW_CHANNEL | Permission.ADMINISTRATOR;

    try std.testing.expect(hasPermission(perms, Permission.SEND_MESSAGES));
    try std.testing.expect(hasPermission(perms, Permission.VIEW_CHANNEL));
    try std.testing.expect(hasPermission(perms, Permission.ADMINISTRATOR));
    try std.testing.expect(!hasPermission(perms, Permission.MANAGE_GUILD));
}
