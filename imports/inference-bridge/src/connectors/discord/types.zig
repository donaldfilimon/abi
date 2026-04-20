//! Discord API Types
//!
//! Core type definitions for the Discord API including:
//! - User, Guild, Channel, Message types
//! - Application, Interaction, Component types
//! - Application Command types
//! - Webhook, Voice, Gateway, OAuth2 types

const std = @import("std");

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
    users: ?std.StringHashMapUnmanaged(User) = null,
    members: ?std.StringHashMapUnmanaged(GuildMember) = null,
    roles: ?std.StringHashMapUnmanaged(Role) = null,
    channels: ?std.StringHashMapUnmanaged(Channel) = null,
    messages: ?std.StringHashMapUnmanaged(Message) = null,
    attachments: ?std.StringHashMapUnmanaged(Attachment) = null,
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
    name_localizations: ?std.StringHashMapUnmanaged([]const u8) = null,
    description: []const u8,
    description_localizations: ?std.StringHashMapUnmanaged([]const u8) = null,
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
    name_localizations: ?std.StringHashMapUnmanaged([]const u8) = null,
    description: []const u8,
    description_localizations: ?std.StringHashMapUnmanaged([]const u8) = null,
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
    name_localizations: ?std.StringHashMapUnmanaged([]const u8) = null,
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

    pub const ALL_UNPRIVILEGED: u32 = GUILDS | GUILD_MODERATION |
        GUILD_EMOJIS_AND_STICKERS | GUILD_INTEGRATIONS | GUILD_WEBHOOKS |
        GUILD_INVITES | GUILD_VOICE_STATES | GUILD_MESSAGES |
        GUILD_MESSAGE_REACTIONS | GUILD_MESSAGE_TYPING | DIRECT_MESSAGES |
        DIRECT_MESSAGE_REACTIONS | DIRECT_MESSAGE_TYPING |
        GUILD_SCHEDULED_EVENTS | AUTO_MODERATION_CONFIGURATION |
        AUTO_MODERATION_EXECUTION;

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
// Gateway Bot Info
// ============================================================================

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
