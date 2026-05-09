pub const client = @import("client.zig");
pub const engine = @import("engine.zig");
pub const server = @import("server.zig");
pub const discord_bot = @import("discord.zig");
pub const ralph_multi = @import("ralph_multi.zig");
pub const ralph_swarm = @import("ralph_swarm.zig");
pub const custom_framework = @import("custom_framework.zig");

// Re-exports
pub const ChatMessage = client.ChatMessage;
pub const CompletionRequest = client.CompletionRequest;
pub const CompletionResponse = client.CompletionResponse;
pub const StreamChunk = client.StreamChunk;
pub const LLMClient = client.LLMClient;
pub const EchoBackend = client.EchoBackend;
pub const createClient = client.createClient;
pub const ClientWrapper = client.ClientWrapper;
pub const RetryHandler = client.RetryHandler;

pub const AbbeyEngine = engine.AbbeyEngine;
pub const EngineResponse = engine.Response;
pub const EngineStats = engine.EngineStats;

pub const HttpServerConfig = server.ServerConfig;
pub const AbbeyServerConfig = server.AbbeyServerConfig;
pub const ServerError = server.ServerError;
pub const serveHttp = server.serve;
pub const serveHttpWithConfig = server.serveWithConfig;

pub const AbbeyDiscordBot = discord_bot.AbbeyDiscordBot;
pub const DiscordBotConfig = discord_bot.DiscordBotConfig;
pub const DiscordBotError = discord_bot.DiscordBotError;
pub const SessionManager = discord_bot.SessionManager;
pub const BotStats = discord_bot.BotStats;
pub const GatewayBridge = discord_bot.GatewayBridge;
pub const GatewayStats = discord_bot.GatewayStats;
pub const AbbeyCommands = discord_bot.AbbeyCommands;

pub const CustomAI = custom_framework.CustomAI;
pub const CustomAIConfig = custom_framework.CustomAIConfig;
pub const ProfileTemplate = custom_framework.ProfileTemplate;
pub const CustomAIBuilder = custom_framework.Builder;
pub const CustomAIResponse = custom_framework.Response;
pub const CustomAIStats = custom_framework.Stats;
pub const Stats = custom_framework.Stats;

pub const createCustomAI = custom_framework.create;
pub const createFromProfile = custom_framework.createFromProfile;
pub const createWithSeedPrompt = custom_framework.createWithSeedPrompt;
pub const createResearcher = custom_framework.createResearcher;
pub const createCoder = custom_framework.createCoder;
pub const createWriter = custom_framework.createWriter;
pub const createCompanion = custom_framework.createCompanion;
pub const createOpinionated = custom_framework.createOpinionated;
