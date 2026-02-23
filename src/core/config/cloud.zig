//! Cloud Functions Configuration
//!
//! Configuration for cloud function adapters supporting AWS Lambda,
//! Google Cloud Functions, and Azure Functions.

const std = @import("std");

/// Cloud provider selection.
pub const CloudProvider = enum {
    /// Auto-detect based on environment variables.
    auto,
    /// AWS Lambda.
    aws_lambda,
    /// Google Cloud Functions.
    gcp_functions,
    /// Azure Functions.
    azure_functions,
};

/// Cloud functions configuration.
pub const CloudConfig = struct {
    /// Target cloud provider (auto-detect if not specified).
    provider: CloudProvider = .auto,

    /// Memory allocation in MB.
    memory_mb: u32 = 256,

    /// Function timeout in seconds.
    timeout_seconds: u32 = 30,

    /// Enable distributed tracing.
    tracing_enabled: bool = false,

    /// Enable structured logging.
    logging_enabled: bool = true,

    /// Log level for cloud function execution.
    log_level: LogLevel = .info,

    /// Enable cold start optimization.
    cold_start_optimization: bool = true,

    /// Enable response compression.
    compression_enabled: bool = true,

    /// CORS configuration for HTTP functions.
    cors: ?CorsConfig = null,

    /// AWS-specific configuration.
    aws: ?AwsConfig = null,

    /// GCP-specific configuration.
    gcp: ?GcpConfig = null,

    /// Azure-specific configuration.
    azure: ?AzureConfig = null,

    pub const LogLevel = enum {
        debug,
        info,
        warn,
        @"error",

        pub fn toString(self: LogLevel) []const u8 {
            return @tagName(self);
        }
    };

    pub const CorsConfig = struct {
        /// Allowed origins (use "*" for any).
        allowed_origins: []const []const u8 = &.{"*"},
        /// Allowed HTTP methods.
        allowed_methods: []const []const u8 = &.{ "GET", "POST", "PUT", "DELETE", "OPTIONS" },
        /// Allowed headers.
        allowed_headers: []const []const u8 = &.{ "Content-Type", "Authorization", "X-Request-ID" },
        /// Exposed headers.
        exposed_headers: []const []const u8 = &.{},
        /// Max age for preflight cache (seconds).
        max_age_seconds: u32 = 86400,
        /// Allow credentials.
        allow_credentials: bool = false,
    };

    pub const AwsConfig = struct {
        /// AWS region.
        region: ?[]const u8 = null,
        /// Lambda architecture (x86_64 or arm64).
        architecture: Architecture = .x86_64,
        /// Enable Lambda SnapStart for faster cold starts.
        snap_start_enabled: bool = false,
        /// Enable Lambda Insights for monitoring.
        insights_enabled: bool = false,
        /// Provisioned concurrency (0 = disabled).
        provisioned_concurrency: u32 = 0,
        /// Reserved concurrency (0 = unreserved).
        reserved_concurrency: u32 = 0,
        /// VPC configuration.
        vpc: ?VpcConfig = null,

        pub const Architecture = enum {
            x86_64,
            arm64,
        };

        pub const VpcConfig = struct {
            subnet_ids: []const []const u8 = &.{},
            security_group_ids: []const []const u8 = &.{},
        };
    };

    pub const GcpConfig = struct {
        /// GCP project ID.
        project_id: ?[]const u8 = null,
        /// GCP region.
        region: ?[]const u8 = null,
        /// Cloud Functions generation (1 or 2).
        generation: u8 = 2,
        /// Maximum instances (0 = unlimited).
        max_instances: u32 = 0,
        /// Minimum instances for warm starts.
        min_instances: u32 = 0,
        /// VPC connector for private network access.
        vpc_connector: ?[]const u8 = null,
        /// Ingress settings.
        ingress: IngressSetting = .all,
        /// Service account email.
        service_account: ?[]const u8 = null,

        pub const IngressSetting = enum {
            all,
            internal_only,
            internal_and_gclb,
        };
    };

    pub const AzureConfig = struct {
        /// Azure subscription ID.
        subscription_id: ?[]const u8 = null,
        /// Azure resource group.
        resource_group: ?[]const u8 = null,
        /// Azure region.
        region: ?[]const u8 = null,
        /// Function app plan (consumption, premium, dedicated).
        plan: PlanType = .consumption,
        /// Maximum scale out instances.
        max_scale_out: u32 = 200,
        /// Pre-warmed instances (Premium plan only).
        pre_warmed_instances: u32 = 1,
        /// Application Insights key.
        app_insights_key: ?[]const u8 = null,

        pub const PlanType = enum {
            consumption,
            premium,
            dedicated,
        };
    };

    /// Default configuration.
    pub fn defaults() CloudConfig {
        return .{};
    }

    /// Configuration optimized for low latency.
    pub fn lowLatency() CloudConfig {
        return .{
            .memory_mb = 512,
            .cold_start_optimization = true,
            .aws = .{
                .snap_start_enabled = true,
                .architecture = .arm64,
            },
            .gcp = .{
                .min_instances = 1,
            },
            .azure = .{
                .plan = .premium,
                .pre_warmed_instances = 2,
            },
        };
    }

    /// Configuration optimized for cost.
    pub fn costOptimized() CloudConfig {
        return .{
            .memory_mb = 128,
            .timeout_seconds = 15,
            .cold_start_optimization = false,
            .aws = .{
                .architecture = .arm64, // ARM is cheaper
            },
            .gcp = .{
                .min_instances = 0,
            },
            .azure = .{
                .plan = .consumption,
            },
        };
    }

    /// Configuration optimized for high throughput.
    pub fn highThroughput() CloudConfig {
        return .{
            .memory_mb = 1024,
            .timeout_seconds = 60,
            .aws = .{
                .reserved_concurrency = 100,
            },
            .gcp = .{
                .max_instances = 1000,
            },
            .azure = .{
                .max_scale_out = 500,
            },
        };
    }
};
