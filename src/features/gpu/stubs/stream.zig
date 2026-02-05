const std = @import("std");

pub const Stream = struct {};
pub const StreamOptions = struct {};
pub const StreamPriority = enum { low, normal, high };
pub const StreamFlags = packed struct {};
pub const StreamState = enum { idle, running, @"error" };
pub const StreamManager = struct {};
pub const Event = struct {};
pub const EventOptions = struct {};
pub const EventFlags = packed struct {};
pub const EventState = enum { pending, completed };
