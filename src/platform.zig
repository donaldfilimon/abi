//! Platform-specific optimizations and abstractions

const std = @import("std");
const builtin = @import("builtin");

pub const PlatformLayer = struct {
    /// iOS-specific optimizations for a-Shell
    pub const iOS = struct {
        const max_memory = 256 * 1024 * 1024; // 256MB limit
        const max_file_handles = 256;

        pub fn init() !void {
            // Set up iOS-specific memory limits
            if (builtin.os.tag == .ios) {
                // Configure memory pressure handler
                const dispatch = @cImport({
                    @cInclude("dispatch/dispatch.h");
                });

                dispatch.dispatch_source_set_event_handler(dispatch.dispatch_source_create(dispatch.DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0, dispatch.DISPATCH_MEMORYPRESSURE_WARN, dispatch.dispatch_get_main_queue()), struct {
                    fn handler() callconv(.C) void {
                        // Aggressive memory cleanup placeholder
                    }
                }.handler);
            }
        }

        pub fn openFile(path: []const u8) !std.fs.File {
            // iOS sandbox restrictions
            const allowed_prefixes = [_][]const u8{
                "~/Documents/",
                "~/tmp/",
                "/private/var/mobile/",
            };

            for (allowed_prefixes) |prefix| {
                if (std.mem.startsWith(u8, path, prefix)) {
                    return std.fs.cwd().openFile(path, .{});
                }
            }

            return error.SandboxViolation;
        }
    };

    /// Windows-specific console optimizations
    pub const Windows = struct {
        pub const ConPTY = struct {
            handle: *anyopaque,
            input: *anyopaque,
            output: *anyopaque,
        };
        pub fn enableAnsiColors() !void {
            const kernel32 = @cImport({
                @cInclude("windows.h");
            });

            const stdout_handle = kernel32.GetStdHandle(kernel32.STD_OUTPUT_HANDLE);
            var mode: kernel32.DWORD = 0;

            if (kernel32.GetConsoleMode(stdout_handle, &mode) == 0) {
                return error.GetConsoleModeFailed;
            }

            mode |= kernel32.ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            mode |= kernel32.ENABLE_PROCESSED_OUTPUT;

            if (kernel32.SetConsoleMode(stdout_handle, mode) == 0) {
                return error.SetConsoleModeFailed;
            }
        }

        pub fn createConPTY(cols: u16, rows: u16) !ConPTY {
            const kernel32 = @cImport({
                @cInclude("windows.h");
                @cInclude("consoleapi.h");
            });

            const size = kernel32.COORD{ .X = cols, .Y = rows };
            var input_pipe: kernel32.HANDLE = undefined;
            var output_pipe: kernel32.HANDLE = undefined;
            var pty: kernel32.HPCON = undefined;

            // Create pipes
            if (kernel32.CreatePipe(&input_pipe, null, null, 0) == 0) {
                return error.CreatePipeFailed;
            }

            if (kernel32.CreatePipe(null, &output_pipe, null, 0) == 0) {
                return error.CreatePipeFailed;
            }

            // Create pseudo console
            const hr = kernel32.CreatePseudoConsole(size, input_pipe, output_pipe, 0, &pty);

            if (hr != kernel32.S_OK) {
                return error.CreatePseudoConsoleFailed;
            }

            return ConPTY{
                .handle = pty,
                .input = input_pipe,
                .output = output_pipe,
            };
        }
    };

    /// Linux io_uring for maximum async I/O performance
    pub const Linux = struct {
        pub const AsyncIO = struct {
            ring: std.os.linux.io_uring,
            submission_queue: []std.os.linux.io_uring_sqe,
            completion_queue: []std.os.linux.io_uring_cqe,

            pub fn init(queue_depth: u13) !AsyncIO {
                var ring: std.os.linux.io_uring = undefined;
                const params = std.os.linux.io_uring_params{};

                try std.os.linux.io_uring_setup(queue_depth, &params, &ring);

                return AsyncIO{
                    .ring = ring,
                    .submission_queue = undefined, // Mapped separately
                    .completion_queue = undefined,
                };
            }

            pub fn readFile(self: *AsyncIO, path: []const u8, buffer: []u8) !usize {
                const fd = try std.os.open(path, .{ .ACCMODE = .RDONLY }, 0);
                defer std.os.close(fd);

                // Get submission queue entry
                const sqe = try self.getSQE();
                std.os.linux.io_uring_prep_read(sqe, fd, buffer.ptr, buffer.len, 0);
                sqe.user_data = 1;

                // Submit and wait
                _ = try std.os.linux.io_uring_submit(&self.ring);

                var cqe: *std.os.linux.io_uring_cqe = undefined;
                _ = try std.os.linux.io_uring_wait_cqe(&self.ring, &cqe);
                defer std.os.linux.io_uring_cqe_seen(&self.ring, cqe);

                if (cqe.res < 0) {
                    return error.ReadFailed;
                }

                return @intCast(cqe.res);
            }
        };
    };

    /// macOS unified memory optimizations
    pub const macOS = struct {
        pub fn createUnifiedBuffer(size: usize) ![]u8 {
            const mach = @cImport({
                @cInclude("mach/mach.h");
                @cInclude("mach/vm_map.h");
            });

            var address: mach.vm_address_t = 0;
            const kr = mach.vm_allocate(mach.mach_task_self(), &address, size, mach.VM_FLAGS_ANYWHERE);

            if (kr != mach.KERN_SUCCESS) {
                return error.VmAllocateFailed;
            }

            // Mark as purgeable for memory pressure handling
            var state: mach.vm_purgable_t = mach.VM_PURGABLE_NONVOLATILE;
            _ = mach.vm_purgable_control(mach.mach_task_self(), address, mach.VM_PURGABLE_SET_STATE, &state);

            return @as([*]u8, @ptrFromInt(address))[0..size];
        }
    };
};
