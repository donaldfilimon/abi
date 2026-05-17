const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    const gpu = abi.features.gpu;
    const status = gpu.nativeKernelStatus();
    std.debug.print("GPU native kernel status: linked={s}, backend={s}, message={s}\n", .{
        if (status.linked) "true" else "false",
        std.fmt.allocPrint(std.heap.page_allocator, "{}", .{status.backend}) catch "unknown",
        status.message
    });
    
    const backend_status = gpu.detectBackend();
    std.debug.print("GPU backend status: available={s}, accelerated={s}, backend={s}, message={s}\n", .{
        if (backend_status.available) "true" else "false",
        if (backend_status.accelerated) "true" else "false",
        std.fmt.allocPrint(std.heap.page_allocator, "{}", .{backend_status.backend}) catch "unknown",
        backend_status.message
    });
}
