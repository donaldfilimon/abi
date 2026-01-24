const std = @import("std");
pub fn main() !void {
    const Threaded = std.Io.Threaded;
    const InitOptions = @typeInfo(@TypeOf(Threaded.init)).Fn.params[1].type.?;
    @compileLog("InitOptions fields:", @typeInfo(InitOptions).Struct.fields);
}
