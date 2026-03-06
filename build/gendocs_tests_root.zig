const std = @import("std");

test {
    _ = @import("gendocs_source_cli");
}

test {
    std.testing.refAllDecls(@This());
}
