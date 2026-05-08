pub fn writeHelpers(self: anytype) !void {
    // CUDA needs helper functions for operations that don't have direct intrinsics
    if (self.config.language == .cuda) {
        try self.writer.writeLine("#ifndef CLAMP_DEFINED");
        try self.writer.writeLine("#define CLAMP_DEFINED");
        try self.writer.writeLine("__device__ __forceinline__ float clamp(float x, float lo, float hi) {");
        try self.writer.writeLine("    return fminf(fmaxf(x, lo), hi);");
        try self.writer.writeLine("}");
        try self.writer.writeLine("#endif");
        try self.writer.newline();

        try self.writer.writeLine("__device__ __forceinline__ float __fract_helper(float x) {");
        try self.writer.writeLine("    return x - floorf(x);");
        try self.writer.writeLine("}");
        try self.writer.newline();

        try self.writer.writeLine("__device__ __forceinline__ float __sign_helper(float x) {");
        try self.writer.writeLine("    return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);");
        try self.writer.writeLine("}");
        try self.writer.newline();
    }
}
