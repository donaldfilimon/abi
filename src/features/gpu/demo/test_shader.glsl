#version 450

layout(local_size_x = 256) in;

layout(binding = 0) buffer InputBuffer {
    float data[];
} input_buffer;

layout(binding = 1) buffer OutputBuffer {
    float data[];
} output_buffer;

// Simple compute shader for testing
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < input_buffer.data.length()) {
        // Simple computation: square the input
        output_buffer.data[index] = input_buffer.data[index] * input_buffer.data[index];
    }
}
