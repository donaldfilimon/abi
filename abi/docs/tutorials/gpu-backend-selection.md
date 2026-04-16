# GPU Backend Selection

This guide helps choose between `metal`, `cuda`, `vulkan`, and other backends based on platform and constraints.

## Considerations
- Target OS (macOS -> Metal), hardware availability (NVIDIA -> CUDA)
- Feature parity and performance tradeoffs
- Driver support and installation complexity

## Example
- For macOS dev: prefer `-Dgpu-backend=metal`
- For Linux with NVIDIA: `-Dgpu-backend=cuda`
