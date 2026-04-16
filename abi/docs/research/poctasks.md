# Hardware Acceleration PoC Tasks

This checklist accompanies `hardware-acceleration-fpga-asic.md` and enumerates concrete PoC steps.

- [ ] Select FPGA target (vendor/model)
- [ ] Set up toolchain (Vivado/Vitis or Quartus)
- [ ] Implement simple kernel (matrix multiply / GEMM)
- [ ] Cross-compile and run on dev board or cloud FPGA instance
- [ ] Collect latency and throughput metrics, produce JSON output
- [ ] Compare results against GPU baseline
- [ ] Document findings in `docs/research/hardware-acceleration-fpga-asic.md`

Notes:
- Keep PoC focused on a single representative kernel to reduce scope and time.
- Where possible, leverage vendor examples and existing open-source toolchains.
