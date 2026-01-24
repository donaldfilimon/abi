---
title: "custom-asic-considerations"
tags: ["research", "asic", "hardware", "custom-silicon"]
---
# Custom ASIC Considerations for ABI

> **Document Version:** 1.0
> **Date:** January 2026
> **Status:** Research Phase
> **Related:** [hardware-acceleration-fpga-asic.md](./hardware-acceleration-fpga-asic.md)

## Executive Summary

This document evaluates when custom ASIC (Application-Specific Integrated Circuit) development makes sense for the ABI framework, analyzing the technical, financial, and strategic considerations. While ASICs offer the highest performance and efficiency, they require substantial upfront investment and long development cycles.

### Decision Framework

| Criteria | Threshold for ASIC | ABI Current Status |
|----------|-------------------|-------------------|
| Annual volume | >100,000 units | Not yet applicable |
| Workload stability | >3 years unchanged | Evolving |
| Power budget | <5W required | 75W acceptable |
| Latency requirement | <1μs deterministic | 10ms acceptable |
| Development budget | >$10M available | To be determined |

**Recommendation:** Focus on FPGA development for 18-24 months, then re-evaluate ASIC based on market validation and volume projections.

---

## Table of Contents

1. [When ASICs Make Sense](#1-when-asics-make-sense)
2. [ASIC vs Alternatives](#2-asic-vs-alternatives)
3. [ASIC Development Process](#3-asic-development-process)
4. [Cost Analysis](#4-cost-analysis)
5. [Architecture Options](#5-architecture-options)
6. [Vendor and Partner Landscape](#6-vendor-and-partner-landscape)
7. [Risk Assessment](#7-risk-assessment)
8. [Decision Timeline](#8-decision-timeline)
9. [References](#9-references)

---

## 1. When ASICs Make Sense

### 1.1 Volume Requirements

ASIC development only makes economic sense at scale:

```
Break-Even Analysis:

ASIC NRE Cost: $15M (7nm)
ASIC Unit Cost: $50 (high volume)
FPGA Unit Cost: $6,500 (Alveo U250)

Break-even volume = NRE / (FPGA_cost - ASIC_cost)
                  = $15M / ($6,450)
                  = ~2,300 units

For meaningful ROI (3x return):
  Target volume = 3 × 2,300 = ~7,000 units
  Over 3 years = ~2,300 units/year minimum
```

### 1.2 Performance Requirements

ASICs are justified when other solutions cannot meet requirements:

| Requirement | GPU Achievable | FPGA Achievable | ASIC Needed |
|-------------|----------------|-----------------|-------------|
| 1000 TOPS @ 10W | No | Marginal | Yes |
| <100ns latency | No | Marginal | Yes |
| 1M vectors/sec search | Yes | Yes | Overkill |
| 100 tok/s @ 5W | No | Marginal | Yes |

### 1.3 Market Fit Indicators

ASICs should only be considered when:

1. **Product-market fit established** - Clear demand for hardware acceleration
2. **Workload patterns stable** - Algorithms unlikely to change significantly
3. **Competitive advantage clear** - Performance gap creates market differentiation
4. **Business model validated** - Revenue model supports hardware investment

---

## 2. ASIC vs Alternatives

### 2.1 Comparison Matrix

| Aspect | GPU | FPGA | Structured ASIC | Full Custom ASIC |
|--------|-----|------|-----------------|------------------|
| NRE Cost | $0 | $200K | $2-5M | $15-50M |
| Unit Cost (volume) | $1K-$15K | $5K-$15K | $100-$500 | $20-$100 |
| Time to Market | 0 | 6-12 mo | 12-18 mo | 18-36 mo |
| Performance/W | 1x | 5-10x | 15-30x | 30-100x |
| Flexibility | High | Medium | Low | None |
| Risk | Low | Medium | Medium | High |

### 2.2 Decision Tree

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hardware Selection Decision                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Volume < 1K/year?                                               │
│  ├── Yes → Use GPU or FPGA                                      │
│  └── No ↓                                                        │
│                                                                  │
│  Power budget > 75W?                                             │
│  ├── Yes → Use GPU                                              │
│  └── No ↓                                                        │
│                                                                  │
│  Latency requirement > 1ms?                                      │
│  ├── Yes → FPGA sufficient                                      │
│  └── No ↓                                                        │
│                                                                  │
│  Workload stable for 3+ years?                                   │
│  ├── No → FPGA (reconfigurable)                                 │
│  └── Yes ↓                                                       │
│                                                                  │
│  Volume > 10K/year?                                              │
│  ├── No → Structured ASIC / eFPGA                               │
│  └── Yes ↓                                                       │
│                                                                  │
│  Budget > $15M NRE?                                              │
│  ├── No → Partner with ASIC company                             │
│  └── Yes → Full custom ASIC                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Hybrid Approaches

**eFPGA Integration:**
- Embed FPGA fabric in custom ASIC
- Keep accelerator cores fixed, algorithms flexible
- Vendors: Flex Logix, Achronix, Menta

**Chiplet Architecture:**
- Use existing accelerator chiplets (AMD, Intel)
- Custom I/O and integration chiplet
- Lower NRE ($5-10M) with good performance

---

## 3. ASIC Development Process

### 3.1 Development Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASIC Development Timeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Specification (3-4 months)                            │
│  ├── Architecture definition                                     │
│  ├── Performance modeling                                        │
│  ├── IP selection                                                │
│  └── Power/area budgeting                                        │
│                                                                  │
│  Phase 2: RTL Design (6-9 months)                               │
│  ├── RTL implementation                                          │
│  ├── Functional verification                                     │
│  ├── Synthesis trials                                            │
│  └── FPGA prototyping                                            │
│                                                                  │
│  Phase 3: Physical Design (4-6 months)                          │
│  ├── Synthesis optimization                                      │
│  ├── Floor planning                                              │
│  ├── Place and route                                             │
│  └── Timing closure                                              │
│                                                                  │
│  Phase 4: Signoff (2-3 months)                                  │
│  ├── DRC/LVS                                                     │
│  ├── Timing signoff                                              │
│  ├── Power analysis                                              │
│  └── Tape-out                                                    │
│                                                                  │
│  Phase 5: Silicon (4-6 months)                                  │
│  ├── Fabrication (2-3 months)                                   │
│  ├── Packaging (1 month)                                        │
│  └── Silicon validation (1-2 months)                            │
│                                                                  │
│  Total: 18-28 months                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Team Requirements

| Role | Headcount | Monthly Cost |
|------|-----------|--------------|
| Architect | 2 | $50K |
| RTL Engineers | 8-12 | $150K |
| Verification Engineers | 6-10 | $120K |
| Physical Design | 4-6 | $80K |
| Firmware/Driver | 2-4 | $40K |
| Program Manager | 1-2 | $25K |
| **Total** | **25-36** | **$465K/month** |

**18-month development: ~$8.4M labor cost**

### 3.3 Tool Costs

| Category | Tool | Annual License |
|----------|------|----------------|
| RTL Simulation | Synopsys VCS | $500K |
| Formal Verification | Cadence JasperGold | $300K |
| Synthesis | Synopsys DC | $400K |
| Place & Route | Cadence Innovus | $500K |
| Timing | Synopsys PrimeTime | $300K |
| DRC/LVS | Calibre | $400K |
| **Total** | | **$2.4M/year** |

---

## 4. Cost Analysis

### 4.1 Full NRE Breakdown

**7nm Process Node (TSMC N7):**

| Category | Cost | Notes |
|----------|------|-------|
| Mask Set | $3-5M | ~80 masks |
| Engineering Labor | $8-10M | 25+ engineers, 18 months |
| EDA Tools | $3-4M | 18-month licenses |
| IP Licensing | $2-3M | SerDes, memory, etc. |
| FPGA Prototype | $500K | Pre-silicon validation |
| Silicon Lots | $1-2M | Engineering samples |
| Packaging | $500K | Development |
| Testing | $1M | ATE development |
| **Total NRE** | **$19-27M** | |

### 4.2 Unit Economics

**Volume Pricing (7nm, 100mm² die):**

| Volume | Wafer Cost | Die Cost | Packaged |
|--------|------------|----------|----------|
| 1K | N/A | $200 | $350 |
| 10K | $8K/wafer | $80 | $130 |
| 100K | $6K/wafer | $50 | $80 |
| 1M | $4K/wafer | $30 | $50 |

### 4.3 Total Cost of Ownership (5-year)

**Scenario: 50K units over 5 years**

| Option | Year 1 | Years 2-5 | Total |
|--------|--------|-----------|-------|
| **GPU (A100)** | $75M | $50M/yr | $275M |
| **FPGA (U250)** | $32.5M | $22M/yr | $120M |
| **ASIC** | $25M NRE | $1M/yr | $29M |

**ASIC saves $91M over FPGA at 50K volume**

---

## 5. Architecture Options

### 5.1 ABI Vector Accelerator (AVA)

**Purpose:** Accelerate vector database operations and similarity search

```
┌─────────────────────────────────────────────────────────────────┐
│                      ABI Vector Accelerator                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Vector Distance Engine (VDE)               │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │ 256-wide │  │ Reduction│  │ Top-K    │             │   │
│  │  │ SIMD FMA │→ │ Network  │→ │ Selector │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘             │   │
│  │                                                          │   │
│  │  Throughput: 1 billion distances/second                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Graph Traversal Engine (GTE)                │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │ Neighbor │  │ Distance │  │ Priority │             │   │
│  │  │ Fetch    │→ │ Compute  │→ │ Queue    │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘             │   │
│  │                                                          │   │
│  │  Supports: HNSW, IVF-PQ, Flat index                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  SRAM Cache   │  │  DMA Engine   │  │  PCIe Gen5    │      │
│  │  4 MB         │  │  4 channels   │  │  x16          │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Target Specs:
- 1B vectors/sec @ 768 dimensions
- <10μs latency for 1M vector search
- 10W power consumption
- 50mm² die area (7nm)
```

### 5.2 ABI Inference Accelerator (AIA)

**Purpose:** Accelerate LLM inference with quantized models

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABI Inference Accelerator                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Quantized Matrix Engine (QME)                 │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │ Dequant  │  │ Systolic │  │ Accumul  │             │   │
│  │  │ Pipeline │→ │ Array    │→ │ + Bias   │             │   │
│  │  │ Q4/Q8    │  │ 512×512  │  │ + Act    │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘             │   │
│  │                                                          │   │
│  │  Performance: 100 TOPS (INT4), 50 TOPS (INT8)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Attention Processing Unit (APU)             │   │
│  │                                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │ Q@K^T    │  │ Softmax  │  │ Attn@V   │             │   │
│  │  │ (fused)  │→ │ (stream) │→ │ (fused)  │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘             │   │
│  │                                                          │   │
│  │  Supports: Multi-head, Flash attention, KV-cache        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  KV-Cache     │  │  Weight SRAM  │  │  HBM2e        │      │
│  │  32 MB SRAM   │  │  64 MB        │  │  16 GB        │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Target Specs:
- 100 tokens/sec for 7B model
- 15W power consumption
- 100mm² die area (7nm)
```

### 5.3 Combined ABI Accelerator

**Unified chip combining vector and inference:**

| Component | Area (mm²) | Power (W) |
|-----------|-----------|-----------|
| Vector Distance Engine | 15 | 3 |
| Graph Traversal Engine | 10 | 2 |
| Quantized Matrix Engine | 30 | 8 |
| Attention Processing Unit | 20 | 5 |
| Memory + I/O | 25 | 7 |
| **Total** | **100** | **25** |

---

## 6. Vendor and Partner Landscape

### 6.1 Design Partners

| Company | Specialization | Notable Projects |
|---------|---------------|------------------|
| Broadcom | Custom silicon | Google TPU, Meta MTIA |
| Marvell | AI accelerators | Amazon Graviton |
| Synopsys | DesignWare IP | AI subsystems |
| Cadence | Tensilica DSP | Edge AI |
| Arm | CPU/NPU cores | Ethos NPU |

### 6.2 Foundries

| Foundry | Nodes | AI-Optimized | Min Volume |
|---------|-------|--------------|------------|
| TSMC | 7nm, 5nm, 3nm | Yes | 10K wafers |
| Samsung | 7nm, 5nm, 4nm | Yes | 5K wafers |
| Intel | Intel 4, Intel 3 | Yes | 3K wafers |
| GlobalFoundries | 12nm, 22nm | Limited | 1K wafers |

### 6.3 IP Vendors

| IP Type | Vendors | Cost Range |
|---------|---------|------------|
| PCIe Gen5 | Synopsys, Cadence | $1-2M |
| HBM2e PHY | Synopsys, Rambus | $2-3M |
| Memory Compiler | ARM, Synopsys | $500K-1M |
| SerDes | Synopsys, Cadence | $1-2M |

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Timing closure failure | Medium | High | Early RTL optimization, margin |
| Power overshoot | Medium | Medium | Conservative estimates, DVFS |
| Silicon bugs | Low | Very High | Extensive verification, ECO budget |
| Yield issues | Low | High | Process characterization, redundancy |
| Algorithm changes | Medium | High | FPGA prototyping first |

### 7.2 Business Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Volume not achieved | High | Critical | Partner/licensing model |
| Competitor ASIC | Medium | High | First-mover advantage |
| Technology shift | Medium | High | Modular architecture |
| Funding gap | Medium | High | Staged investment |
| Key person dependency | Medium | Medium | Knowledge documentation |

### 7.3 Risk-Adjusted ROI

**Base case: 50K units, 5 years**
- Expected ROI: 4.1x

**Risk-adjusted (30% probability of failure):**
- Expected ROI: 2.9x

**Breakeven scenario:**
- Minimum viable volume: 12K units

---

## 8. Decision Timeline

### 8.1 Prerequisites for ASIC Decision

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| FPGA prototype validated | Q3 2026 | In progress |
| 10K+ unit demand validated | Q4 2026 | Pending |
| $20M+ funding secured | Q1 2027 | Pending |
| Design partner selected | Q2 2027 | Pending |
| RTL frozen on FPGA | Q3 2027 | Pending |

### 8.2 Recommended Path

```
2026 Q1-Q2: FPGA development and validation
2026 Q3-Q4: Market validation, volume forecasting
2027 Q1: Go/No-Go decision for ASIC
2027 Q2-Q4: ASIC specification and partner selection
2028 Q1-Q4: RTL development and verification
2029 Q1-Q2: Physical design and tape-out
2029 Q3-Q4: Silicon bring-up and production
```

---

## 9. References

### Industry Analysis

- "AI Chip Market Analysis 2025-2030" - McKinsey & Company
- "Custom Silicon for AI: Build vs Buy" - Gartner
- "The Economics of AI Accelerators" - IEEE Micro

### Technical Resources

- "ASIC Design Flow Overview" - Synopsys
- "AI Accelerator Architecture Trends" - Hot Chips 2024
- "Custom Chip Development Guide" - TSMC

### ABI Framework References

- `docs/research/hardware-acceleration-fpga-asic.md` - Overview
- `docs/research/fpga-inference-acceleration.md` - FPGA for LLM
- `src/gpu/backends/fpga/` - FPGA backend implementation

---

*Document prepared for ABI Framework - ASIC Considerations Research*
