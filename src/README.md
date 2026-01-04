//! src Directory Overview
//!
//! This directory contains the core source modules of the ABI framework.
//! The layout follows a feature‑centric organization that aligns with the
//! project's roadmap and build system.
//!
//! **Sub‑directories**
//! - `compute` – Low‑level compute primitives and runtime.
//! - `core` – Core utilities and profiling.
//! - `features` – Optional feature implementations (AI, GPU, networking, …).
//! - `framework` – High‑level orchestration layer.
//! - `shared` – Reusable utilities (logging, platform abstractions, plugins, …).
//!
//! Each sub‑folder contains a `README.md` that explains its purpose and how to
//! extend it.
//!
//! Top‑level sub‑directories:
//!
//! - **compute** – Low‑level compute primitives and runtime helpers.
//! - **core** – Core utilities such as profiling, the main entry point, and
//!   documentation for the core package.
//! - **features** – Implementation of optional and optional‑style features. Each
//!   feature lives in its own sub‑folder (e.g., `ai`, `gpu`, `network`, `web`).
//!   Inside each folder a `mod.zig` aggregates the public API for that feature.
//! - **framework** – High‑level framework glue code that ties the core and
//!   feature modules together.
//! - **shared** – Reusable utilities shared across the codebase (logging,
//!   observability, platform abstractions, plugins, and a rich collection of
//!   helper modules under `utils`).
//!
//! Each sub‑directory includes its own `README.md` with more detailed
//! information and build instructions.
//!
//! The layout matches the conventions described in `AGENTS.md` and is kept
//! up‑to‑date as new features are added.
