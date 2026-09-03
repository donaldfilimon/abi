//! xtask task runner - Rust port of Python gate scripts.

mod abbey;
mod ci;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "xtask", about = "ABI xtask - Rust port of Python gate scripts", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// CI contract checks (port of `tools/ci_contract.py`)
    Ci(CiArgs),
    /// Abbey contract checks (port of `tools/abbey_contracts.py` + vendor)
    Abbey(AbbeyArgs),
}

#[derive(Parser)]
struct CiArgs {
    #[command(subcommand)]
    command: CiCommands,
}

#[derive(Subcommand)]
enum CiCommands {
    /// Verify Cargo sibling deps, workflow YAML, and runner trust boundary
    Verify(VerifyArgs),
}

#[derive(Parser)]
struct VerifyArgs {
    /// Path to Cargo.toml (default: find from current dir or manifest dir)
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Path to workflow file (default: .github/workflows/ci.yml)
    #[arg(long)]
    workflow: Option<PathBuf>,
    /// Repo root (alternative to manifest/workflow)
    #[arg(long)]
    root: Option<PathBuf>,
}

#[derive(Parser)]
struct AbbeyArgs {
    #[command(subcommand)]
    command: AbbeyCommands,
}

#[derive(Subcommand)]
enum AbbeyCommands {
    /// Verify Abbey contract corpus (strict JSON, digests, aggregate)
    Verify(AbbeyVerifyArgs),
    /// Vendor exact-byte Abbey corpus with lock (exact-byte, atomic publish)
    Vendor(AbbeyVendorArgs),
}

#[derive(Parser)]
struct AbbeyVerifyArgs {
    /// Corpus root (default: contracts/abbey)
    #[arg(default_value = "contracts/abbey")]
    corpus: PathBuf,
}

#[derive(Parser)]
struct AbbeyVendorArgs {
    /// Source corpus root
    #[arg(long)]
    source: PathBuf,
    /// Destination vendor root (parent must exist)
    #[arg(long)]
    destination: PathBuf,
    /// 40-char lowercase hex source revision
    #[arg(long = "source-revision")]
    source_revision: String,
    /// Write vendor tree (atomic publish)
    #[arg(long, conflicts_with = "check")]
    write: bool,
    /// Check vendor tree against source (read-only)
    #[arg(long, conflicts_with = "write")]
    check: bool,
}

fn find_repo_root() -> PathBuf {
    // Try current dir walk up
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if dir.join("Cargo.toml").exists() && dir.join(".github/workflows/ci.yml").exists() {
            return dir;
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    // Fallback: manifest dir's ancestors (compile-time)
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // manifest_dir is crates/xtask, so two parents up is abi root
    if let Some(root) = manifest_dir.ancestors().nth(2)
        && root.join("Cargo.toml").exists()
    {
        return root.to_path_buf();
    }
    // Last fallback: current dir
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Ci(ci_args) => match ci_args.command {
            CiCommands::Verify(args) => run_ci_verify(&args),
        },
        Commands::Abbey(abbey_args) => match abbey_args.command {
            AbbeyCommands::Verify(args) => run_abbey_verify(&args),
            AbbeyCommands::Vendor(args) => run_abbey_vendor(&args),
        },
    }
}

fn run_abbey_verify(args: &AbbeyVerifyArgs) {
    let corpus = if args.corpus.is_absolute() {
        args.corpus.clone()
    } else {
        // Resolve relative to current dir; if not found, try repo root fallback
        let candidate = Path::new(&args.corpus);
        if candidate.exists() {
            candidate.to_path_buf()
        } else {
            find_repo_root().join(&args.corpus)
        }
    };
    match abbey::verify(&corpus) {
        Ok(report) => {
            println!(
                "abbey-contracts: verified {} artifacts ({} bytes), digest={}",
                report.artifact_count, report.total_bytes, report.aggregate_digest
            );
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("abbey-contracts: {e}");
            std::process::exit(1);
        }
    }
}

fn run_abbey_vendor(args: &AbbeyVendorArgs) {
    if !args.write && !args.check {
        eprintln!("abbey-contracts-vendor: --write or --check required");
        std::process::exit(2);
    }
    let check = args.check;
    match abbey::vendor(
        &args.source,
        &args.destination,
        &args.source_revision,
        check,
    ) {
        Ok(report) => {
            let action = if check { "verified" } else { "wrote" };
            println!(
                "abbey-contracts-vendor: {action} {} artifacts ({} bytes), digest={}",
                report.artifact_count, report.total_bytes, report.aggregate_digest
            );
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("abbey-contracts-vendor: {e}");
            std::process::exit(1);
        }
    }
}

fn run_ci_verify(args: &VerifyArgs) {
    let root = args.root.clone().unwrap_or_else(find_repo_root);
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| root.join("Cargo.toml"));
    let workflow_path = args
        .workflow
        .clone()
        .unwrap_or_else(|| root.join(".github/workflows/ci.yml"));

    let cargo_toml = match std::fs::read_to_string(&manifest_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("xtask: failed to read {}: {e}", manifest_path.display());
            std::process::exit(2);
        }
    };
    let workflow = match std::fs::read_to_string(&workflow_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("xtask: failed to read {}: {e}", workflow_path.display());
            std::process::exit(2);
        }
    };

    let failures = ci::validate_workflow(&workflow, &cargo_toml);
    if failures.is_empty() {
        // Byte-identical success path: Python oracle returns () and unittest shows OK.
        // Print nothing or minimal OK for check.sh parsing.
        // We print to stdout for visibility but keep stderr empty for parity.
        println!("ci contract: ok");
        std::process::exit(0);
    }
    for f in &failures {
        eprintln!("{f}");
    }
    std::process::exit(1);
}
