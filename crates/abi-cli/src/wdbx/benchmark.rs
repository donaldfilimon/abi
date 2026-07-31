//! WDBX `benchmark` subcommand: in-process insert/search timing.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use std::time::Instant;

use crate::app::Outcome;
use abi_wdbx::HnswIndex;

pub(crate) const BENCHMARK_HELP: &str = "usage: abi wdbx benchmark [count]\n\nMeasure local insert/search timing for the in-process vector store.\n";

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn percentile_sorted(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let rank = percentile.saturating_mul(samples.len()).saturating_add(99) / 100;
    samples[rank.clamp(1, samples.len()) - 1]
}

fn benchmark_result(count: usize) -> Result<String, String> {
    let mut index = HnswIndex::new(4).map_err(|detail| detail.to_string())?;
    let mut insert_samples = Vec::new();
    insert_samples
        .try_reserve_exact(count)
        .map_err(|detail| detail.to_string())?;
    let insert_start = Instant::now();
    for position in 0..count {
        let x = u16::try_from(position % 97).expect("modulo result fits u16");
        let y = u16::try_from(position % 31).expect("modulo result fits u16");
        let vector = [f32::from(x), f32::from(y), 0.0, 0.0];
        let operation_start = Instant::now();
        let id = u64::try_from(position)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "benchmark vector id overflow".to_owned())?;
        index
            .insert(id, &vector)
            .map_err(|detail| detail.to_string())?;
        index.commit_last_insert(id);
        insert_samples.push(elapsed_ns(operation_start));
    }
    let insert_ns = elapsed_ns(insert_start);

    let query_count = count.min(200);
    let mut search_samples = Vec::new();
    search_samples
        .try_reserve_exact(query_count)
        .map_err(|detail| detail.to_string())?;
    let search_start = Instant::now();
    for _ in 0..query_count {
        let operation_start = Instant::now();
        let results = index
            .search(&[1.0, 0.0, 0.0, 0.0], 10)
            .map_err(|detail| detail.to_string())?;
        search_samples.push(elapsed_ns(operation_start));
        drop(results);
    }
    let search_ns = elapsed_ns(search_start);

    insert_samples.sort_unstable();
    search_samples.sort_unstable();
    let insert_average = if count == 0 {
        0
    } else {
        insert_ns / u64::try_from(count).unwrap_or(u64::MAX)
    };
    let search_average = if query_count == 0 {
        0
    } else {
        search_ns / u64::try_from(query_count).unwrap_or(u64::MAX)
    };
    Ok(format!(
        "benchmark (local, in-memory; not a published throughput claim):\n  inserts: {count} in {insert_ns} ns  (avg {insert_average} ns/op; includes per-op acceleration-kernel dispatch)\n    p50={} ns  p95={} ns  p99={} ns\n  searches: {query_count} in {search_ns} ns (avg {search_average} ns/op, k=10 over {} vectors)\n    p50={} ns  p95={} ns  p99={} ns\n",
        percentile_sorted(&insert_samples, 50),
        percentile_sorted(&insert_samples, 95),
        percentile_sorted(&insert_samples, 99),
        index.len(),
        percentile_sorted(&search_samples, 50),
        percentile_sorted(&search_samples, 95),
        percentile_sorted(&search_samples, 99),
    ))
}

pub(crate) fn run_benchmark(args: &[String]) -> Outcome {
    let count = match args {
        [] => 256,
        [count] => match count.parse::<usize>() {
            Ok(count) => count,
            Err(_) => return super::usage(),
        },
        _ => return super::usage(),
    };
    match benchmark_result(count) {
        Ok(report) => Outcome::stderr(report, 0),
        Err(detail) => super::error("benchmark failed", detail),
    }
}
