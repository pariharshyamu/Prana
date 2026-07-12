//! A minimal work-splitting `parallel_for`, the safe analogue of the
//! `CactusThreading::parallel_for` used throughout `cactus-kernels`.
//!
//! Cactus ships a bespoke `ThreadPool` (`cactus-kernels/src/threading.h`) with
//! manual core pinning to performance vs. efficiency clusters. That tuning
//! matters on a phone and a real Prana would port it; for the prototype we use
//! `std::thread::scope`, which gives data-parallel fan-out with a compiler
//! guarantee that no worker outlives the borrowed slices — removing an entire
//! class of lifetime bugs the C++ pool has to avoid by convention.

use std::thread;

/// Split `total` units of work across up to `threads` scoped workers, invoking
/// `f(start, end)` on disjoint half-open ranges. Falls back to a direct call for
/// tiny problems where thread spin-up would dominate.
pub fn parallel_for<F>(total: usize, min_parallel: usize, threads: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if total == 0 {
        return;
    }
    let threads = threads.max(1).min(total);
    if threads == 1 || total < min_parallel {
        f(0, total);
        return;
    }

    let chunk = total.div_ceil(threads);
    thread::scope(|scope| {
        for t in 0..threads {
            let start = t * chunk;
            if start >= total {
                break;
            }
            let end = (start + chunk).min(total);
            let f = &f;
            scope.spawn(move || f(start, end));
        }
    });
}

/// Default worker count: available parallelism, capped for sanity on big hosts.
pub fn default_threads() -> usize {
    thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn covers_every_index_exactly_once() {
        let n = 10_000;
        let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        parallel_for(n, 1, 4, |s, e| {
            for h in &hits[s..e] {
                h.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn small_work_runs_inline() {
        // Below the min_parallel gate, work runs as a single inline range.
        let calls = AtomicUsize::new(0);
        let covered = AtomicUsize::new(0);
        parallel_for(3, 1024, 4, |s, e| {
            calls.fetch_add(1, Ordering::Relaxed);
            covered.fetch_add(e - s, Ordering::Relaxed);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(covered.load(Ordering::Relaxed), 3);
    }
}
