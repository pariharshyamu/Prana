//! A persistent worker pool — the piece the llama2.c benchmark showed was
//! missing: OpenMP amortizes thread startup across the whole run, while
//! `thread::scope` pays spawn cost (~50µs) on every matmul, forcing small
//! per-layer projections to run single-threaded.
//!
//! Design: N-1 workers are spawned once (lazily, on first use) and park in a
//! spin-then-wait loop. A call to [`Pool::run`] publishes one job — an erased
//! `&dyn Fn(usize)` plus a chunk counter, behind an `Arc` — bumps an epoch,
//! and wakes everyone. Workers and the submitting thread all pull chunk
//! indices from a shared atomic counter (work stealing), so uneven chunks
//! self-balance. The submitter returns only when every chunk has finished,
//! which is what makes the lifetime erasure sound.
//!
//! This module and `simd_x86` are the workspace's entire `unsafe` surface;
//! here it is exactly one act: erasing the closure's borrow lifetime so it
//! can sit in the shared slot. The SAFETY argument is the completion barrier:
//! workers only dereference the pointer while `remaining > 0`, and `run`
//! does not return until `remaining == 0`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

/// One published job: an erased pointer to the caller's closure plus the
/// chunk-distribution state.
struct Job {
    /// Lifetime-erased `&dyn Fn(usize) + Sync` borrowed from `run`'s caller.
    f: *const (dyn Fn(usize) + Sync),
    n_chunks: u32,
    /// Next chunk index to claim.
    next: AtomicU32,
    /// Chunks not yet finished; `f` is only dereferenced while this is > 0
    /// under a claimed index.
    remaining: AtomicU32,
    /// Set if any chunk panicked (swallowed in the worker, re-raised by the
    /// submitter once).
    panicked: AtomicBool,
    /// Team job: each chunk index is a team membership (`ith`), whose body
    /// contains barriers — so a thread must claim at most ONE, or it would
    /// deadlock waiting for itself at a barrier.
    team: bool,
}

// SAFETY: `f` is only dereferenced between publication and `remaining == 0`,
// and the publishing thread blocks inside `run` for that whole window, keeping
// the borrow alive. The referent is `Sync`, so shared calls are permitted.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

/// Claim and execute chunks until the counter runs out (or, for team jobs,
/// after at most one membership).
fn execute_chunks(job: &Job) {
    // SAFETY: see `Job` — we hold an Arc to the job and remaining > 0 for
    // every index we claim, so the erased borrow is live.
    let f = unsafe { &*job.f };
    loop {
        let i = job.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.n_chunks {
            break;
        }
        if catch_unwind(AssertUnwindSafe(|| f(i as usize))).is_err() {
            job.panicked.store(true, Ordering::Relaxed);
        }
        job.remaining.fetch_sub(1, Ordering::AcqRel);
        if job.team {
            break;
        }
    }
}

struct Shared {
    /// Incremented for each published job; workers watch it to wake up.
    epoch: AtomicU64,
    slot: Mutex<Option<Arc<Job>>>,
    /// Wakes parked workers when a new epoch is published.
    work_cv: Condvar,
    /// Workers currently blocked on `work_cv` — publishing only pays the
    /// kernel notify when someone is actually parked (in the decode loop
    /// workers stay spinning, so the fast path is entirely user-space).
    parked: AtomicU32,
    /// Wakes the submitter when a job's last chunk completes.
    done: Mutex<()>,
    done_cv: Condvar,
    /// True only while the submitter is blocked on `done_cv` (it spins
    /// first); workers skip the kernel notify otherwise.
    submitter_waiting: AtomicBool,
    /// Serializes whole dispatches: one job owns the pool at a time. Team
    /// jobs *require* this — every worker must join the same job, so a
    /// second concurrent job would starve the first's barriers.
    submit: Mutex<()>,
}

pub struct Pool {
    shared: &'static Shared,
    pub threads: usize,
}

/// Spin briefly before parking: decode-loop jobs arrive every few hundred
/// microseconds (separated by the serial norm/rope/attention work between
/// matmuls), so a generous spin keeps workers hot across those gaps —
/// parked workers pay a futex wake before contributing, which on small
/// per-layer projections means they arrive after the work is gone.
// Workers spin this long before parking on the condvar. Deliberately modest:
// on hybrid laptop parts, threads that busy-spin for milliseconds drain the
// package power budget that the actually-working cores need for turbo — a
// larger budget (tried at 2M) measurably *lowered* decode throughput.
const SPIN_ITERS: u32 = 200_000;

/// Opt-in (`PRANA_PIN=1`, Windows x86_64): pin pool thread `i` to logical
/// processor `2*i`. On hybrid Intel parts the P-core hyperthread pairs
/// enumerate first, so this lands one member per physical P-core and keeps
/// the OS from parking barrier-synchronized members on E/LPE cores — every
/// barrier and every redundant-glue section is priced at the *slowest*
/// member, so one straggler taxes the whole team. Heuristic, not topology
/// detection; that's why it is opt-in.
#[cfg(all(windows, target_arch = "x86_64"))]
fn pin_current_thread(i: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("PRANA_PIN").is_ok()) {
        return;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn SetThreadAffinityMask(thread: *mut core::ffi::c_void, mask: usize) -> usize;
    }
    let cpu = (2 * i) % usize::BITS as usize;
    // SAFETY: both calls take only the pseudo-handle for the calling thread
    // and a bitmask; no memory is passed. A failed call (mask outside the
    // process affinity) returns 0 and changes nothing.
    unsafe {
        SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu);
    }
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
fn pin_current_thread(_i: usize) {}

impl Pool {
    fn new(threads: usize) -> Self {
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            epoch: AtomicU64::new(0),
            slot: Mutex::new(None),
            work_cv: Condvar::new(),
            parked: AtomicU32::new(0),
            done: Mutex::new(()),
            done_cv: Condvar::new(),
            submitter_waiting: AtomicBool::new(false),
            submit: Mutex::new(()),
        }));
        pin_current_thread(0);
        for i in 1..threads {
            thread::Builder::new()
                .name("prana-pool".into())
                .spawn(move || {
                    pin_current_thread(i);
                    worker_loop(shared)
                })
                .expect("spawn pool worker");
        }
        Self { shared, threads }
    }

    /// Run `f(chunk)` for every chunk in `0..n_chunks` across the pool,
    /// including the calling thread. Returns when all chunks are done.
    /// A panic inside `f` is re-raised here (once).
    pub fn run(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_chunks == 0 {
            return;
        }
        if n_chunks == 1 || self.threads == 1 {
            for i in 0..n_chunks {
                f(i);
            }
            return;
        }
        self.dispatch(n_chunks, f, false);
    }

    /// Run `f(ith)` exactly once per pool thread (`ith` in `0..threads`),
    /// concurrently — team execution, where `f`'s body may contain
    /// [`crate::team::Team`] barriers. Prefer [`crate::team::run_team`].
    pub fn run_team(&self, f: &(dyn Fn(usize) + Sync)) {
        if self.threads == 1 {
            f(0);
            return;
        }
        self.dispatch(self.threads, f, true);
    }

    fn dispatch(&self, n_chunks: usize, f: &(dyn Fn(usize) + Sync), team: bool) {
        // One job owns the pool at a time (see `Shared::submit`). Held
        // across the re-raise of worker panics, so tolerate poisoning —
        // the lock guards no data, only exclusivity.
        let _submit = self.shared.submit.lock().unwrap_or_else(|e| e.into_inner());

        /// The one unsafe act in this module: forget the closure borrow's
        /// lifetime so it can sit in the shared slot.
        ///
        /// # Safety
        /// The caller (`run`) must not return until no thread can dereference
        /// the pointer again — enforced by waiting for `remaining == 0`.
        fn erase<'a>(f: &'a (dyn Fn(usize) + Sync + 'a)) -> *const (dyn Fn(usize) + Sync + 'static) {
            // SAFETY: fat-pointer layout is identical; only the lifetime is
            // forgotten, and the completion barrier keeps the borrow live for
            // every dereference.
            unsafe { std::mem::transmute(f as *const (dyn Fn(usize) + Sync + 'a)) }
        }
        let erased = erase(f);
        let job = Arc::new(Job {
            f: erased,
            n_chunks: n_chunks as u32,
            next: AtomicU32::new(0),
            remaining: AtomicU32::new(n_chunks as u32),
            panicked: AtomicBool::new(false),
            team,
        });

        {
            let mut slot = self.shared.slot.lock().unwrap();
            *slot = Some(Arc::clone(&job));
            self.shared.epoch.fetch_add(1, Ordering::Release);
        }
        // Spinning workers see the epoch bump directly; the kernel wakeup is
        // only needed (and only paid) for workers parked on the condvar.
        if self.shared.parked.load(Ordering::Acquire) > 0 {
            self.shared.work_cv.notify_all();
        }

        // The submitter works too, then waits out stragglers.
        execute_chunks(&job);
        let mut spins = 0u32;
        while job.remaining.load(Ordering::Acquire) != 0 {
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                let g = self.shared.done.lock().unwrap();
                self.shared.submitter_waiting.store(true, Ordering::Release);
                if job.remaining.load(Ordering::Acquire) == 0 {
                    self.shared.submitter_waiting.store(false, Ordering::Release);
                    break;
                }
                // Timeout bounds any lost-wakeup window to 1ms.
                let _ = self.shared.done_cv.wait_timeout(g, Duration::from_millis(1)).unwrap();
                self.shared.submitter_waiting.store(false, Ordering::Release);
            }
        }

        self.shared.slot.lock().unwrap().take();
        if job.panicked.load(Ordering::Relaxed) {
            panic!("worker pool task panicked");
        }
    }
}

fn worker_loop(shared: &'static Shared) {
    let mut seen_epoch = 0u64;
    loop {
        // Wait for a new epoch: spin first, then park on the condvar.
        let mut spins = 0u32;
        loop {
            let e = shared.epoch.load(Ordering::Acquire);
            if e != seen_epoch {
                seen_epoch = e;
                break;
            }
            if spins < SPIN_ITERS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                let guard = shared.slot.lock().unwrap();
                // Publish parked-ness under the same mutex the publisher's
                // epoch bump happens under: the publisher either sees
                // parked > 0 and notifies, or we see the new epoch here.
                shared.parked.fetch_add(1, Ordering::Release);
                if shared.epoch.load(Ordering::Acquire) == seen_epoch {
                    drop(shared.work_cv.wait(guard).unwrap());
                }
                shared.parked.fetch_sub(1, Ordering::Release);
                spins = 0;
            }
        }

        // Grab the job (briefly), release the lock, then help execute.
        let job = shared.slot.lock().unwrap().clone();
        if let Some(job) = job {
            execute_chunks(&job);
            // The submitter normally spin-waits and sees `remaining` hit 0
            // itself; the kernel notify is only for the parked case.
            if job.remaining.load(Ordering::Acquire) == 0
                && shared.submitter_waiting.load(Ordering::Acquire)
            {
                let _g = shared.done.lock().unwrap();
                shared.done_cv.notify_all();
            }
        }
    }
}

/// The process-wide pool, sized to available parallelism, spawned on first use.
pub fn global() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new(crate::threading::default_threads()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn covers_every_chunk_exactly_once_across_reuse() {
        let pool = global();
        for round in 0..300 {
            let n = 1 + (round % 13);
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "round {round}");
        }
    }

    #[test]
    fn borrowed_output_is_visible_after_run() {
        let pool = global();
        let out: Vec<AtomicUsize> = (0..64).map(|_| AtomicUsize::new(0)).collect();
        pool.run(8, &|c| {
            for (i, v) in out.iter().enumerate().skip(c * 8).take(8) {
                v.store(i * 3, Ordering::Relaxed);
            }
        });
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v.load(Ordering::Relaxed), i * 3);
        }
    }

    #[test]
    fn panic_in_chunk_is_reported_and_pool_survives() {
        let pool = global();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.run(4, &|i| {
                if i == 2 {
                    panic!("chunk exploded");
                }
            });
        }));
        assert!(result.is_err(), "panic must propagate to the submitter");
        // Pool must still work afterwards.
        let count = AtomicUsize::new(0);
        pool.run(6, &|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), 6);
    }
}
