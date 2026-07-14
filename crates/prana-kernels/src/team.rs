//! Team execution: llama.cpp's threading model, adapted to Prana.
//!
//! ggml dispatches worker threads **once per graph**, and every thread then
//! walks all nodes together — each executing its share of every op — with a
//! two-atomic spin barrier (`ggml_barrier`) between ops and an atomic chunk
//! counter for work stealing inside an op. Per-op synchronization costs
//! ~1-2µs of spinning instead of a publish/notify/join cycle. Prana's
//! measured per-op dispatch cost (~20-60µs across ~170 matmuls per token)
//! was the dominant decode overhead; this module removes it the same way.
//!
//! Pieces:
//! - [`Team`]: a thread's view of one team run (`ith` of `nth`), with
//!   [`Team::barrier`], [`Team::serial`] (thread 0 runs a closure, everyone
//!   else waits), and [`Team::for_chunks`] (barrier-bracketed work stealing).
//! - [`TeamCell`]: a shared mutable slot with runtime-checked borrow guards,
//!   so `prana-model` can orchestrate buffers across a team entirely in safe
//!   code. Parallel *writers* (matmul outputs) exist only inside this
//!   crate's team kernels, which access the cell through
//!   [`TeamCell::ptr_for_writers`] under a documented protocol.
//!
//! This module shares the workspace's unsafe budget with `pool`/`simd_x86`:
//! the unsafe here is the `UnsafeCell` access behind the guard flags plus
//! the kernels' disjoint-range writes.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Shared state for one team run.
pub struct TeamCtx {
    arrived: AtomicU32,
    generation: AtomicU32,
    steal: AtomicU32,
    panicked: AtomicBool,
    nth: u32,
}

impl TeamCtx {
    pub fn new(nth: usize) -> Self {
        Self {
            arrived: AtomicU32::new(0),
            generation: AtomicU32::new(0),
            steal: AtomicU32::new(0),
            panicked: AtomicBool::new(false),
            nth: nth as u32,
        }
    }

    /// Mark the team poisoned (a member panicked); spinning barriers notice
    /// and unwind instead of waiting forever for the dead member.
    pub fn poison(&self) {
        self.panicked.store(true, Ordering::Release);
    }
}

/// One thread's membership in a team run: index `ith` of `nth`.
#[derive(Clone, Copy)]
pub struct Team<'a> {
    pub ith: usize,
    pub nth: usize,
    ctx: &'a TeamCtx,
}

impl<'a> Team<'a> {
    pub fn new(ith: usize, ctx: &'a TeamCtx) -> Self {
        Self { ith, nth: ctx.nth as usize, ctx }
    }

    /// ggml-style generation-counting spin barrier. Panics (unwinding the
    /// member) if another member panicked, so the team never deadlocks.
    pub fn barrier(&self) {
        if self.nth == 1 {
            return;
        }
        let c = self.ctx;
        let gen = c.generation.load(Ordering::Relaxed);
        if c.arrived.fetch_add(1, Ordering::AcqRel) + 1 == c.nth {
            c.arrived.store(0, Ordering::Relaxed);
            c.generation.fetch_add(1, Ordering::Release);
        } else {
            while c.generation.load(Ordering::Acquire) == gen {
                if c.panicked.load(Ordering::Acquire) {
                    panic!("team member panicked");
                }
                std::hint::spin_loop();
            }
        }
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Run `f` on thread 0 only (norms, rope, sampling glue — work too small
    /// to split), bracketed by barriers: the one *before* guarantees every
    /// member finished the previous phase (and dropped its `TeamCell`
    /// guards) before `f` mutates shared cells; the one after publishes
    /// `f`'s effects. The other members spin for the duration; keep `f` at
    /// microsecond scale or give it a `for_chunks` instead.
    pub fn serial(&self, f: impl FnOnce()) {
        self.barrier();
        if self.ith == 0 {
            f();
        }
        self.barrier();
    }

    /// Execute `f(chunk)` for every chunk in `0..n_chunks` across the team,
    /// with atomic work stealing (fast members absorb slow ones' backlog —
    /// on hybrid P+E parts equal splits stall on the E-cores). Brackets the
    /// work in barriers: entry so the steal counter reset is seen, exit so
    /// every member sees all chunks' writes.
    pub fn for_chunks(&self, n_chunks: usize, f: impl Fn(usize)) {
        if self.ith == 0 {
            self.ctx.steal.store(self.nth as u32, Ordering::Relaxed);
        }
        self.barrier();
        let mut c = self.ith;
        while c < n_chunks {
            f(c);
            c = self.ctx.steal.fetch_add(1, Ordering::Relaxed) as usize;
        }
        self.barrier();
    }
}

/// Run `f` once per pool thread as a team (one dispatch, barriers inside).
/// If any member panics the team is poisoned: members blocked at barriers
/// unwind too, and the panic re-raises here.
pub fn run_team(f: impl Fn(Team) + Sync) {
    let pool = crate::pool::global();
    let ctx = TeamCtx::new(pool.threads);
    if pool.threads == 1 {
        f(Team::new(0, &ctx));
        return;
    }
    /// Poisons the team when its member unwinds, so no barrier deadlocks.
    struct PoisonOnUnwind<'a>(&'a TeamCtx);
    impl Drop for PoisonOnUnwind<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                self.0.poison();
            }
        }
    }
    pool.run_team(&|ith| {
        let _poison = PoisonOnUnwind(&ctx);
        f(Team::new(ith, &ctx));
    });
}

/// A shared mutable slot for team runs, with runtime-checked access:
/// - [`TeamCell::get_mut`] — exclusive access (serial sections); panics if
///   any guard is live.
/// - [`TeamCell::read`] — shared access (parallel readers); panics if a
///   writer guard is live.
///
/// The checks make the safe API sound (no aliasing `&mut` can be produced),
/// at one atomic op per guard. Team kernels inside this crate additionally
/// write *disjoint ranges* concurrently via [`TeamCell::ptr_for_writers`],
/// under the barrier protocol documented there.
pub struct TeamCell<T> {
    /// Bit 31 = writer live; low bits = reader count.
    state: AtomicU32,
    value: UnsafeCell<T>,
}

// SAFETY: all access to `value` goes through the guard state machine below
// (or the kernels' documented disjoint-write protocol); `T: Send` because a
// guard can hand the value to another thread.
unsafe impl<T: Send> Sync for TeamCell<T> {}

const WRITER: u32 = 1 << 31;

impl<T> TeamCell<T> {
    pub fn new(value: T) -> Self {
        Self { state: AtomicU32::new(0), value: UnsafeCell::new(value) }
    }

    /// Exclusive guard. Panics if any reader or writer guard is live —
    /// that's a protocol bug in the orchestrator, not a runtime condition.
    pub fn get_mut(&self) -> TeamCellMut<'_, T> {
        if self.state.compare_exchange(0, WRITER, Ordering::Acquire, Ordering::Relaxed).is_err() {
            panic!("TeamCell::get_mut while other guards are live");
        }
        TeamCellMut { cell: self }
    }

    /// Shared read guard. Panics if a writer guard is live.
    pub fn read(&self) -> TeamCellRef<'_, T> {
        let prev = self.state.fetch_add(1, Ordering::Acquire);
        if prev & WRITER != 0 {
            self.state.fetch_sub(1, Ordering::Release);
            panic!("TeamCell::read while a writer guard is live");
        }
        TeamCellRef { cell: self }
    }

}

impl TeamCell<Vec<f32>> {
    /// Base pointer + length for team-kernel writers. Panics if any guard
    /// is live — writes and guard-based access must not overlap.
    ///
    /// The pointer is used under this protocol (upheld by [`team_fill_rows`]):
    /// concurrent writers write disjoint ranges only, and a [`Team::barrier`]
    /// separates the writes from subsequent guard-based access.
    fn writer_ptr(&self) -> (SendPtr, usize) {
        assert_eq!(self.state.load(Ordering::Acquire), 0, "guards live during team write");
        // SAFETY: no guards are live (checked above), so reading the Vec's
        // metadata is race-free.
        let v = unsafe { &mut *self.value.get() };
        (SendPtr(v.as_mut_ptr()), v.len())
    }
}

/// Raw f32 pointer that team chunks share; ranges written are disjoint.
struct SendPtr(*mut f32);
// SAFETY: the pointer is only dereferenced for disjoint ranges under the
// team_fill_rows protocol; the pointee outlives the team run (borrowed cell).
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Below this much total work a split costs more than it saves: the two
/// bracketing barriers exist either way, and a split's exit waits on the
/// slowest (E-)core, so tiny ops run whole on thread 0.
const MIN_MACS_TO_SPLIT: usize = 32 * 1024;

/// Fill `n_slices` disjoint `width`-wide slices of `out` across the team:
/// `f(i, slice)` writes slice `i` (attention heads, norm rows, ...).
/// `slice_macs` estimates one slice's work.
pub fn team_fill_slices(
    team: &Team,
    out: &TeamCell<Vec<f32>>,
    n_slices: usize,
    width: usize,
    slice_macs: usize,
    f: impl Fn(usize, &mut [f32]),
) {
    // Entry barrier BEFORE touching the cell: members drop their previous
    // phase's read guards before arriving here, so the writer_ptr guard
    // check cannot race a slow member's still-live guard.
    if team.ith == 0 {
        team.ctx.steal.store(team.nth as u32, Ordering::Relaxed);
    }
    team.barrier();
    let (ptr, len) = out.writer_ptr();
    assert!(n_slices * width <= len, "output cell too small");

    if n_slices * slice_macs < MIN_MACS_TO_SPLIT {
        if team.ith == 0 {
            // SAFETY: exclusive — thread 0 only, between barriers, no
            // guards live (writer_ptr checked after the entry barrier).
            let slot = unsafe { std::slice::from_raw_parts_mut(ptr.0, n_slices * width) };
            for (i, s) in slot.chunks_exact_mut(width).enumerate() {
                f(i, s);
            }
        }
        team.barrier();
        return;
    }

    let mut i = team.ith;
    while i < n_slices {
        // SAFETY: the steal counter hands out each index exactly once, so
        // the [i*width, (i+1)*width) ranges are disjoint; no guards are
        // live; the exit barrier orders writes before subsequent readers.
        let slot = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(i * width), width) };
        f(i, slot);
        i = team.ctx.steal.fetch_add(1, Ordering::Relaxed) as usize;
    }
    team.barrier();
}

/// Fill `out[..n_rows]` with `dot(r)`, split across the team with work
/// stealing and bracketed by barriers (all members see the writes after).
/// This is the parallel core of every team matmul.
pub fn team_fill_rows(team: &Team, out: &TeamCell<Vec<f32>>, n_rows: usize, dot: impl Fn(usize) -> f32 + Sync) {
    team_fill_rows_weighted(team, out, n_rows, 1, dot)
}

/// [`team_fill_rows`] with an explicit per-row cost (`k` MACs) for the
/// split-vs-thread-0 decision. Barrier discipline as in
/// [`team_fill_slices`]: entry barrier first, then the guard check.
pub fn team_fill_rows_weighted(
    team: &Team,
    out: &TeamCell<Vec<f32>>,
    n_rows: usize,
    row_macs: usize,
    dot: impl Fn(usize) -> f32 + Sync,
) {
    // ~4 chunks per member smooths P/E-core imbalance without shrinking
    // chunks below prefetch-friendly runs.
    let chunk = n_rows.div_ceil(team.nth * 4).max(16);
    let n_chunks = n_rows.div_ceil(chunk);

    if team.ith == 0 {
        team.ctx.steal.store(team.nth as u32, Ordering::Relaxed);
    }
    team.barrier();
    let (ptr, len) = out.writer_ptr();
    assert!(n_rows <= len, "output cell too small");

    if n_rows * row_macs < MIN_MACS_TO_SPLIT {
        if team.ith == 0 {
            // SAFETY: exclusive — thread 0 only, between barriers, no
            // guards live (writer_ptr checked after the entry barrier).
            let slot = unsafe { std::slice::from_raw_parts_mut(ptr.0, n_rows) };
            for (r, o) in slot.iter_mut().enumerate() {
                *o = dot(r);
            }
        }
        team.barrier();
        return;
    }

    let mut c = team.ith;
    while c < n_chunks {
        let start = c * chunk;
        let n = chunk.min(n_rows - start);
        // SAFETY: the steal counter hands out each chunk exactly once, so
        // [start, start+n) ranges are disjoint across writers; no guards
        // are live (writer_ptr checked after the entry barrier); the exit
        // barrier orders these writes before any subsequent reader.
        let slot = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(start), n) };
        for (i, o) in slot.iter_mut().enumerate() {
            *o = dot(start + i);
        }
        c = team.ctx.steal.fetch_add(1, Ordering::Relaxed) as usize;
    }
    team.barrier();
}

pub struct TeamCellMut<'a, T> {
    cell: &'a TeamCell<T>,
}

impl<T> std::ops::Deref for TeamCellMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the WRITER bit grants exclusive access until drop.
        unsafe { &*self.cell.value.get() }
    }
}
impl<T> std::ops::DerefMut for TeamCellMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as above; `&mut self` prevents aliasing through the guard.
        unsafe { &mut *self.cell.value.get() }
    }
}
impl<T> Drop for TeamCellMut<'_, T> {
    fn drop(&mut self) {
        self.cell.state.fetch_and(!WRITER, Ordering::Release);
    }
}

pub struct TeamCellRef<'a, T> {
    cell: &'a TeamCell<T>,
}

impl<T> std::ops::Deref for TeamCellRef<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: reader count > 0 keeps writers out until drop.
        unsafe { &*self.cell.value.get() }
    }
}
impl<T> Drop for TeamCellRef<'_, T> {
    fn drop(&mut self) {
        self.cell.state.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn team_of_one_runs_everything_inline() {
        let ctx = TeamCtx::new(1);
        let team = Team::new(0, &ctx);
        let hits = AtomicUsize::new(0);
        team.barrier();
        team.serial(|| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        team.for_chunks(5, |_| {
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn barriers_and_chunks_across_real_threads() {
        let nth = 4;
        let ctx = TeamCtx::new(nth);
        let counter = AtomicUsize::new(0);
        let hits: Vec<AtomicUsize> = (0..37).map(|_| AtomicUsize::new(0)).collect();
        std::thread::scope(|s| {
            for ith in 0..nth {
                let (ctx, counter, hits) = (&ctx, &counter, &hits);
                s.spawn(move || {
                    let team = Team::new(ith, ctx);
                    // Phase 1: everyone bumps, barrier, everyone must see 4.
                    counter.fetch_add(1, Ordering::Relaxed);
                    team.barrier();
                    assert_eq!(counter.load(Ordering::Relaxed), nth);
                    // Keep phase 2's write out of phase 1's asserts: reads
                    // between barriers race anything after the next barrier.
                    team.barrier();
                    // Phase 2: serial runs exactly once.
                    team.serial(|| {
                        counter.fetch_add(100, Ordering::Relaxed);
                    });
                    assert_eq!(counter.load(Ordering::Relaxed), nth + 100);
                    // Phase 3: stolen chunks cover every index exactly once.
                    team.for_chunks(hits.len(), |c| {
                        hits[c].fetch_add(1, Ordering::Relaxed);
                    });
                    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
                });
            }
        });
    }

    #[test]
    fn run_team_visits_every_member_once_and_barriers_work() {
        for _ in 0..50 {
            let seen = std::sync::Mutex::new(Vec::new());
            let sum = AtomicUsize::new(0);
            super::run_team(|team| {
                seen.lock().unwrap().push(team.ith);
                sum.fetch_add(1, Ordering::Relaxed);
                team.barrier();
                assert_eq!(sum.load(Ordering::Relaxed), team.nth);
            });
            let mut ids = seen.into_inner().unwrap();
            ids.sort_unstable();
            let nth = crate::pool::global().threads;
            assert_eq!(ids, (0..nth).collect::<Vec<_>>());
        }
    }

    #[test]
    fn panicking_member_poisons_instead_of_deadlocking() {
        let result = std::panic::catch_unwind(|| {
            super::run_team(|team| {
                if team.ith == 0 {
                    panic!("member exploded");
                }
                // Everyone else waits at a barrier the dead member never
                // reaches; poisoning must unwind them.
                team.barrier();
            });
        });
        assert!(result.is_err());
        // The pool must still be usable afterwards.
        let count = AtomicUsize::new(0);
        super::run_team(|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::Relaxed), crate::pool::global().threads);
    }

    #[test]
    fn cell_guards_enforce_the_protocol() {
        let cell = TeamCell::new(vec![0f32; 8]);
        {
            let mut w = cell.get_mut();
            w[3] = 7.0;
        }
        let r1 = cell.read();
        let r2 = cell.read();
        assert_eq!(r1[3], 7.0);
        assert_eq!(r2[3], 7.0);
        drop((r1, r2));
        assert_eq!(cell.get_mut()[3], 7.0);
    }

    #[test]
    fn cell_panics_on_write_while_reading() {
        let cell = TeamCell::new(0u32);
        let _r = cell.read();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _w = cell.get_mut();
        }))
        .is_err());
    }
}
