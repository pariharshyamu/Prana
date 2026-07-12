use std::collections::HashMap;

/// A size-bucketed buffer pool, the safe-Rust analogue of Cactus's `BufferPool`
/// (`cactus-graph/src/core.cpp`).
///
/// Transient activation buffers during graph execution are recycled instead of
/// being freed, which is what lets an inference engine keep a flat memory
/// profile across many decode steps. Cactus does this with `new char[]` /
/// `unique_ptr<char[]>` and manual accounting; here each recycled buffer is a
/// plain `Vec<u8>`, so there is no way to leak it, double-free it, or hand out a
/// dangling pointer — the borrow checker and `Drop` handle teardown.
#[derive(Default)]
pub struct BufferPool {
    free: HashMap<usize, Vec<Vec<u8>>>,
    active_bytes: usize,
    pooled_bytes: usize,
    peak_bytes: usize,
}

impl BufferPool {
    const ALIGNMENT: usize = 64;
    const MIN_BUCKET: usize = 1024;

    pub fn new() -> Self {
        Self::default()
    }

    /// Round a request up to a power-of-two bucket (min 1 KiB), matching the
    /// Cactus bucketing so pool reuse behaviour is comparable.
    fn round_up(size: usize) -> usize {
        if size <= Self::MIN_BUCKET {
            return Self::MIN_BUCKET;
        }
        let aligned = (size + Self::ALIGNMENT - 1) & !(Self::ALIGNMENT - 1);
        let mut bucket = Self::MIN_BUCKET;
        while bucket < aligned {
            bucket *= 2;
        }
        bucket
    }

    /// Acquire a zeroed buffer of at least `byte_size` bytes.
    pub fn acquire(&mut self, byte_size: usize) -> Vec<u8> {
        if byte_size == 0 {
            return Vec::new();
        }
        let bucket = Self::round_up(byte_size);
        self.active_bytes += bucket;
        self.peak_bytes = self.peak_bytes.max(self.active_bytes);

        if let Some(list) = self.free.get_mut(&bucket) {
            if let Some(mut buf) = list.pop() {
                self.pooled_bytes -= bucket;
                buf.clear();
                buf.resize(bucket, 0);
                return buf;
            }
        }
        vec![0u8; bucket]
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, buf: Vec<u8>) {
        let cap = buf.capacity();
        if cap == 0 {
            return;
        }
        let bucket = Self::round_up(cap);
        // active_bytes is a coarse accounting counter; saturate to stay honest
        // if a foreign buffer is released.
        self.active_bytes = self.active_bytes.saturating_sub(bucket);
        self.pooled_bytes += bucket;
        self.free.entry(bucket).or_default().push(buf);
    }

    /// Bytes currently checked out of the pool.
    pub fn active_bytes(&self) -> usize {
        self.active_bytes
    }

    /// Bytes held in the free lists.
    pub fn pooled_bytes(&self) -> usize {
        self.pooled_bytes
    }

    /// High-water mark of `active_bytes` over the pool's lifetime.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Drop all cached free buffers.
    pub fn clear(&mut self) {
        self.free.clear();
        self.pooled_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recycles_same_bucket() {
        let mut pool = BufferPool::new();
        let a = pool.acquire(2000);
        assert_eq!(a.len(), 2048);
        assert_eq!(pool.active_bytes(), 2048);
        pool.release(a);
        assert_eq!(pool.active_bytes(), 0);
        assert_eq!(pool.pooled_bytes(), 2048);
        // Second acquire of the same bucket must reuse, not grow the pool.
        let b = pool.acquire(1500);
        assert_eq!(b.len(), 2048);
        assert_eq!(pool.pooled_bytes(), 0);
        assert_eq!(pool.peak_bytes(), 2048);
    }

    #[test]
    fn zero_is_free() {
        let mut pool = BufferPool::new();
        assert!(pool.acquire(0).is_empty());
    }
}
