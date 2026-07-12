/// Element / storage precision for a tensor.
///
/// Mirrors Cactus's `Precision` enum (`INT8, FP16, FP32, CQ1..CQ4`). Prana's
/// prototype implements the two dense formats plus a block-quantized `Q8`
/// (int8 weights + per-block fp32 scale) that stands in for the Cactus CQ
/// family. The quantized variants carry a `block` size so a `DType` alone is
/// enough to compute packed byte sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE-754 single precision.
    F32,
    /// 8-bit signed integer (per-tensor or per-row scaled elsewhere).
    I8,
    /// Block-quantized int8: for every `block` weights there is one fp32 scale.
    /// Packed layout per block is `[i8; block]` followed by `[f32; 1]`.
    Q8 { block: usize },
}

impl DType {
    /// Number of bytes needed to store `n_elems` logical elements.
    pub const fn packed_bytes(self, n_elems: usize) -> usize {
        match self {
            DType::F32 => n_elems * 4,
            DType::I8 => n_elems,
            DType::Q8 { block } => {
                let blocks = n_elems.div_ceil(block);
                // block int8 values + one f32 scale per block.
                blocks * (block + 4)
            }
        }
    }

    /// Whether this dtype stores dense, directly-addressable scalars.
    pub const fn is_dense(self) -> bool {
        matches!(self, DType::F32 | DType::I8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_sizes() {
        assert_eq!(DType::F32.packed_bytes(10), 40);
        assert_eq!(DType::I8.packed_bytes(10), 10);
        // 10 elems, block 4 -> 3 blocks -> 3 * (4 + 4) = 24 bytes.
        assert_eq!(DType::Q8 { block: 4 }.packed_bytes(10), 24);
        // exact multiple: 8 elems, block 4 -> 2 blocks -> 16 bytes.
        assert_eq!(DType::Q8 { block: 4 }.packed_bytes(8), 16);
    }
}
