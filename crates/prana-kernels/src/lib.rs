//! Compute kernels for Prana.
//!
//! # Design intent (this is the crux of the feasibility question)
//!
//! Cactus's hot path is ~35k lines of hand-written ARM NEON in
//! `cactus-kernels/`. The open question a Rust rewrite has to answer is:
//! *can we keep that performance without keeping that much unsafe code?*
//!
//! Prana's answer, demonstrated here, is a two-tier strategy:
//!
//! 1. **Default tier — safe, autovectorized.** The scalar kernels below are
//!    written in a shape LLVM reliably turns into SIMD (`chunks_exact`,
//!    accumulator arrays, no aliasing). No `unsafe`, portable to every target,
//!    and already within a small constant factor of hand-tuned code.
//! 2. **Fast tier — small, audited `unsafe`.** The same kernel signatures get
//!    a target-gated intrinsics implementation, selected at runtime. This
//!    ships today for x86-64 (`simd_x86.rs`: AVX2+FMA dot products, ~100 lines
//!    of commented `unsafe`); an aarch64 NEON twin slots in beside it the same
//!    way. That `unsafe` surface is a few hundred lines, not tens of
//!    thousands, because everything above the kernel boundary is safe.
//!
//! Tier 1 keeps the prototype correct and portable everywhere; tier 2 is used
//! automatically wherever the CPU supports it, and every SIMD kernel is tested
//! for parity against its scalar twin.

mod attention;
mod kquant;
mod matmul;
mod norms;
mod pool;
mod simd_x86;
mod team;
mod threading;

pub use attention::{
    attention, attention_decode, attention_decode_team, f16_kv_write, rope, rope_interleaved,
    rope_neox, F16KvView,
};
pub use kquant::{
    dequant_q40_block, dequant_q4k_block, dequant_q6k_block, matmul_kquant_f32,
    matmul_kquant_prefill, matmul_kquant_team, q4k_scale_min, KQuantKind, KQuantMatrix,
    Q4_0_BLOCK, Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K,
};
pub use matmul::{
    dequantize_q8, dot_f32, matmul_f32, matmul_f32_team, matmul_q8_f32, matmul_q8_prefill,
    matmul_q8_team, quantize_acts, quantize_q8, QuantActs, QuantMatrix, Q8_BLOCK,
};
pub use norms::{gelu_tanh, rmsnorm, silu, softmax};
pub use pool::{global as pool, Pool};
pub use simd_x86::{available as simd_available, available_vnni as simd_vnni_available};
pub use team::{
    run_team, team_fill_rows, team_fill_rows_weighted, team_fill_slices, Team, TeamCell, TeamCtx,
};
pub use threading::parallel_for;

/// IEEE 754 half → single conversion (handles subnormals, inf, NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        // subnormal: exact value is frac * 2^-24 (sign applied numerically)
        (0, f) => {
            let v = f as f32 * 2f32.powi(-24);
            return if sign == 1 { -v } else { v };
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        (e, f) => (sign << 31) | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

/// IEEE 754 single → half conversion, round-to-nearest-even. Used to pack
/// the KV cache at half precision (half the attention-read bandwidth).
/// Overflow saturates to ±inf; subnormals and zero are handled exactly.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32; // biased f32 exponent
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        // inf / NaN: keep a nonzero mantissa for NaN so it stays NaN.
        return sign | 0x7c00 | if frac != 0 { 0x0200 } else { 0 };
    }
    // Unbias f32 (127) and rebias to f16 (15).
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        // Subnormal or zero half. Shift the implicit-1 mantissa right.
        if e < -10 {
            return sign; // too small → signed zero
        }
        let m = frac | 0x0080_0000; // restore implicit leading 1
        let shift = (14 - e) as u32; // 14 = 23 - 10 + 1
        let half = (m >> shift) as u16;
        // Round to nearest even on the shifted-out bits.
        let round_bit = 1u32 << (shift - 1);
        let rem = m & (round_bit * 2 - 1);
        let round = rem > round_bit || (rem == round_bit && (half & 1) == 1);
        return sign | (half + round as u16);
    }
    // Normal half. Round the 23-bit mantissa to 10 bits, nearest-even.
    let half = (e as u16) << 10 | (frac >> 13) as u16;
    let rem = frac & 0x1fff; // 13 dropped bits
    let round = rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1);
    // Carry from mantissa rounding propagates into exp naturally via the add.
    sign | (half + round as u16)
}

#[cfg(test)]
mod f16_tests {
    use super::{f16_to_f32, f32_to_f16};

    #[test]
    fn exact_representable_values_round_trip() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, -2.0, 0.25, 65504.0, -65504.0] {
            let back = f16_to_f32(f32_to_f16(v));
            assert_eq!(back.to_bits(), v.to_bits(), "{v} round-trip");
        }
    }

    #[test]
    fn rounds_to_nearest_and_stays_close() {
        // Typical activation magnitudes: error must be <= half an ulp of the
        // f16 grid (~1e-3 relative for values near 1).
        for i in -2000..2000 {
            let v = i as f32 * 0.0017;
            let r = f16_to_f32(f32_to_f16(v));
            let tol = v.abs() * 1e-3 + 1e-6;
            assert!((r - v).abs() <= tol, "{v} -> {r}, err {}", (r - v).abs());
        }
    }

    #[test]
    fn handles_special_and_overflow() {
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f32_to_f16(1e30), 0x7c00, "overflow saturates to +inf");
        assert_eq!(f32_to_f16(1e-30), 0x0000, "underflow to +0");
    }

    #[test]
    fn round_to_nearest_even_on_the_boundary() {
        // A value exactly between two f16s rounds to the even neighbor.
        // 1.0 + 2^-11 sits on the tie between 1.0 (0x3c00) and next up.
        let mid = 1.0f32 + 2f32.powi(-11);
        assert_eq!(f32_to_f16(mid), 0x3c00, "ties to even (1.0)");
    }
}
