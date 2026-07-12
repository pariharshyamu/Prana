use crate::dtype::DType;

/// A shape is just an owned list of dimensions. Kept as a newtype so it can grow
/// helper methods without leaking `Vec` semantics everywhere.
pub type Shape = Vec<usize>;

/// An owned, contiguous tensor.
///
/// Data is stored as raw bytes (`Vec<u8>`) with a `DType` describing how to
/// interpret them, which lets one type carry F32, I8, and block-quantized Q8
/// payloads — the same erasure Cactus does with `BufferDesc` + `void*`, but with
/// the length and ownership tracked by `Vec` instead of by hand.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    dtype: DType,
    data: Vec<u8>,
}

impl Tensor {
    /// Allocate a zeroed tensor of the given shape and dtype.
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        let n: usize = shape.iter().product();
        let data = vec![0u8; dtype.packed_bytes(n)];
        Self { shape, dtype, data }
    }

    /// Build an F32 tensor from logical values. Panics if `values.len()` does
    /// not match the product of `shape` (a programming error, not input error).
    pub fn from_f32(shape: Shape, values: &[f32]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, values.len(), "shape {shape:?} does not match {} values", values.len());
        let mut data = vec![0u8; n * 4];
        for (chunk, &v) in data.chunks_exact_mut(4).zip(values) {
            chunk.copy_from_slice(&v.to_ne_bytes());
        }
        Self { shape, dtype: DType::F32, data }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total number of logical elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Raw packed bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// View the payload as `f32`, or `None` if this tensor is not F32.
    pub fn as_f32(&self) -> Option<Vec<f32>> {
        if self.dtype != DType::F32 {
            return None;
        }
        Some(
            self.data
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_f32() {
        let t = Tensor::from_f32(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.as_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn zeros_has_correct_byte_len() {
        let t = Tensor::zeros(vec![3, 4], DType::Q8 { block: 4 });
        // 12 elems / block 4 = 3 blocks * (4 + 4) = 24 bytes.
        assert_eq!(t.bytes().len(), 24);
    }
}
