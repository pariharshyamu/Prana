//! An in-process Rust implementation of the subset of the Cactus C ABI that
//! the safe wrapper calls, active whenever the `link-cactus` feature is off.
//!
//! Why this exists: `libcactus_engine.a` is built with ARM NEON and Metal and
//! cannot be produced on an x86 dev/CI box — but the *interesting* logic of
//! Phase 0 (RAII lifetimes, error propagation, callback trampolines, buffer
//! grow-and-retry) lives in the safe wrapper and deserves real tests. Defining
//! the same `extern "C"` symbols here lets the wrapper run against a faithful
//! stand-in of the ABI contract: null-on-failure handles, `0`/negative return
//! codes, `cactus_get_last_error`, synchronous token callbacks, and out-param
//! buffer sizing.
//!
//! It is deliberately dumb about *content* (canned completion, per-char
//! "tokenizer", hashed "embeddings"): the contract under test is the ABI, not
//! the model.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::sys::cactus_token_callback;

/// Live model count, so tests can prove `Drop` really releases the handle.
static LIVE_MODELS: AtomicUsize = AtomicUsize::new(0);

pub fn live_models() -> usize {
    LIVE_MODELS.load(Ordering::SeqCst)
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(msg).unwrap_or_default());
}

struct MockModel {
    #[allow(dead_code)]
    path: String,
}

/// The canned completion the mock streams and returns.
pub const MOCK_RESPONSE: &str = "pong from prana mock";

/// Read a required C string argument, or record an error and return `None`.
///
/// # Safety
/// `ptr` must be null or a valid NUL-terminated string (the ABI precondition).
unsafe fn read_cstr(ptr: *const c_char, what: &str) -> Option<String> {
    if ptr.is_null() {
        set_last_error(&format!("{what} must not be null"));
        return None;
    }
    // SAFETY: non-null checked above; caller guarantees NUL termination.
    Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
}

/// Write `text` (plus NUL) into a caller-provided `char` buffer, honoring the
/// fixed-buffer contract of the real engine: fail if it does not fit.
///
/// # Safety
/// `buf` must be valid for writes of `buf_size` bytes.
unsafe fn write_response(buf: *mut c_char, buf_size: usize, text: &str) -> c_int {
    if buf.is_null() || buf_size < text.len() + 1 {
        set_last_error("response buffer too small");
        return -2;
    }
    // SAFETY: bounds checked above; caller guarantees the buffer is writable.
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buf as *mut u8, text.len());
        *buf.add(text.len()) = 0;
    }
    0
}

/// # Safety
/// `model_path`/`_corpus_dir` must be null or valid NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn cactus_init(model_path: *const c_char, _corpus_dir: *const c_char, _cache_index: bool) -> *mut c_void {
    // SAFETY: ABI precondition — model_path is null or NUL-terminated.
    let Some(path) = (unsafe { read_cstr(model_path, "model_path") }) else {
        return std::ptr::null_mut();
    };
    if path.is_empty() || path.contains("missing") {
        set_last_error(&format!("failed to load model at '{path}'"));
        return std::ptr::null_mut();
    }
    LIVE_MODELS.fetch_add(1, Ordering::SeqCst);
    Box::into_raw(Box::new(MockModel { path })) as *mut c_void
}

/// # Safety
/// `model` must be null or a handle returned by [`cactus_init`], passed at
/// most once (the handle is freed).
#[no_mangle]
pub unsafe extern "C" fn cactus_destroy(model: *mut c_void) {
    if model.is_null() {
        return;
    }
    LIVE_MODELS.fetch_sub(1, Ordering::SeqCst);
    // SAFETY: the only non-null values handed out are Box::into_raw pointers
    // from cactus_init, and the wrapper's Drop calls this exactly once.
    drop(unsafe { Box::from_raw(model as *mut MockModel) });
}

#[no_mangle]
pub extern "C" fn cactus_reset(_model: *mut c_void) {}

#[no_mangle]
pub extern "C" fn cactus_stop(_model: *mut c_void) {}

/// # Safety
/// Pointer arguments must satisfy the header's contract: strings NUL-terminated
/// or null, `response_buffer` writable for `buffer_size` bytes, and `user_data`
/// whatever the given `callback` expects.
#[no_mangle]
pub unsafe extern "C" fn cactus_complete(
    model: *mut c_void,
    messages_json: *const c_char,
    response_buffer: *mut c_char,
    buffer_size: usize,
    _options_json: *const c_char,
    _tools_json: *const c_char,
    callback: cactus_token_callback,
    user_data: *mut c_void,
    _pcm_buffer: *const u8,
    _pcm_buffer_size: usize,
) -> c_int {
    if model.is_null() {
        set_last_error("model must not be null");
        return -1;
    }
    // SAFETY: ABI precondition on messages_json.
    if unsafe { read_cstr(messages_json, "messages_json") }.is_none() {
        return -1;
    }

    // Stream the canned response as whitespace-split tokens, exactly like the
    // real engine: synchronously, on this thread, one call per token.
    if let Some(cb) = callback {
        for (i, tok) in ["pong", " from", " prana", " mock"].iter().enumerate() {
            let c_tok = CString::new(*tok).unwrap();
            // SAFETY: c_tok outlives the call; cb is the caller's trampoline
            // invoked with the caller's own user_data, per the ABI contract.
            unsafe { cb(c_tok.as_ptr(), i as u32 + 1, user_data) };
        }
    }

    // SAFETY: ABI precondition — response_buffer is writable for buffer_size.
    unsafe { write_response(response_buffer, buffer_size, MOCK_RESPONSE) }
}

/// # Safety
/// `text` must be null or NUL-terminated; `token_buffer` must hold
/// `token_buffer_len` `u32`s; `out_token_len` must be null or writable.
#[no_mangle]
pub unsafe extern "C" fn cactus_tokenize(
    model: *mut c_void,
    text: *const c_char,
    token_buffer: *mut u32,
    token_buffer_len: usize,
    out_token_len: *mut usize,
) -> c_int {
    if model.is_null() || out_token_len.is_null() {
        set_last_error("model/out_token_len must not be null");
        return -1;
    }
    // SAFETY: ABI precondition on text.
    let Some(text) = (unsafe { read_cstr(text, "text") }) else {
        return -1;
    };
    // "Tokenize" one token per char; always report the true count so the
    // caller can grow its buffer and retry — the same contract as the header's
    // out_token_len.
    let tokens: Vec<u32> = text.chars().map(|c| c as u32).collect();
    // SAFETY: out_token_len non-null checked above.
    unsafe { *out_token_len = tokens.len() };
    let n = tokens.len().min(token_buffer_len);
    if n > 0 && !token_buffer.is_null() {
        // SAFETY: caller guarantees token_buffer holds token_buffer_len u32s.
        unsafe { std::ptr::copy_nonoverlapping(tokens.as_ptr(), token_buffer, n) };
    }
    0
}

/// # Safety
/// `text` must be null or NUL-terminated; `embeddings_buffer` must be writable
/// for `buffer_size` bytes; `embedding_dim` must be null or writable.
#[no_mangle]
pub unsafe extern "C" fn cactus_embed(
    model: *mut c_void,
    text: *const c_char,
    embeddings_buffer: *mut c_float,
    buffer_size: usize,
    embedding_dim: *mut usize,
    _normalize: bool,
) -> c_int {
    const DIM: usize = 8;
    if model.is_null() || embedding_dim.is_null() {
        set_last_error("model/embedding_dim must not be null");
        return -1;
    }
    // SAFETY: ABI precondition on text.
    let Some(text) = (unsafe { read_cstr(text, "text") }) else {
        return -1;
    };
    if buffer_size < DIM * std::mem::size_of::<c_float>() || embeddings_buffer.is_null() {
        set_last_error("embeddings buffer too small");
        return -2;
    }
    // Deterministic pseudo-embedding derived from the text bytes.
    let mut h: u32 = 2166136261;
    for b in text.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    for i in 0..DIM {
        let v = ((h.rotate_left(i as u32 * 4) & 0xffff) as f32 / 65535.0) - 0.5;
        // SAFETY: bounds checked against buffer_size above.
        unsafe { *embeddings_buffer.add(i) = v };
    }
    // SAFETY: embedding_dim non-null checked above.
    unsafe { *embedding_dim = DIM };
    0
}

#[no_mangle]
pub extern "C" fn cactus_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}
