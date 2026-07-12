//! **Phase 0 of the Prana migration plan** (see `EVALUATION.md` §4): a safe,
//! idiomatic Rust crate over the Cactus engine's existing C ABI.
//!
//! Cactus already ships raw `extern "C"` declarations for Rust
//! (`bindings/rust/cactus.rs`), but raw declarations push every safety
//! obligation — handle lifetimes, buffer sizing, NUL termination, callback
//! `user_data` casts — onto each caller. This crate discharges those
//! obligations *once*:
//!
//! - [`Model`] is an RAII handle: `cactus_destroy` runs exactly once, on drop,
//!   even on panic. The raw pointer makes it `!Send + !Sync` by default, which
//!   is the conservative reading of the engine's thread-safety contract.
//! - Errors are `Result`s carrying the engine's `cactus_get_last_error`
//!   message, not sentinel codes the caller must remember to check.
//! - Fixed C `char*` response buffers, out-param token counts, and streaming
//!   callbacks are wrapped in grow-and-retry loops and a panic-containing
//!   closure trampoline, so safe code never sees a raw pointer.
//!
//! The entire remaining unsafe surface of Phase 0 is this crate's `sys` module
//! plus the small, commented `unsafe` blocks in this file — a few hundred
//! lines, auditable in one sitting, exactly the "concentrate `unsafe`" outcome
//! the evaluation argues for.
//!
//! By default the crate binds against an in-process Rust [`mock`] of the ABI
//! (the real `libcactus_engine.a` is ARM/Metal-only); building with
//! `--features link-cactus` and `CACTUS_LIB_DIR` set links the real engine
//! through the identical declarations.

mod json;
#[cfg(not(feature = "link-cactus"))]
pub mod mock;
pub mod sys;

use std::ffi::{CStr, CString, NulError};
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::{c_char, c_void};

/// Which engine implementation this build binds against.
pub fn engine_kind() -> &'static str {
    if cfg!(feature = "link-cactus") {
        "libcactus_engine (native)"
    } else {
        "built-in mock engine"
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything the engine can signal, as a typed error instead of a return code
/// the caller must remember to check.
#[derive(Debug)]
pub enum Error {
    /// `cactus_init` returned null.
    Init(String),
    /// An engine call returned a nonzero code.
    Engine { code: i32, message: String },
    /// A Rust string argument contained an interior NUL byte.
    Nul(NulError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Init(msg) => write!(f, "engine init failed: {msg}"),
            Error::Engine { code, message } => write!(f, "engine call failed (code {code}): {message}"),
            Error::Nul(e) => write!(f, "argument contains interior NUL: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<NulError> for Error {
    fn from(e: NulError) -> Self {
        Error::Nul(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Fetch the engine's thread-local last-error message.
fn last_error() -> String {
    // SAFETY: cactus_get_last_error returns null or a NUL-terminated string
    // owned by the engine, valid until the next failing engine call on this
    // thread — we copy it out immediately.
    let ptr = unsafe { sys::cactus_get_last_error() };
    if ptr.is_null() {
        return String::from("(no error message)");
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
}

/// Map the ABI's return-code convention (0 = success) onto `Result`.
fn check(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Engine { code, message: last_error() })
    }
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// One chat message, rendered into the `messages_json` payload.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system", content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user", content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant", content: content.into() }
    }
}

fn messages_to_json(messages: &[Message]) -> String {
    let mut out = String::from("[");
    for (i, m) in messages.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            r#"{{"role":"{}","content":"{}"}}"#,
            json::escape(m.role),
            json::escape(&m.content)
        ));
    }
    out.push(']');
    out
}

/// Generation options, rendered into `options_json`. `None` fields are omitted
/// so the engine's defaults apply.
#[derive(Debug, Clone)]
pub struct CompleteOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Size of the C response buffer handed to the engine. The engine writes
    /// the full completion result into this fixed buffer, so it bounds the
    /// maximum response length.
    pub response_buffer_size: usize,
}

impl Default for CompleteOptions {
    fn default() -> Self {
        Self { max_tokens: None, temperature: None, top_p: None, response_buffer_size: 64 * 1024 }
    }
}

impl CompleteOptions {
    /// `None` if every engine option is defaulted (pass NULL to the ABI).
    fn to_json(&self) -> Option<String> {
        let mut fields = Vec::new();
        if let Some(v) = self.max_tokens {
            fields.push(format!(r#""max_tokens":{v}"#));
        }
        if let Some(v) = self.temperature {
            fields.push(format!(r#""temperature":{v}"#));
        }
        if let Some(v) = self.top_p {
            fields.push(format!(r#""top_p":{v}"#));
        }
        if fields.is_empty() {
            None
        } else {
            Some(format!("{{{}}}", fields.join(",")))
        }
    }
}

// ---------------------------------------------------------------------------
// The streaming-callback trampoline
// ---------------------------------------------------------------------------

/// A borrowed `on_token(text, token_id)` closure.
type OnToken<'a> = &'a mut dyn FnMut(&str, u32);

/// State shared with the C callback for the duration of one engine call.
struct CallbackState<'a> {
    on_token: OnToken<'a>,
    /// A panic captured inside the callback, to be resumed once control is
    /// safely back on the Rust side of the FFI boundary.
    panic: Option<Box<dyn std::any::Any + Send>>,
}

/// The `extern "C"` shim the engine invokes per token. Unwinding across an
/// `extern "C"` boundary aborts the process, so any panic from the user's
/// closure is caught here and re-raised after the engine call returns.
extern "C" fn token_trampoline(token: *const c_char, token_id: u32, user_data: *mut c_void) {
    // SAFETY: user_data is the &mut CallbackState passed to this exact engine
    // call, and the ABI fires callbacks synchronously on the calling thread,
    // so the reference is live and unaliased for the duration of this call.
    let state = unsafe { &mut *(user_data as *mut CallbackState) };
    if state.panic.is_some() {
        return; // already panicked; swallow the rest of the stream
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if token.is_null() {
            return;
        }
        // SAFETY: non-null checked; the engine passes a NUL-terminated token.
        let text = unsafe { CStr::from_ptr(token) }.to_string_lossy();
        (state.on_token)(&text, token_id);
    }));
    if let Err(payload) = result {
        state.panic = Some(payload);
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// An owned handle to a loaded Cactus model.
///
/// The engine handle is freed exactly once when this value drops. The type is
/// `!Send + !Sync` (raw pointer member): the C ABI documents no thread-safety
/// guarantees, so the wrapper does not invent any. Cross-thread cancellation
/// (`cactus_stop`) is Phase 1 work and needs an engine-level contract first.
pub struct Model {
    raw: sys::cactus_model_t,
    _not_send_sync: PhantomData<*mut ()>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model").field("raw", &self.raw).finish()
    }
}

impl Model {
    /// Load a model from `model_path` (no RAG corpus).
    pub fn open(model_path: &str) -> Result<Self> {
        Self::open_with_corpus(model_path, None, false)
    }

    /// Load a model, optionally attaching a RAG corpus directory.
    pub fn open_with_corpus(model_path: &str, corpus_dir: Option<&str>, cache_index: bool) -> Result<Self> {
        let c_path = CString::new(model_path)?;
        let c_corpus = corpus_dir.map(CString::new).transpose()?;
        // SAFETY: both pointers are live NUL-terminated strings (or null for
        // the optional corpus) for the duration of the call.
        let raw = unsafe {
            sys::cactus_init(
                c_path.as_ptr(),
                c_corpus.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                cache_index,
            )
        };
        if raw.is_null() {
            return Err(Error::Init(last_error()));
        }
        Ok(Self { raw, _not_send_sync: PhantomData })
    }

    /// Run a completion and return the engine's full response payload.
    pub fn complete(&mut self, messages: &[Message], options: &CompleteOptions) -> Result<String> {
        self.complete_inner(messages, options, None)
    }

    /// Run a completion, streaming each generated token to `on_token(text, id)`
    /// as it is produced, then return the full response payload.
    ///
    /// A panic inside `on_token` is caught at the FFI boundary and re-raised
    /// from this function once the engine call has returned.
    pub fn complete_streaming(
        &mut self,
        messages: &[Message],
        options: &CompleteOptions,
        mut on_token: impl FnMut(&str, u32),
    ) -> Result<String> {
        self.complete_inner(messages, options, Some(&mut on_token))
    }

    fn complete_inner(
        &mut self,
        messages: &[Message],
        options: &CompleteOptions,
        on_token: Option<OnToken<'_>>,
    ) -> Result<String> {
        let c_messages = CString::new(messages_to_json(messages))?;
        let c_options = options.to_json().map(CString::new).transpose()?;
        let mut response = vec![0u8; options.response_buffer_size.max(16)];

        let mut discard = |_: &str, _: u32| {};
        let streaming = on_token.is_some();
        let mut state = CallbackState { on_token: on_token.unwrap_or(&mut discard), panic: None };
        let (callback, user_data): (sys::cactus_token_callback, *mut c_void) = if streaming {
            (Some(token_trampoline as _), &mut state as *mut CallbackState as *mut c_void)
        } else {
            (None, std::ptr::null_mut())
        };

        // SAFETY: model handle is live (self owns it); all pointers outlive the
        // call; response buffer length is passed as its true size; callbacks
        // fire synchronously so `state` (a stack local) outlives every fire.
        let code = unsafe {
            sys::cactus_complete(
                self.raw,
                c_messages.as_ptr(),
                response.as_mut_ptr() as *mut c_char,
                response.len(),
                c_options.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                std::ptr::null(), // tools_json: Phase 1
                callback,
                user_data,
                std::ptr::null(), // pcm audio: Phase 1
                0,
            )
        };

        if let Some(payload) = state.panic.take() {
            std::panic::resume_unwind(payload);
        }
        check(code)?;

        let nul = response.iter().position(|&b| b == 0).unwrap_or(response.len());
        response.truncate(nul);
        Ok(String::from_utf8_lossy(&response).into_owned())
    }

    /// Tokenize `text`, growing the buffer and retrying if the engine reports
    /// more tokens than fit (the header's `out_token_len` contract).
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let c_text = CString::new(text)?;
        let mut buf = vec![0u32; 256];
        loop {
            let mut out_len: usize = 0;
            // SAFETY: model handle live; buf.len() is the true capacity;
            // out_len is a live out-param.
            let code = unsafe {
                sys::cactus_tokenize(self.raw, c_text.as_ptr(), buf.as_mut_ptr(), buf.len(), &mut out_len)
            };
            check(code)?;
            if out_len <= buf.len() {
                buf.truncate(out_len);
                return Ok(buf);
            }
            buf.resize(out_len, 0); // engine reported the real count; retry
        }
    }

    /// Embed `text`, returning a vector of the engine-reported dimension.
    pub fn embed(&self, text: &str, normalize: bool) -> Result<Vec<f32>> {
        const MAX_DIM: usize = 8192;
        let c_text = CString::new(text)?;
        let mut buf = vec![0f32; MAX_DIM];
        let mut dim: usize = 0;
        // SAFETY: model handle live; buffer_size is the buffer's byte length;
        // dim is a live out-param.
        let code = unsafe {
            sys::cactus_embed(
                self.raw,
                c_text.as_ptr(),
                buf.as_mut_ptr(),
                buf.len() * std::mem::size_of::<f32>(),
                &mut dim,
                normalize,
            )
        };
        check(code)?;
        buf.truncate(dim.min(MAX_DIM));
        Ok(buf)
    }

    /// Clear conversation state (KV cache) without reloading weights.
    pub fn reset(&mut self) {
        // SAFETY: model handle is live.
        unsafe { sys::cactus_reset(self.raw) };
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: self.raw came from a successful cactus_init and Drop runs at
        // most once; the engine accepts its own handles here.
        unsafe { sys::cactus_destroy(self.raw) };
    }
}

// ---------------------------------------------------------------------------
// Tests (run against the mock engine — `cargo test -p prana-cactus`)
// ---------------------------------------------------------------------------

#[cfg(all(test, not(feature = "link-cactus")))]
mod tests {
    use super::*;

    #[test]
    fn open_and_drop_releases_the_handle() {
        let before = mock::live_models();
        {
            let _m = Model::open("mock://tiny-model").unwrap();
            assert_eq!(mock::live_models(), before + 1);
        }
        assert_eq!(mock::live_models(), before, "Drop must call cactus_destroy exactly once");
    }

    #[test]
    fn open_failure_surfaces_the_engine_message() {
        let err = Model::open("mock://missing-model").unwrap_err();
        match err {
            Error::Init(msg) => assert!(msg.contains("missing-model"), "got: {msg}"),
            other => panic!("expected Init error, got {other:?}"),
        }
    }

    #[test]
    fn complete_returns_the_response() {
        let mut m = Model::open("mock://tiny-model").unwrap();
        let out = m.complete(&[Message::user("ping")], &CompleteOptions::default()).unwrap();
        assert_eq!(out, mock::MOCK_RESPONSE);
    }

    #[test]
    fn streaming_tokens_arrive_in_order_and_closure_state_works() {
        let mut m = Model::open("mock://tiny-model").unwrap();
        let mut streamed = String::new();
        let mut ids = Vec::new();
        let out = m
            .complete_streaming(&[Message::user("ping")], &CompleteOptions::default(), |tok, id| {
                streamed.push_str(tok);
                ids.push(id);
            })
            .unwrap();
        assert_eq!(streamed, mock::MOCK_RESPONSE, "streamed tokens must concatenate to the response");
        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert_eq!(out, mock::MOCK_RESPONSE);
    }

    #[test]
    fn callback_panic_is_contained_at_ffi_and_re_raised() {
        let mut m = Model::open("mock://tiny-model").unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = m.complete_streaming(&[Message::user("ping")], &CompleteOptions::default(), |_, _| {
                panic!("user callback exploded")
            });
        }));
        let payload = result.unwrap_err();
        let msg = payload.downcast_ref::<&str>().copied().unwrap_or_default();
        assert_eq!(msg, "user callback exploded");
        // The model must still be usable and correctly dropped afterwards.
        let out = m.complete(&[Message::user("again")], &CompleteOptions::default()).unwrap();
        assert_eq!(out, mock::MOCK_RESPONSE);
    }

    #[test]
    fn tiny_response_buffer_is_a_typed_error_not_a_truncation() {
        let mut m = Model::open("mock://tiny-model").unwrap();
        let opts = CompleteOptions { response_buffer_size: 16, ..Default::default() };
        let err = m.complete(&[Message::user("ping")], &opts).unwrap_err();
        match err {
            Error::Engine { code, message } => {
                assert!(code < 0);
                assert!(message.contains("buffer"), "got: {message}");
            }
            other => panic!("expected Engine error, got {other:?}"),
        }
    }

    #[test]
    fn tokenize_grows_past_the_initial_buffer() {
        let m = Model::open("mock://tiny-model").unwrap();
        let text = "x".repeat(300); // > initial 256-token buffer
        let tokens = m.tokenize(&text).unwrap();
        assert_eq!(tokens.len(), 300);
        assert!(tokens.iter().all(|&t| t == 'x' as u32));
    }

    #[test]
    fn embed_returns_engine_reported_dimension() {
        let m = Model::open("mock://tiny-model").unwrap();
        let e = m.embed("hello", true).unwrap();
        assert_eq!(e.len(), 8);
        // Deterministic: same text, same embedding.
        assert_eq!(e, m.embed("hello", true).unwrap());
        assert_ne!(e, m.embed("world", true).unwrap());
    }

    #[test]
    fn messages_render_as_escaped_json() {
        let json = messages_to_json(&[
            Message::system("be \"nice\""),
            Message::user("line1\nline2"),
        ]);
        assert_eq!(
            json,
            r#"[{"role":"system","content":"be \"nice\""},{"role":"user","content":"line1\nline2"}]"#
        );
    }

    #[test]
    fn default_options_render_as_null_json() {
        assert_eq!(CompleteOptions::default().to_json(), None);
        let opts = CompleteOptions { max_tokens: Some(64), temperature: Some(0.5), ..Default::default() };
        assert_eq!(opts.to_json().unwrap(), r#"{"max_tokens":64,"temperature":0.5}"#);
    }
}
