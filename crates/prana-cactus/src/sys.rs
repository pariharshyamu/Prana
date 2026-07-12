//! Raw C ABI of `libcactus_engine`, transcribed 1:1 from
//! `cactus-engine/cactus_engine.h` (Cactus v2.x). This is the *entire* unsafe
//! import surface of Phase 0 — everything above it is safe Rust.
//!
//! Return-code contract (observed convention, asserted by the mock and to be
//! validated against the real engine in integration tests): functions returning
//! `c_int` yield `0` on success and a negative code on failure, with a
//! human-readable message retrievable via [`cactus_get_last_error`].
//! Handle-returning functions yield null on failure.
//!
//! With the `link-cactus` feature these resolve against the real static
//! library; without it, `crate::mock` provides in-process Rust definitions of
//! the same symbols so the safe wrapper is testable on any host.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_float, c_int, c_void};

pub type cactus_model_t = *mut c_void;
pub type cactus_index_t = *mut c_void;
pub type cactus_stream_transcribe_t = *mut c_void;

/// Streaming token callback: fired synchronously from inside `cactus_complete`
/// / `cactus_transcribe` on the calling thread, once per generated token.
pub type cactus_token_callback =
    Option<unsafe extern "C" fn(token: *const c_char, token_id: u32, user_data: *mut c_void)>;

pub type cactus_log_callback_t = Option<
    unsafe extern "C" fn(level: c_int, component: *const c_char, message: *const c_char, user_data: *mut c_void),
>;

#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "Accelerate", kind = "framework"))]
#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "Metal", kind = "framework"))]
#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "MetalPerformanceShaders", kind = "framework"))]
#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "Foundation", kind = "framework"))]
#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "Security", kind = "framework"))]
#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "SystemConfiguration", kind = "framework"))]
#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "CFNetwork", kind = "framework"))]
#[cfg_attr(all(feature = "link-cactus", target_os = "macos"), link(name = "curl"))]
#[cfg_attr(feature = "link-cactus", link(name = "cactus_engine", kind = "static"))]
extern "C" {
    // --- lifecycle ---
    pub fn cactus_init(model_path: *const c_char, corpus_dir: *const c_char, cache_index: bool) -> cactus_model_t;
    pub fn cactus_destroy(model: cactus_model_t);
    pub fn cactus_reset(model: cactus_model_t);
    pub fn cactus_stop(model: cactus_model_t);
    pub fn cactus_set_backend(backend: *const c_char) -> c_int;

    // --- generation ---
    pub fn cactus_complete(
        model: cactus_model_t,
        messages_json: *const c_char,
        response_buffer: *mut c_char,
        buffer_size: usize,
        options_json: *const c_char,
        tools_json: *const c_char,
        callback: cactus_token_callback,
        user_data: *mut c_void,
        pcm_buffer: *const u8,
        pcm_buffer_size: usize,
    ) -> c_int;
    pub fn cactus_prefill(
        model: cactus_model_t,
        messages_json: *const c_char,
        response_buffer: *mut c_char,
        buffer_size: usize,
        options_json: *const c_char,
        tools_json: *const c_char,
        pcm_buffer: *const u8,
        pcm_buffer_size: usize,
    ) -> c_int;
    pub fn cactus_tokenize(
        model: cactus_model_t,
        text: *const c_char,
        token_buffer: *mut u32,
        token_buffer_len: usize,
        out_token_len: *mut usize,
    ) -> c_int;
    pub fn cactus_render_prompt(
        model: cactus_model_t,
        messages_json: *const c_char,
        options_json: *const c_char,
        tools_json: *const c_char,
        prompt_buffer: *mut c_char,
        buffer_size: usize,
    ) -> c_int;
    pub fn cactus_score_window(
        model: cactus_model_t,
        tokens: *const u32,
        token_len: usize,
        start: usize,
        end: usize,
        context: usize,
        response_buffer: *mut c_char,
        buffer_size: usize,
    ) -> c_int;
    pub fn cactus_benchmark_tokens(
        model: cactus_model_t,
        prompt_tokens: *const u32,
        prompt_token_len: usize,
        decode_token_len: usize,
        response_buffer: *mut c_char,
        buffer_size: usize,
    ) -> c_int;

    // --- speech ---
    pub fn cactus_transcribe(
        model: cactus_model_t,
        audio_file_path: *const c_char,
        prompt: *const c_char,
        response_buffer: *mut c_char,
        buffer_size: usize,
        options_json: *const c_char,
        callback: cactus_token_callback,
        user_data: *mut c_void,
        pcm_buffer: *const u8,
        pcm_buffer_size: usize,
    ) -> c_int;
    pub fn cactus_stream_transcribe_start(model: cactus_model_t, options_json: *const c_char) -> cactus_stream_transcribe_t;
    pub fn cactus_stream_transcribe_process(
        stream: cactus_stream_transcribe_t,
        pcm_buffer: *const u8,
        pcm_buffer_size: usize,
        response_buffer: *mut c_char,
        buffer_size: usize,
    ) -> c_int;
    pub fn cactus_stream_transcribe_stop(
        stream: cactus_stream_transcribe_t,
        response_buffer: *mut c_char,
        buffer_size: usize,
    ) -> c_int;
    pub fn cactus_preprocess_audio_features(
        audio_file_path: *const c_char,
        model_type: *const c_char,
        mel_bins: usize,
        features_buffer: *mut c_float,
        buffer_size: usize,
        feature_count: *mut usize,
        out_mel_bins: *mut usize,
        out_frames: *mut usize,
    ) -> c_int;

    // --- embeddings / RAG / vector index ---
    pub fn cactus_embed(
        model: cactus_model_t,
        text: *const c_char,
        embeddings_buffer: *mut c_float,
        buffer_size: usize,
        embedding_dim: *mut usize,
        normalize: bool,
    ) -> c_int;
    pub fn cactus_image_embed(
        model: cactus_model_t,
        image_path: *const c_char,
        embeddings_buffer: *mut c_float,
        buffer_size: usize,
        embedding_dim: *mut usize,
    ) -> c_int;
    pub fn cactus_audio_embed(
        model: cactus_model_t,
        audio_path: *const c_char,
        embeddings_buffer: *mut c_float,
        buffer_size: usize,
        embedding_dim: *mut usize,
    ) -> c_int;
    pub fn cactus_rag_query(
        model: cactus_model_t,
        query: *const c_char,
        response_buffer: *mut c_char,
        buffer_size: usize,
        top_k: usize,
    ) -> c_int;
    pub fn cactus_index_init(index_dir: *const c_char, embedding_dim: usize) -> cactus_index_t;
    pub fn cactus_index_add(
        index: cactus_index_t,
        ids: *const c_int,
        documents: *const *const c_char,
        metadatas: *const *const c_char,
        embeddings: *const *const c_float,
        count: usize,
        embedding_dim: usize,
    ) -> c_int;
    pub fn cactus_index_delete(index: cactus_index_t, ids: *const c_int, ids_count: usize) -> c_int;
    pub fn cactus_index_get(
        index: cactus_index_t,
        ids: *const c_int,
        ids_count: usize,
        document_buffers: *mut *mut c_char,
        document_buffer_sizes: *mut usize,
        metadata_buffers: *mut *mut c_char,
        metadata_buffer_sizes: *mut usize,
        embedding_buffers: *mut *mut c_float,
        embedding_buffer_sizes: *mut usize,
    ) -> c_int;
    pub fn cactus_index_query(
        index: cactus_index_t,
        embeddings: *const *const c_float,
        embeddings_count: usize,
        embedding_dim: usize,
        options_json: *const c_char,
        id_buffers: *mut *mut c_int,
        id_buffer_sizes: *mut usize,
        score_buffers: *mut *mut c_float,
        score_buffer_sizes: *mut usize,
    ) -> c_int;
    pub fn cactus_index_compact(index: cactus_index_t) -> c_int;
    pub fn cactus_index_destroy(index: cactus_index_t);

    // --- diagnostics / telemetry ---
    pub fn cactus_get_last_error() -> *const c_char;
    pub fn cactus_log_set_level(level: c_int);
    pub fn cactus_log_set_callback(callback: cactus_log_callback_t, user_data: *mut c_void);
    pub fn cactus_set_telemetry_environment(framework: *const c_char, cache_location: *const c_char, version: *const c_char);
    pub fn cactus_set_app_id(app_id: *const c_char);
    pub fn cactus_telemetry_flush();
    pub fn cactus_telemetry_shutdown();
}
