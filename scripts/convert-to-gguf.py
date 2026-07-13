#!/usr/bin/env python3
"""Convert a llama2.c checkpoint (+ tokenizer.bin) to GGUF v3.

Stdlib-only. Produces the same tensor names/metadata llama.cpp's converter
emits for a llama model, so the output is loadable by prana (and by any
GGUF llama loader). Tied classifier (output.weight omitted) when the
checkpoint shares weights.

With the optional trailing `q4km` argument, tensors whose rows are a
multiple of 256 are K-quantized (alternating Q4_K / Q6_K per layer, i.e.
a Q4_K_M-style mixture); everything else stays F32. The encoders here are
simple (per-group scale/min, no llama.cpp search) but emit bit-exact valid
blocks for any K-quant decoder.

Usage: convert-to-gguf.py model.bin tokenizer.bin out.gguf [q4km]
"""
import struct
import sys

GGUF_MAGIC = 0x46554747
ALIGN = 32
T_U32, T_F32, T_STR, T_ARR, T_I32 = 4, 6, 8, 9, 5
GGML_F32 = 0
GGML_Q4_K = 12
GGML_Q6_K = 14
QK_K = 256


def q4k_encode_block(vals):
    """256 floats -> one 144-byte Q4_K super-block (8 groups of 32)."""
    scs, ms = [], []
    for g in range(8):
        grp = vals[g * 32:(g + 1) * 32]
        vmin, vmax = min(grp), max(grp)
        m = max(0.0, -vmin)
        sc = max(0.0, (vmax + m) / 15)
        scs.append(sc)
        ms.append(m)
    d = max(scs) / 63 or 1e-12
    dmin = max(ms) / 63 or 1e-12
    ls = [min(63, max(0, round(s / d))) for s in scs]
    lm = [min(63, max(0, round(m / dmin))) for m in ms]

    scales = bytearray(12)
    for j in range(4):
        scales[j] = (ls[j] & 63) | ((ls[j + 4] >> 4) << 6)
        scales[j + 4] = (lm[j] & 63) | ((lm[j + 4] >> 4) << 6)
        scales[j + 8] = (ls[j + 4] & 0xF) | ((lm[j + 4] & 0xF) << 4)

    qs = bytearray(128)
    for pair in range(4):
        for kind in (0, 1):  # low nibbles = group 2p, high = group 2p+1
            g = pair * 2 + kind
            d1 = d * ls[g]
            m1 = dmin * lm[g]
            for l in range(32):
                x = vals[g * 32 + l]
                q = min(15, max(0, round((x + m1) / d1))) if d1 > 0 else 0
                qs[pair * 32 + l] |= q << (4 * kind)
    return struct.pack("<ee", d, dmin) + bytes(scales) + bytes(qs)


def q6k_encode_block(vals):
    """256 floats -> one 210-byte Q6_K super-block (16 sub-blocks of 16)."""
    # Sub-block s covers y[half*128 + g*32 + (s%2)*16 ..+16], half=s//8, g=(s%8)//2
    def sub_slice(s):
        half, w = s // 8, s % 8
        base = half * 128 + (w // 2) * 32 + (w % 2) * 16
        return vals[base:base + 16]

    s16 = [max(abs(v) for v in sub_slice(s)) / 31 for s in range(16)]
    d = max(s16) / 127 or 1e-12
    ls = [min(127, max(1, round(s / d))) for s in s16]

    def q6(v, s):
        eff = d * ls[s]
        return min(63, max(0, round(v / eff) + 32)) if eff > 0 else 32

    ql = bytearray(128)
    qh = bytearray(64)
    for half in range(2):
        for l in range(32):
            sub = half * 8 + l // 16
            qs = [q6(vals[half * 128 + g * 32 + l], sub + g * 2) for g in range(4)]
            ql[half * 64 + l] = (qs[0] & 0xF) | ((qs[2] & 0xF) << 4)
            ql[half * 64 + l + 32] = (qs[1] & 0xF) | ((qs[3] & 0xF) << 4)
            qh[half * 32 + l] = (qs[0] >> 4) | ((qs[1] >> 4) << 2) | ((qs[2] >> 4) << 4) | ((qs[3] >> 4) << 6)
    scales = struct.pack("<16b", *ls)
    return bytes(ql) + bytes(qh) + scales + struct.pack("<e", d)


def kquant_tensor(f32_bytes, cols, ggml_type):
    """Quantize a row-major f32 tensor whose rows are QK_K-aligned."""
    vals = struct.unpack(f"<{len(f32_bytes)//4}f", f32_bytes)
    enc = q4k_encode_block if ggml_type == GGML_Q4_K else q6k_encode_block
    out = bytearray()
    for start in range(0, len(vals), QK_K):
        out += enc(list(vals[start:start + QK_K]))
    return bytes(out)


def w_str(out, s: bytes):
    out += struct.pack("<Q", len(s)) + s


def kv(out, key, ty, packer):
    w_str(out, key.encode())
    out += struct.pack("<I", ty)
    packer(out)


def main(model_path, tok_path, out_path, quant=None):
    raw = open(model_path, "rb").read()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq = struct.unpack("<7i", raw[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    head_dim = dim // n_heads
    kv_dim = n_kv * head_dim
    off = [28]

    def f32s(n):
        s = off[0]
        off[0] += 4 * n
        return raw[s : s + 4 * n]

    tok_emb = f32s(vocab * dim)
    rms_att = [f32s(dim) for _ in range(n_layers)]
    wq = [f32s(dim * dim) for _ in range(n_layers)]
    wk = [f32s(dim * kv_dim) for _ in range(n_layers)]
    wv = [f32s(dim * kv_dim) for _ in range(n_layers)]
    wo = [f32s(dim * dim) for _ in range(n_layers)]
    rms_ffn = [f32s(dim) for _ in range(n_layers)]
    w1 = [f32s(hidden * dim) for _ in range(n_layers)]
    w2 = [f32s(dim * hidden) for _ in range(n_layers)]
    w3 = [f32s(hidden * dim) for _ in range(n_layers)]
    rms_final = f32s(dim)
    f32s(seq * head_dim)  # rope tables, skipped
    wcls = None if shared else f32s(vocab * dim)

    # tokenizer.bin: u32 max_len, then vocab x (f32 score, u32 len, bytes)
    tb = open(tok_path, "rb").read()
    pos = 4
    pieces, scores = [], []
    for _ in range(vocab):
        (score,) = struct.unpack_from("<f", tb, pos)
        (ln,) = struct.unpack_from("<I", tb, pos + 4)
        pieces.append(tb[pos + 8 : pos + 8 + ln])
        scores.append(score)
        pos += 8 + ln
    # token types: 2=unknown, 3=control, 6=byte, 1=normal
    ttypes = [2, 3, 3] + [6] * 256 + [1] * (vocab - 259)

    # tensors: (name, [ne0=cols, ne1=rows], ggml_type, bytes) — all F32 first.
    tensors = [("token_embd.weight", [dim, vocab], GGML_F32, tok_emb)]
    for i in range(n_layers):
        tensors += [
            (f"blk.{i}.attn_norm.weight", [dim], GGML_F32, rms_att[i]),
            (f"blk.{i}.attn_q.weight", [dim, dim], GGML_F32, wq[i]),
            (f"blk.{i}.attn_k.weight", [dim, kv_dim], GGML_F32, wk[i]),
            (f"blk.{i}.attn_v.weight", [dim, kv_dim], GGML_F32, wv[i]),
            (f"blk.{i}.attn_output.weight", [dim, dim], GGML_F32, wo[i]),
            (f"blk.{i}.ffn_norm.weight", [dim], GGML_F32, rms_ffn[i]),
            (f"blk.{i}.ffn_gate.weight", [dim, hidden], GGML_F32, w1[i]),
            (f"blk.{i}.ffn_down.weight", [hidden, dim], GGML_F32, w2[i]),
            (f"blk.{i}.ffn_up.weight", [dim, hidden], GGML_F32, w3[i]),
        ]
    tensors.append(("output_norm.weight", [dim], GGML_F32, rms_final))
    if wcls is not None:
        tensors.append(("output.weight", [dim, vocab], GGML_F32, wcls))

    if quant == "q4km":
        # Q4_K_M-style mixture: K-quantize every 2-D tensor whose rows are
        # QK_K-aligned (llama.cpp applies the same eligibility rule),
        # alternating Q4_K / Q6_K so both decoders get exercised.
        quantized = []
        n_q = 0
        for name, ne, ty, data in tensors:
            if len(ne) == 2 and ne[0] % QK_K == 0:
                ty = GGML_Q4_K if n_q % 2 == 0 else GGML_Q6_K
                data = kquant_tensor(data, ne[0], ty)
                n_q += 1
            quantized.append((name, ne, ty, data))
        tensors = quantized
        print(f"K-quantized {n_q} tensors (rows % {QK_K} == 0)")

    out = bytearray()
    out += struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), 14)  # 14 KVs below
    kv(out, "general.architecture", T_STR, lambda o: w_str(o, b"llama"))
    kv(out, "general.name", T_STR, lambda o: w_str(o, model_path.encode()))
    kv(out, "general.alignment", T_U32, lambda o: o.extend(struct.pack("<I", ALIGN)))
    for key, val in [
        ("llama.embedding_length", dim),
        ("llama.block_count", n_layers),
        ("llama.attention.head_count", n_heads),
        ("llama.attention.head_count_kv", n_kv),
        ("llama.feed_forward_length", hidden),
        ("llama.context_length", seq),
    ]:
        kv(out, key, T_U32, lambda o, v=val: o.extend(struct.pack("<I", v)))
    kv(out, "llama.rope.freq_base", T_F32, lambda o: o.extend(struct.pack("<f", 10000.0)))
    kv(out, "tokenizer.ggml.model", T_STR, lambda o: w_str(o, b"llama"))

    def arr(o, elem_ty, items, pack_item):
        o += struct.pack("<IQ", elem_ty, len(items))
        for it in items:
            pack_item(o, it)

    kv(out, "tokenizer.ggml.tokens", T_ARR, lambda o: arr(o, T_STR, pieces, w_str))
    kv(out, "tokenizer.ggml.scores", T_ARR,
       lambda o: arr(o, T_F32, scores, lambda o2, s: o2.extend(struct.pack("<f", s))))
    kv(out, "tokenizer.ggml.token_type", T_ARR,
       lambda o: arr(o, T_I32, ttypes, lambda o2, t: o2.extend(struct.pack("<i", t))))

    # tensor directory, assigning aligned offsets in the data section
    data_off = 0
    for name, ne, ty, data in tensors:
        w_str(out, name.encode())
        out += struct.pack("<I", len(ne))
        for d in ne:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", ty, data_off)
        data_off = (data_off + len(data) + ALIGN - 1) // ALIGN * ALIGN

    while len(out) % ALIGN:
        out.append(0)
    for name, ne, ty, data in tensors:
        while len(out) % ALIGN:
            out.append(0)
        out += data

    open(out_path, "wb").write(out)
    print(f"wrote {out_path}: {len(out)} bytes, {len(tensors)} tensors, "
          f"dim={dim} layers={n_layers} vocab={vocab} shared_cls={shared}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    main(*sys.argv[1:5])
