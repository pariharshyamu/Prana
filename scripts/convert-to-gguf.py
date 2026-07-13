#!/usr/bin/env python3
"""Convert a llama2.c checkpoint (+ tokenizer.bin) to GGUF v3.

Stdlib-only. Produces the same tensor names/metadata llama.cpp's converter
emits for a llama model, so the output is loadable by prana (and by any
GGUF llama loader). F32 tensors; tied classifier (output.weight omitted)
when the checkpoint shares weights.

Usage: convert-to-gguf.py model.bin tokenizer.bin out.gguf
"""
import struct
import sys

GGUF_MAGIC = 0x46554747
ALIGN = 32
T_U32, T_F32, T_STR, T_ARR, T_I32 = 4, 6, 8, 9, 5
GGML_F32 = 0


def w_str(out, s: bytes):
    out += struct.pack("<Q", len(s)) + s


def kv(out, key, ty, packer):
    w_str(out, key.encode())
    out += struct.pack("<I", ty)
    packer(out)


def main(model_path, tok_path, out_path):
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

    # tensors: (name, [ne0=cols, ne1=rows], bytes)
    tensors = [("token_embd.weight", [dim, vocab], tok_emb)]
    for i in range(n_layers):
        tensors += [
            (f"blk.{i}.attn_norm.weight", [dim], rms_att[i]),
            (f"blk.{i}.attn_q.weight", [dim, dim], wq[i]),
            (f"blk.{i}.attn_k.weight", [dim, kv_dim], wk[i]),
            (f"blk.{i}.attn_v.weight", [dim, kv_dim], wv[i]),
            (f"blk.{i}.attn_output.weight", [dim, dim], wo[i]),
            (f"blk.{i}.ffn_norm.weight", [dim], rms_ffn[i]),
            (f"blk.{i}.ffn_gate.weight", [dim, hidden], w1[i]),
            (f"blk.{i}.ffn_down.weight", [hidden, dim], w2[i]),
            (f"blk.{i}.ffn_up.weight", [dim, hidden], w3[i]),
        ]
    tensors.append(("output_norm.weight", [dim], rms_final))
    if wcls is not None:
        tensors.append(("output.weight", [dim, vocab], wcls))

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
    offsets = []
    for name, ne, data in tensors:
        w_str(out, name.encode())
        out += struct.pack("<I", len(ne))
        for d in ne:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", GGML_F32, data_off)
        offsets.append(data_off)
        data_off = (data_off + len(data) + ALIGN - 1) // ALIGN * ALIGN

    while len(out) % ALIGN:
        out.append(0)
    for (name, ne, data), o in zip(tensors, offsets):
        while len(out) % ALIGN:
            out.append(0)
        out += data

    open(out_path, "wb").write(out)
    print(f"wrote {out_path}: {len(out)} bytes, {len(tensors)} tensors, "
          f"dim={dim} layers={n_layers} vocab={vocab} shared_cls={shared}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
