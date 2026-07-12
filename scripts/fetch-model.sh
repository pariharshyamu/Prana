#!/usr/bin/env bash
# Fetch the stories15M TinyStories model (Karpathy's llama2.c checkpoint) and
# the Llama tokenizer into models/, for `prana run`.
#
# Primary source is Hugging Face. On networks where huggingface.co is not
# reachable (e.g. egress-allowlisted CI), fall back to carving the identical
# bytes out of the `llama2.c-emscripten` npm package, whose Emscripten preload
# bundle contains model.bin at offset 0..60816028 (see its dist/llama2.js
# loadPackage metadata). Both routes yield the same checkpoint:
# dim=288, 6 layers, 6 heads, vocab 32000, ctx 256, 60,816,028 bytes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

MODEL=models/stories15M.bin
TOK=models/tokenizer.bin

if [ ! -f "$TOK" ]; then
  echo "fetching tokenizer.bin (llama2.c repo)..."
  curl -fSL -o "$TOK" https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
fi

if [ -f "$MODEL" ]; then
  echo "$MODEL already present"
  exit 0
fi

echo "trying Hugging Face..."
if curl -fSL --connect-timeout 15 -o "$MODEL" \
    https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin; then
  echo "fetched from Hugging Face"
else
  echo "Hugging Face unreachable; carving from the llama2.c-emscripten npm package..."
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  curl -fSL -o "$TMP/pkg.tgz" \
    https://registry.npmjs.org/llama2.c-emscripten/-/llama2.c-emscripten-0.1.0.tgz
  tar -xzf "$TMP/pkg.tgz" -C "$TMP" package/dist/llama2.data
  head -c 60816028 "$TMP/package/dist/llama2.data" > "$MODEL"
fi

SIZE=$(wc -c < "$MODEL")
if [ "$SIZE" -ne 60816028 ]; then
  echo "ERROR: unexpected model size $SIZE (want 60816028)" >&2
  exit 1
fi
echo "OK: $MODEL ($SIZE bytes)"
