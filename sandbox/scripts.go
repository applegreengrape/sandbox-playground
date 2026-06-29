package sandbox

// InferenceSimPy is a Python script that simulates the memory access pattern
// of LLM inference (weight streaming, repeated large matmuls, KV-cache growth).
//
// It deliberately replicates what makes LLM inference memory-bound:
//   - large tensors allocated upfront (model weights)
//   - weights streamed through CPU cache on every forward pass
//   - KV-cache growing with each new token (sequence length)
//
// Run inside a sandbox to generate real /proc metrics for USE analysis.
const InferenceSimPy = `#!/usr/bin/env python3
"""
Simulates LLM inference memory pressure for observability testing.

Memory pattern mirrors a real transformer:
  - Weight tensors: loaded once, streamed through cache each forward pass
  - KV-cache: grows linearly with generated tokens (memory leak pattern)
  - Attention: O(n^2) memory in naive implementation

Scale factor: 0.01 = 1% of a real 7B model (~140MB instead of 14GB).
Set SCALE=1.0 and give the sandbox 32GB RAM to test a real model footprint.
"""
import numpy as np
import time
import os
import sys

SCALE = float(os.environ.get("INFERENCE_SCALE", "0.01"))
N_LAYERS = 32
D_MODEL = int(4096 * SCALE)
D_FF = int(16384 * SCALE)
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "500"))
TOKEN_DELAY = float(os.environ.get("TOKEN_DELAY", "0.05"))  # seconds per token

print(f"pid={os.getpid()}", flush=True)
print(f"scale={SCALE} d_model={D_MODEL} d_ff={D_FF} n_layers={N_LAYERS}", flush=True)

# ── Phase 1: Model loading (major page faults expected here) ──
print("phase=loading", flush=True)
t0 = time.time()

# Weights — alloc triggers minor faults; first access triggers major faults
# if OS uses lazy allocation (which Linux does by default).
wq = [np.random.randn(D_MODEL, D_MODEL).astype(np.float16) for _ in range(N_LAYERS)]
wk = [np.random.randn(D_MODEL, D_MODEL).astype(np.float16) for _ in range(N_LAYERS)]
wv = [np.random.randn(D_MODEL, D_MODEL).astype(np.float16) for _ in range(N_LAYERS)]
wo = [np.random.randn(D_MODEL, D_MODEL).astype(np.float16) for _ in range(N_LAYERS)]
w1 = [np.random.randn(D_MODEL, D_FF).astype(np.float16) for _ in range(N_LAYERS)]
w2 = [np.random.randn(D_FF, D_MODEL).astype(np.float16) for _ in range(N_LAYERS)]

weight_mb = sum(
    w.nbytes for weights in [wq, wk, wv, wo, w1, w2] for w in weights
) / 1e6
print(f"weights_loaded_mb={weight_mb:.1f} t={time.time()-t0:.2f}s", flush=True)

# ── Phase 2: Inference loop (memory bandwidth pressure) ──
print("phase=inference", flush=True)

# KV-cache: grows with each token — key memory leak pattern in LLM serving
# Shape: [n_layers, seq_len, d_model]
kv_cache_k = []
kv_cache_v = []

for token_idx in range(MAX_TOKENS):
    x = np.random.randn(1, D_MODEL).astype(np.float16)

    # Forward pass through all layers — streams weights through cache each time
    for layer in range(N_LAYERS):
        # Self-attention: Q, K, V projections
        q = x @ wq[layer]          # [1, D_MODEL] x [D_MODEL, D_MODEL]
        k = x @ wk[layer]
        v = x @ wv[layer]

        # Append to KV-cache (grows with sequence length)
        kv_cache_k.append(k)
        kv_cache_v.append(v)

        # Attention over full sequence (simplified — no masking)
        if len(kv_cache_k) > 1:
            all_k = np.concatenate(kv_cache_k[-64:], axis=0)  # cap at 64
            all_v = np.concatenate(kv_cache_v[-64:], axis=0)
            scores = q @ all_k.T / np.sqrt(D_MODEL)
            attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
            x = (attn @ all_v) @ wo[layer]
        else:
            x = (q * 0.1) @ wo[layer]

        # FFN
        x = np.maximum(0, x @ w1[layer].astype(np.float32)) @ w2[layer].astype(np.float32)
        x = x.astype(np.float16)

    kv_mb = sum(a.nbytes for a in kv_cache_k + kv_cache_v) / 1e6
    print(
        f"token={token_idx} kv_cache_mb={kv_mb:.1f} "
        f"norm={float(np.linalg.norm(x.astype(np.float32))):.4f}",
        flush=True,
    )
    time.sleep(TOKEN_DELAY)

print("phase=done", flush=True)
`

// ObserveScript is a shell script that runs Brendan Gregg's /proc-based
// WSS estimation for the inference process.
const ObserveScript = `#!/bin/bash
# Brendan Gregg's working set size (WSS) estimation via /proc.
# Ref: https://www.brendangregg.com/wss.html
#
# Usage: ./observe.sh <pid> [interval_seconds]

# Slim images lack procps (pgrep), so fall back to the pidfile written at launch.
PID=${1:-$(cat /tmp/inference.pid 2>/dev/null)}
INTERVAL=${2:-5}

if [ -z "$PID" ]; then
  echo "usage: observe.sh <pid>" >&2
  exit 1
fi

echo "=== WSS estimation for PID $PID ==="
echo ""

echo "-- /proc/$PID/status (process memory) --"
cat /proc/$PID/status 2>/dev/null | grep -E '^(Name|VmRSS|VmPeak|VmSize|VmSwap|RssAnon|RssFile):'

echo ""
echo "-- /proc/$PID/smaps_rollup (Brendan Gregg WSS source) --"
cat /proc/$PID/smaps_rollup 2>/dev/null

echo ""
echo "-- /proc/$PID/wchan (what is the process blocked on?) --"
echo -n "wchan: "
cat /proc/$PID/wchan 2>/dev/null
echo ""
# Interpretation:
#   0                 = running on CPU (good during inference)
#   futex_wait*       = waiting on lock (normal, BLAS thread pool)
#   do_swap_page      = swap fault DURING inference (critical)
#   io_schedule       = blocked on disk (bad)

echo ""
echo "-- /proc/vmstat (system-wide fault counters) --"
grep -E '^(pgfault|pgmajfault|pswpin|pswpout)' /proc/vmstat

echo ""
echo "-- /proc/meminfo (system memory) --"
grep -E '^(MemTotal|MemAvailable|Active|Inactive|AnonPages|Mapped|SwapTotal|SwapFree)' /proc/meminfo
`
