# sandbox-playground — local sandbox + LLM inference observability

Local reimplementation of the microVM Sandbox using Docker, combined with
Brendan Gregg's USE method (/proc-based) observability for LLM inference memory pressure.

## What it does

1. Creates an isolated Docker container (equivalent to a microVM sandbox)
2. Installs Python + numpy inside it
3. Runs a simulated LLM inference workload (transformer forward passes with growing KV-cache)
4. Polls `/proc` inside the sandbox every N seconds using the USE method:
   - Memory utilization (`/proc/meminfo`)
   - Page fault saturation (`/proc/vmstat`: pgmajfault, pswpin)
   - Per-process WSS via `smaps_rollup` (Brendan Gregg's recommended approach)
   - Blocking diagnosis via `/proc/<pid>/wchan`
   - Optional: `perf stat` cache miss counters (requires `--privileged`)

## Why microVMs are better for this than containers

In a Docker container, `perf` and eBPF tools need `--privileged` (breaks security model).
In a Firecracker/Cloud Hypervisor microVM, the guest kernel
owns its own virtualised PMU — so `perf stat`, `bpftrace`, and flame graphs work
inside the sandbox without any host privileges. This is a feature of the microVM
architecture, not just a security benefit.

Here is why:
```
% go run ./cmd/main.go
╔══════════════════════════════════════════════════════╗
║  Sandbox — LLM Inference Observability Demo          ║
╚══════════════════════════════════════════════════════╝

config: scale=0.010 tokens=100 mem=2048MB privileged=false

► creating sandbox...
  sandbox id: c8e1abc28022

install numpy

══ BASELINE (before inference) ══
━━━ USE METHOD REPORT ━━━
(Utilization · Saturation · Errors — Brendan Gregg)

MEMORY utilization:  735 MB used / 7936 MB total (9.3%)
  working set (WSS):  946 MB  (Active + AnonPages)
  available:          7201 MB
  mmap'd tensors:     164 MB  (Mapped)

MEMORY saturation:
  minor faults:  897659  (cheap, expected)
  major faults:  1788  (disk fetch — bad if non-zero during inference)
  swap in:       0  (*** inference latency cliff if > 0 ***)
  swap out:      0
  swap used:     0 MB / 1023 MB total

ANOMALY FLAGS:
  [WARN] 1788 major page faults — model not fully warm or WSS > RAM


━━━ END USE REPORT ━━━

► starting inference simulation (scale=0.010, 100 tokens)...
  inference pid: 48


══ USE REPORT #1 (t+10s) ══
━━━ USE METHOD REPORT ━━━
(Utilization · Saturation · Errors — Brendan Gregg)

MEMORY utilization:  684 MB used / 7936 MB total (8.6%)
  working set (WSS):  954 MB  (Active + AnonPages)
  available:          7252 MB
  mmap'd tensors:     164 MB  (Mapped)

MEMORY saturation:
  minor faults:  920184  (cheap, expected)
  major faults:  1789  (disk fetch — bad if non-zero during inference)
  swap in:       0  (*** inference latency cliff if > 0 ***)
  swap out:      0
  swap used:     0 MB / 1023 MB total

ANOMALY FLAGS:
  [WARN] 1789 major page faults — model not fully warm or WSS > RAM


━━━ END USE REPORT ━━━

process [pid=48 python3]:
  rss=0MB  peak=0MB  swap=0MB
  rss_anon=0MB (tensors)  rss_file=0MB (mmap'd weights)
  smaps_rollup: rss=0MB  referenced(WSS)=0MB  swap=0MB
  wchan: 0  → running on CPU (good)

inference log (last 5 lines):
  token=96 kv_cache_mb=0.5 norm=nan
  token=97 kv_cache_mb=0.5 norm=nan
  token=98 kv_cache_mb=0.5 norm=nan
  token=99 kv_cache_mb=0.5 norm=nan
  phase=done

══ USE REPORT #2 (t+20s) ══
━━━ USE METHOD REPORT ━━━
(Utilization · Saturation · Errors — Brendan Gregg)

MEMORY utilization:  686 MB used / 7936 MB total (8.6%)
  working set (WSS):  948 MB  (Active + AnonPages)
  available:          7250 MB
  mmap'd tensors:     164 MB  (Mapped)

MEMORY saturation:
  minor faults:  937335  (cheap, expected)
  major faults:  1789  (disk fetch — bad if non-zero during inference)
  swap in:       0  (*** inference latency cliff if > 0 ***)
  swap out:      0
  swap used:     0 MB / 1023 MB total

ANOMALY FLAGS:
  [WARN] 1789 major page faults — model not fully warm or WSS > RAM
```
Container: one shared kernel → /proc/meminfo/vmstat describe the host VM, not the sandbox. To get truthful per-sandbox numbers you must read cgroup files instead; to get perf you need --privileged (punches a hole in isolation).
microVM: the sandbox is a VM with its own guest kernel. Its /proc/meminfo genuinely is the sandbox's memory, /proc/vmstat faults are genuinely its faults, and perf reads the guest's own PMU — unprivileged, no leakage. The USE method "just works" and the numbers are real.



## Project structure

```
sandbox-playground/
  go.mod
  cmd/
    main.go          ← entry point, wires everything together
  sandbox/
    sandbox.go       ← Create, Exec, WriteFile, ReadFile, Destroy
    observe.go       ← USE method: ReadMemInfo, ReadPageFaults, SmapsRollup, WChan, PerfStat
    scripts.go       ← embedded Python inference sim + shell observe script
```

## Prerequisites

- Docker running locally
- Go 1.22+
- (optional) `--privileged` flag for perf stat

## Run

```bash
cd /Users/pingzhouliu/Documents/koyeb
go mod tidy
go run ./cmd/main.go

# With larger model footprint (500MB) and more tokens:
go run ./cmd/main.go -scale 0.05 -tokens 500 -mem 4096

# With perf stat (Docker Desktop on Mac doesn't support this — use Linux):
go run ./cmd/main.go -privileged
```

## Key /proc files explained

| File | What it tells you |
|---|---|
| `/proc/meminfo` | System-wide memory — MemAvailable is the key number |
| `/proc/vmstat` | pgmajfault (disk fetches), pswpin (swap-ins) — both are inference killers |
| `/proc/<pid>/status` | VmRSS (physical RAM used), VmSwap (in swap) |
| `/proc/<pid>/smaps_rollup` | Referenced = true working set (Brendan Gregg WSS) |
| `/proc/<pid>/wchan` | What kernel function is blocking the inference process |

## USE method for LLM inference

Brendan Gregg's USE method (Utilization, Saturation, Errors) applied to inference:

**Memory utilization**: `(MemTotal - MemAvailable) / MemTotal`
**Memory saturation**: swap-ins (`pswpin`) > 0, or major faults (`pgmajfault`) > 0 during steady-state
**Memory errors**: OOM kills (`dmesg | grep -i oom`)

**The critical signal**: `wchan == do_swap_page` during inference means a transformer
layer's weights had to be fetched from swap — token latency goes from milliseconds to seconds.

## References

- Brendan Gregg, USE Method: https://www.brendangregg.com/usemethod.html
- Brendan Gregg, WSS Estimation: https://www.brendangregg.com/wss.html
- Brendan Gregg, AI Flame Graphs: https://www.brendangregg.com/blog/2024-10-29/ai-flame-graphs.html

