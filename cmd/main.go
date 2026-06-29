// Command sandbox-playground demonstrates a local sandbox running LLM inference simulation
// with live USE-method observability (memory utilization, saturation, page faults)
// modelled after Brendan Gregg's /proc-based performance methodology.
//
// Usage:
//
//	go run ./cmd/main.go
//	go run ./cmd/main.go -scale 0.05 -tokens 200 -privileged
//
// Flags:
//
//	-scale      fraction of a real 7B model to simulate (default 0.01 = ~140MB)
//	-tokens     number of tokens to generate (default 100)
//	-mem        sandbox RAM cap in MB (default 2048)
//	-privileged enable perf stat (needs Docker --privileged or microVM)
//	-interval   seconds between USE reports (default 10)
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"sandbox-playground/sandbox"
)

func main() {
	scale      := flag.Float64("scale", 0.01, "model scale (0.01 = ~140MB, 1.0 = ~14GB)")
	tokens     := flag.Int("tokens", 100, "tokens to generate")
	memMB      := flag.Int64("mem", 2048, "sandbox RAM cap MB")
	privileged := flag.Bool("privileged", false, "enable perf stat (requires Docker privileged or microVM)")
	interval   := flag.Int("interval", 10, "seconds between USE method reports")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	fmt.Println("╔══════════════════════════════════════════════════════╗")
	fmt.Println("║  Sandbox — LLM Inference Observability Demo          ║")
	fmt.Println("╚══════════════════════════════════════════════════════╝")
	fmt.Printf("\nconfig: scale=%.3f tokens=%d mem=%dMB privileged=%v\n\n",
		*scale, *tokens, *memMB, *privileged)

	// ── 1. Create sandbox ──────────────────────────────────────────────────
	fmt.Println("► creating sandbox...")
	sb, err := sandbox.Create(ctx, sandbox.Config{
		Image:       "python:3.12-slim",
		MemoryMB:    *memMB,
		CPUs:        2.0,
		PidsLimit:   512,
		NetworkMode: "bridge",
		Privileged:  *privileged,
	})
	if err != nil {
		log.Fatalf("create sandbox: %v", err)
	}
	defer func() {
		fmt.Println("\n► destroying sandbox...")
		sb.Destroy(context.Background())
	}()
	fmt.Printf("  sandbox id: %s\n\n", sb.ID[:12])

	// 2. install command
	_, err = sb.ExecShell(ctx,
		"pip install --quiet numpy && echo OK")
	fmt.Println("install numpy\n")

	// ── 3. Write scripts into sandbox ─────────────────────────────────────
	if err := sb.WriteFile(ctx, "/tmp/sim_inference.py", sandbox.InferenceSimPy); err != nil {
		log.Fatalf("write inference script: %v", err)
	}
	if err := sb.WriteFile(ctx, "/tmp/observe.sh", sandbox.ObserveScript); err != nil {
		log.Fatalf("write observe script: %v", err)
	}
	sb.Exec(ctx, "chmod", "+x", "/tmp/observe.sh")

	// ── 4. Baseline USE report (before inference) ─────────────────────────
	fmt.Println("══ BASELINE (before inference) ══")
	if err := sandbox.USEReport(ctx, sb); err != nil {
		log.Printf("baseline USE report: %v", err)
	}

	// ── 5. Start inference in background ──────────────────────────────────
	fmt.Printf("\n► starting inference simulation (scale=%.3f, %d tokens)...\n", *scale, *tokens)
	// Capture the background PID via $! (a bash builtin) into a pidfile.
	// The python:3.12-slim image has no procps, so pgrep/ps are unavailable.
	inferenceCmd := fmt.Sprintf(
		"INFERENCE_SCALE=%.4f MAX_TOKENS=%d TOKEN_DELAY=0.1 python3 /tmp/sim_inference.py > /tmp/inference.log 2>&1 & echo $! > /tmp/inference.pid",
		*scale, *tokens,
	)
	sb.ExecShell(ctx, inferenceCmd)
	time.Sleep(2 * time.Second) // let it start

	// Get the inference PID from the pidfile.
	pidResult, err := sb.ExecShell(ctx, "cat /tmp/inference.pid 2>/dev/null")
	pidStr := strings.TrimSpace(pidResult.Stdout)
	if err != nil || pidStr == "" {
		log.Fatal("inference process didn't start — check /tmp/inference.log")
	}
	pid, _ := strconv.Atoi(pidStr)

	// A pidfile alone doesn't prove the script survived import/startup, so
	// confirm the process is actually alive (kill -0 is a bash builtin).
	alive, _ := sb.ExecShell(ctx, fmt.Sprintf("kill -0 %d 2>/dev/null && echo yes || echo no", pid))
	if strings.TrimSpace(alive.Stdout) != "yes" {
		printInferenceLog(ctx, sb)
		log.Fatal("inference process exited early — see log above")
	}
	fmt.Printf("  inference pid: %d\n\n", pid)

	// ── 6. Live USE reports every interval seconds ─────────────────────────
	ticker := time.NewTicker(time.Duration(*interval) * time.Second)
	defer ticker.Stop()

	reportNum := 1
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\ninterrupted.")
			return

		case <-ticker.C:
			fmt.Printf("\n══ USE REPORT #%d (t+%ds) ══\n", reportNum, reportNum**interval)
			reportNum++

			// Check if inference is still running
			running, _ := sb.ExecShell(ctx, fmt.Sprintf("kill -0 %d 2>/dev/null && echo yes || echo no", pid))
			if strings.TrimSpace(running.Stdout) == "no" {
				fmt.Println("inference completed.")
				printInferenceLog(ctx, sb)
				printFinalUSE(ctx, sb)
				return
			}

			// USE method report
			if err := sandbox.USEReport(ctx, sb); err != nil {
				log.Printf("USE report: %v", err)
				continue
			}

			// Per-process detail
			fmt.Println()
			proc, err := sandbox.ReadProcStatus(ctx, sb, pid)
			if err == nil {
				fmt.Printf("process [pid=%d %s]:\n", proc.PID, proc.Comm)
				fmt.Printf("  rss=%dMB  peak=%dMB  swap=%dMB\n",
					proc.VmRSSMB, proc.VmPeakMB, proc.VmSwapMB)
				fmt.Printf("  rss_anon=%dMB (tensors)  rss_file=%dMB (mmap'd weights)\n",
					proc.RssAnonMB, proc.RssFileMB)
				if proc.VmSwapMB > 0 {
					fmt.Printf("  [CRIT] %dMB of inference process is in swap!\n", proc.VmSwapMB)
				}
			}

			// smaps_rollup — Brendan Gregg's WSS
			smaps, err := sandbox.ReadSmapsRollup(ctx, sb, pid)
			if err == nil {
				fmt.Printf("  smaps_rollup: rss=%dMB  referenced(WSS)=%dMB  swap=%dMB\n",
					smaps.RssMB, smaps.ReferencedMB, smaps.SwapMB)
			}

			// wchan — what is the process blocked on right now?
			wchan, err := sandbox.WChan(ctx, sb, pid)
			if err == nil {
				diagnosis := diagWChan(wchan)
				fmt.Printf("  wchan: %s  → %s\n", wchan, diagnosis)
			}

			// perf stat if privileged
			if *privileged {
				fmt.Println("\nperf stat (2s sample):")
				perfOut, err := sandbox.PerfStat(ctx, sb, pid, 2)
				if err != nil {
					fmt.Printf("  %v\n", err)
				} else {
					for _, line := range strings.Split(perfOut, "\n") {
						if strings.Contains(line, "misses") || strings.Contains(line, "cycles") || strings.Contains(line, "instructions") {
							fmt.Printf("  %s\n", strings.TrimSpace(line))
						}
					}
				}
			}

			// Latest inference output
			fmt.Println()
			log, _ := sb.ExecShell(ctx, "tail -5 /tmp/inference.log 2>/dev/null")
			if log.Stdout != "" {
				fmt.Println("inference log (last 5 lines):")
				for _, l := range strings.Split(strings.TrimSpace(log.Stdout), "\n") {
					fmt.Printf("  %s\n", l)
				}
			}
		}
	}
}

func printInferenceLog(ctx context.Context, sb *sandbox.Sandbox) {
	result, _ := sb.ExecShell(ctx, "cat /tmp/inference.log")
	fmt.Println("\n── full inference log ──")
	fmt.Println(result.Stdout)
}

func printFinalUSE(ctx context.Context, sb *sandbox.Sandbox) {
	fmt.Println("\n══ FINAL USE REPORT (post-inference) ══")
	sandbox.USEReport(ctx, sb)
}

// diagWChan maps kernel wait channel names to human-readable inference diagnosis.
func diagWChan(wchan string) string {
	switch {
	case wchan == "0":
		return "running on CPU (good)"
	case strings.HasPrefix(wchan, "futex"):
		return "waiting on lock — normal (BLAS/numpy thread pool sync)"
	case wchan == "do_swap_page":
		return "*** SWAP FAULT during inference — severe latency ***"
	case wchan == "io_schedule" || wchan == "blkdev_issue_discard":
		return "blocked on disk I/O — bad during inference"
	case strings.HasPrefix(wchan, "ep_poll") || wchan == "poll_schedule_timeout":
		return "idle (epoll/select wait)"
	case wchan == "schedule" || wchan == "schedule_hrtimeout":
		return "sleeping (voluntary)"
	default:
		return "unknown — check kernel docs"
	}
}
