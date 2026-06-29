// Package observe implements Brendan Gregg's USE method (Utilization,
// Saturation, Errors) for resources inside a sandbox, with special focus
// on the memory subsystem during LLM inference workloads.
//
// All metrics are gathered by reading /proc virtual filesystem entries
// inside the sandbox — no agent required, works in both Docker containers
// and microVMs (Cloud Hypervisor / Firecracker).
//
// In a microVM the guest kernel exposes hardware PMU counters via perf_event_open
// directly, so perf stat commands work without host privileges.
// In Docker you need --privileged or CAP_SYS_PERF_EVENT.
package sandbox

import (
	"context"
	"fmt"
	"strconv"
	"strings"

)

// MemInfo holds parsed /proc/meminfo fields relevant to LLM inference.
type MemInfo struct {
	TotalMB     int64
	AvailableMB int64
	UsedMB      int64
	ActiveMB    int64   // recently used pages
	InactiveMB  int64   // candidate for reclaim
	AnonPagesMB int64   // heap/stack — model weights live here (mmap'd tensors)
	MappedMB    int64   // mmap'd files (model weight files)
	SwapTotalMB int64
	SwapFreeMB  int64
	SwapUsedMB  int64
	// WorkingSetMB is Brendan Gregg's WSS approximation:
	// Active + AnonPages — what can't be reclaimed without hurting performance.
	WorkingSetMB int64
}

// ReadMemInfo reads and parses /proc/meminfo from inside the sandbox.
func ReadMemInfo(ctx context.Context, sb *Sandbox) (*MemInfo, error) {
	result, err := sb.Exec(ctx, "cat", "/proc/meminfo")
	if err != nil {
		return nil, fmt.Errorf("readmeminfo: %w", err)
	}

	fields := parseKeyValueKB(result.Stdout)
	m := &MemInfo{
		TotalMB:     fields["MemTotal"],
		AvailableMB: fields["MemAvailable"],
		ActiveMB:    fields["Active"],
		InactiveMB:  fields["Inactive"],
		AnonPagesMB: fields["AnonPages"],
		MappedMB:    fields["Mapped"],
		SwapTotalMB: fields["SwapTotal"],
		SwapFreeMB:  fields["SwapFree"],
	}
	m.UsedMB = m.TotalMB - m.AvailableMB
	m.SwapUsedMB = m.SwapTotalMB - m.SwapFreeMB
	// Brendan Gregg WSS approximation — pages in active use that can't be reclaimed.
	m.WorkingSetMB = m.ActiveMB + m.AnonPagesMB
	return m, nil
}

// PageFaultStats holds /proc/vmstat fault counters.
type PageFaultStats struct {
	// MinorFaults: page already in RAM, just needed a new PTE mapping.
	// Normal and cheap.
	MinorFaults int64
	// MajorFaults: page had to be fetched from disk. Expensive (~10ms+).
	// For LLM inference: spikes during model load, should → 0 once warm.
	// If MajorFaults keeps climbing during inference, working set > RAM.
	MajorFaults int64
	// SwapIn / SwapOut: pages moved between RAM and swap device.
	// ANY non-zero SwapIn during inference = severe performance degradation.
	SwapIn  int64
	SwapOut int64
}

// ReadPageFaults reads fault counters from /proc/vmstat.
func ReadPageFaults(ctx context.Context, sb *Sandbox) (*PageFaultStats, error) {
	result, err := sb.Exec(ctx, "cat", "/proc/vmstat")
	if err != nil {
		return nil, fmt.Errorf("readpagefaults: %w", err)
	}
	fields := parseKeyValueInt(result.Stdout)
	return &PageFaultStats{
		MinorFaults: fields["pgfault"],
		MajorFaults: fields["pgmajfault"],
		SwapIn:      fields["pswpin"],
		SwapOut:     fields["pswpout"],
	}, nil
}

// ProcStatus holds per-process memory stats from /proc/<pid>/status.
type ProcStatus struct {
	PID        int
	Comm       string
	VmRSSMB    int64 // resident set size — what's physically in RAM
	VmPeakMB   int64 // peak RSS ever seen
	VmSwapMB   int64 // how much of this process is in swap — BAD for inference
	RssAnonMB  int64 // anonymous (heap) pages — tensors allocated via malloc/mmap
	RssFileMB  int64 // file-backed pages — model weight files loaded via mmap
}

// ReadProcStatus reads /proc/<pid>/status for the given PID inside the sandbox.
func ReadProcStatus(ctx context.Context, sb *Sandbox, pid int) (*ProcStatus, error) {
	result, err := sb.Exec(ctx, "cat", fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return nil, fmt.Errorf("readprocstatus pid=%d: %w", pid, err)
	}
	fields := parseKeyValueKB(result.Stdout)
	strFields := parseKeyValueStr(result.Stdout)
	return &ProcStatus{
		PID:       pid,
		Comm:      strFields["Name"],
		VmRSSMB:   fields["VmRSS"],
		VmPeakMB:  fields["VmPeak"],
		VmSwapMB:  fields["VmSwap"],
		RssAnonMB: fields["RssAnon"],
		RssFileMB: fields["RssFile"],
	}, nil
}

// SmapsRollup is Brendan Gregg's preferred WSS data source.
// /proc/<pid>/smaps_rollup sums all VMAs — faster than reading full smaps.
type SmapsRollup struct {
	RssMB        int64 // total resident pages
	PssMB        int64 // proportional share (splits shared pages)
	ReferencedMB int64 // accessed since last clear — true WSS
	AnonMB       int64 // anonymous (tensor heap)
	SwapMB       int64 // in swap — should be 0 during inference
}

// ReadSmapsRollup reads /proc/<pid>/smaps_rollup.
func ReadSmapsRollup(ctx context.Context, sb *Sandbox, pid int) (*SmapsRollup, error) {
	result, err := sb.Exec(ctx, "cat", fmt.Sprintf("/proc/%d/smaps_rollup", pid))
	if err != nil {
		return nil, fmt.Errorf("readsmaps pid=%d: %w", pid, err)
	}
	fields := parseKeyValueKB(result.Stdout)
	return &SmapsRollup{
		RssMB:        fields["Rss"],
		PssMB:        fields["Pss"],
		ReferencedMB: fields["Referenced"],
		AnonMB:       fields["Anonymous"],
		SwapMB:       fields["Swap"],
	}, nil
}

// WChan reads /proc/<pid>/wchan — the kernel function the process is
// currently blocked in. Key values for LLM inference diagnosis:
//
//	futex_wait       — waiting on a lock (normal, BLAS thread pool sync)
//	do_swap_page     — taking a swap fault DURING inference (very bad)
//	io_schedule      — blocked on disk I/O (bad)
//	ep_poll          — waiting on epoll (idle, fine)
//	0                — running on CPU (good)
func WChan(ctx context.Context, sb *Sandbox, pid int) (string, error) {
	result, err := sb.Exec(ctx, "cat", fmt.Sprintf("/proc/%d/wchan", pid))
	if err != nil {
		return "", fmt.Errorf("wchan pid=%d: %w", pid, err)
	}
	return strings.TrimSpace(result.Stdout), nil
}

// PerfStat runs `perf stat` inside the sandbox for the given duration.
// Requires Privileged: true in the sandbox config, or a microVM (which
// exposes its own virtualised PMU to the guest kernel).
//
// Key counters for LLM inference:
//   - LLC-load-misses: last-level cache misses → goes to DRAM. High = memory BW bottleneck.
//   - dTLB-load-misses: TLB misses → page-table walks. High = large tensor address space.
//   - cache-misses: all cache levels. Proportional to memory bandwidth usage.
func PerfStat(ctx context.Context, sb *Sandbox, pid int, seconds int) (string, error) {
	cmd := fmt.Sprintf(
		"perf stat -e cache-misses,cache-references,LLC-load-misses,dTLB-load-misses,instructions,cycles -p %d sleep %d 2>&1",
		pid, seconds,
	)
	result, err := sb.ExecShell(ctx, cmd)
	if err != nil {
		return "", fmt.Errorf("perfstat: %w", err)
	}
	// perf stat writes to stderr by design
	out := result.Stdout + result.Stderr
	if strings.Contains(out, "Permission denied") || strings.Contains(out, "No such file") {
		return "", fmt.Errorf("perfstat: requires privileged sandbox or microVM — got: %s", out)
	}
	return out, nil
}

// USEReport runs the full USE method checklist for memory and CPU,
// prints a structured report, and flags any anomalies relevant to
// LLM inference workloads.
func USEReport(ctx context.Context, sb *Sandbox) error {
	fmt.Println("━━━ USE METHOD REPORT ━━━")
	fmt.Println("(Utilization · Saturation · Errors — Brendan Gregg)")
	fmt.Println()

	// ── MEMORY: Utilization ──
	mem, err := ReadMemInfo(ctx, sb)
	if err != nil {
		return err
	}
	utilPct := float64(mem.UsedMB) / float64(mem.TotalMB) * 100
	fmt.Printf("MEMORY utilization:  %d MB used / %d MB total (%.1f%%)\n",
		mem.UsedMB, mem.TotalMB, utilPct)
	fmt.Printf("  working set (WSS):  %d MB  (Active + AnonPages)\n", mem.WorkingSetMB)
	fmt.Printf("  available:          %d MB\n", mem.AvailableMB)
	fmt.Printf("  mmap'd tensors:     %d MB  (Mapped)\n", mem.MappedMB)

	// ── MEMORY: Saturation ──
	fmt.Println()
	faults, err := ReadPageFaults(ctx, sb)
	if err != nil {
		return err
	}
	fmt.Printf("MEMORY saturation:\n")
	fmt.Printf("  minor faults:  %d  (cheap, expected)\n", faults.MinorFaults)
	fmt.Printf("  major faults:  %d  (disk fetch — bad if non-zero during inference)\n", faults.MajorFaults)
	fmt.Printf("  swap in:       %d  (*** inference latency cliff if > 0 ***)\n", faults.SwapIn)
	fmt.Printf("  swap out:      %d\n", faults.SwapOut)
	fmt.Printf("  swap used:     %d MB / %d MB total\n", mem.SwapUsedMB, mem.SwapTotalMB)

	// ── Anomaly flags ──
	fmt.Println()
	fmt.Println("ANOMALY FLAGS:")
	flagged := false
	if mem.WorkingSetMB > mem.TotalMB*80/100 {
		fmt.Printf("  [WARN] WSS (%d MB) > 80%% of RAM — risk of eviction under pressure\n", mem.WorkingSetMB)
		flagged = true
	}
	if faults.MajorFaults > 1000 {
		fmt.Printf("  [WARN] %d major page faults — model not fully warm or WSS > RAM\n", faults.MajorFaults)
		flagged = true
	}
	if faults.SwapIn > 0 {
		fmt.Printf("  [CRIT] %d swap-ins detected — inference tokens will have high latency\n", faults.SwapIn)
		flagged = true
	}
	if mem.SwapUsedMB > mem.SwapTotalMB/4 {
		fmt.Printf("  [WARN] %d MB in swap — working set exceeds physical RAM\n", mem.SwapUsedMB)
		flagged = true
	}
	if !flagged {
		fmt.Println("  none — memory pressure looks healthy")
	}

	// ── CPU utilization (vmstat) ──
	fmt.Println()
	vmstat, err := sb.ExecShell(ctx, "vmstat 1 3 2>/dev/null | tail -1")
	if err == nil && vmstat.Stdout != "" {
		fields := strings.Fields(vmstat.Stdout)
		if len(fields) >= 15 {
			fmt.Printf("CPU utilization (last sample):\n")
			fmt.Printf("  us=%s sy=%s id=%s wa=%s (user/sys/idle/iowait)\n",
				fields[12], fields[13], fields[14], fields[15])
		}
	}

	fmt.Println()
	fmt.Println("━━━ END USE REPORT ━━━")
	return nil
}

// ── /proc parsing helpers ──

// parseKeyValueKB parses lines like "MemTotal:   16384 kB" → map[string]int64 in MB.
func parseKeyValueKB(s string) map[string]int64 {
	m := make(map[string]int64)
	for _, line := range strings.Split(s, "\n") {
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		key := strings.TrimSuffix(parts[0], ":")
		val, err := strconv.ParseInt(parts[1], 10, 64)
		if err != nil {
			continue
		}
		// Convert kB → MB
		if len(parts) == 3 && parts[2] == "kB" {
			val /= 1024
		}
		m[key] = val
	}
	return m
}

// parseKeyValueInt parses lines like "pgfault 123456" → map[string]int64.
func parseKeyValueInt(s string) map[string]int64 {
	m := make(map[string]int64)
	for _, line := range strings.Split(s, "\n") {
		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}
		val, err := strconv.ParseInt(parts[1], 10, 64)
		if err != nil {
			continue
		}
		m[parts[0]] = val
	}
	return m
}

// parseKeyValueStr parses lines like "Name:   python3" → map[string]string.
func parseKeyValueStr(s string) map[string]string {
	m := make(map[string]string)
	for _, line := range strings.Split(s, "\n") {
		idx := strings.Index(line, ":")
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])
		m[key] = val
	}
	return m
}
