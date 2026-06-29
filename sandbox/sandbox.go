// Package sandbox provides a local Docker-backed sandbox with the same
// interface as Koyeb Sandboxes: create, exec, file read/write, destroy.
//
// Each sandbox runs in an isolated Docker container with:
//   - no network access (NetworkMode: none)
//   - memory + CPU + PID limits
//   - all Linux capabilities dropped
//
// In production Koyeb uses Cloud Hypervisor microVMs instead of containers,
// giving stronger isolation (guest kernel) and full perf/eBPF access inside
// the sandbox. This implementation is a local dev equivalent.
package sandbox

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/client"
	"github.com/docker/docker/pkg/stdcopy"
)

// Config holds sandbox creation parameters.
type Config struct {
	// Docker image to use. Defaults to "ubuntu:24.04".
	Image string
	// Name is optional — Docker assigns one if empty.
	Name string
	// MemoryMB caps RSS. 0 = unlimited (not recommended).
	MemoryMB int64
	// CPUs is the CPU quota (e.g. 2.0 = 2 full cores).
	CPUs float64
	// PidsLimit prevents fork bombs. Default 256.
	PidsLimit int64
	// NetworkMode defaults to "none" (no outbound access).
	// Set to "bridge" if the workload needs internet.
	NetworkMode string
	// Privileged grants CAP_SYS_ADMIN and perf_event access.
	// Required for perf/eBPF tools. Never use for untrusted code.
	Privileged bool
}

func (c *Config) applyDefaults() {
	if c.Image == "" {
		c.Image = "ubuntu:24.04"
	}
	if c.MemoryMB == 0 {
		c.MemoryMB = 512
	}
	if c.CPUs == 0 {
		c.CPUs = 1.0
	}
	if c.PidsLimit == 0 {
		c.PidsLimit = 256
	}
	if c.NetworkMode == "" {
		c.NetworkMode = "none"
	}
}

// Sandbox is a running isolated environment.
type Sandbox struct {
	// ID is the Docker container ID.
	ID     string
	// Image is the base image used.
	Image  string
	docker *client.Client
}

// Create pulls the image if needed, starts a container, and returns a Sandbox.
// The container runs `sleep infinity` so exec calls can be made at any time.
func Create(ctx context.Context, cfg Config) (*Sandbox, error) {
	cfg.applyDefaults()

	docker, err := client.NewClientWithOpts(
		client.FromEnv,
		client.WithAPIVersionNegotiation(),
	)
	if err != nil {
		return nil, fmt.Errorf("sandbox: docker client init: %w", err)
	}

	if err := ensureImage(ctx, docker, cfg.Image); err != nil {
		return nil, fmt.Errorf("sandbox: image pull: %w", err)
	}

	pids := cfg.PidsLimit
	capAdd := []string{}
	if cfg.Privileged {
		// Needed for perf_event_open, bpftrace, perf stat inside sandbox.
		capAdd = []string{"SYS_PERF_EVENT", "SYS_ADMIN", "SYS_PTRACE"}
	}

	resp, err := docker.ContainerCreate(
		ctx,
		&container.Config{
			Image: cfg.Image,
			// sleep infinity keeps the container alive between exec calls.
			Cmd:        []string{"sleep", "infinity"},
			Tty:        false,
			StopSignal: "SIGKILL",
		},
		&container.HostConfig{
			NetworkMode: container.NetworkMode(cfg.NetworkMode),
			CapDrop:     []string{"ALL"},
			CapAdd:      capAdd,
			Privileged:  cfg.Privileged,
			Resources: container.Resources{
				Memory:    cfg.MemoryMB * 1024 * 1024,
				NanoCPUs:  int64(cfg.CPUs * 1e9),
				PidsLimit: &pids,
			},
		},
		nil,
		nil,
		cfg.Name,
	)
	if err != nil {
		return nil, fmt.Errorf("sandbox: container create: %w", err)
	}

	if err := docker.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		// Clean up the created-but-not-started container.
		docker.ContainerRemove(ctx, resp.ID, container.RemoveOptions{Force: true})
		return nil, fmt.Errorf("sandbox: container start: %w", err)
	}

	return &Sandbox{ID: resp.ID, Image: cfg.Image, docker: docker}, nil
}

// ExecResult holds the output of a command run inside the sandbox.
type ExecResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// Exec runs a command inside the sandbox and returns combined output.
// cmd[0] is the binary; cmd[1:] are arguments.
func (s *Sandbox) Exec(ctx context.Context, cmd ...string) (*ExecResult, error) {
	exec, err := s.docker.ContainerExecCreate(ctx, s.ID, types.ExecConfig{
		Cmd:          cmd,
		AttachStdout: true,
		AttachStderr: true,
	})
	if err != nil {
		return nil, fmt.Errorf("exec create: %w", err)
	}

	attach, err := s.docker.ContainerExecAttach(ctx, exec.ID, types.ExecStartCheck{})
	if err != nil {
		return nil, fmt.Errorf("exec attach: %w", err)
	}
	defer attach.Close()

	var stdout, stderr bytes.Buffer
	if _, err := stdcopy.StdCopy(&stdout, &stderr, attach.Reader); err != nil {
		return nil, fmt.Errorf("exec read: %w", err)
	}

	inspect, err := s.docker.ContainerExecInspect(ctx, exec.ID)
	if err != nil {
		return nil, fmt.Errorf("exec inspect: %w", err)
	}

	return &ExecResult{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		ExitCode: inspect.ExitCode,
	}, nil
}

// ExecShell is a convenience wrapper that runs cmd via `bash -c`.
func (s *Sandbox) ExecShell(ctx context.Context, cmd string) (*ExecResult, error) {
	return s.Exec(ctx, "bash", "-c", cmd)
}

// WriteFile writes content to path inside the sandbox via tar stream.
// Parent directories are created automatically.
func (s *Sandbox) WriteFile(ctx context.Context, path, content string) error {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	hdr := &tar.Header{
		Name:    path,
		Mode:    0644,
		Size:    int64(len(content)),
		ModTime: time.Now(),
	}
	if err := tw.WriteHeader(hdr); err != nil {
		return fmt.Errorf("write file tar header: %w", err)
	}
	if _, err := tw.Write([]byte(content)); err != nil {
		return fmt.Errorf("write file tar body: %w", err)
	}
	tw.Close()

	return s.docker.CopyToContainer(
		ctx, s.ID, "/",
		&buf,
		types.CopyToContainerOptions{},
	)
}

// ReadFile reads the content of path from inside the sandbox.
func (s *Sandbox) ReadFile(ctx context.Context, path string) (string, error) {
	reader, _, err := s.docker.CopyFromContainer(ctx, s.ID, path)
	if err != nil {
		return "", fmt.Errorf("read file copy: %w", err)
	}
	defer reader.Close()

	tr := tar.NewReader(reader)
	if _, err := tr.Next(); err != nil {
		return "", fmt.Errorf("read file tar next: %w", err)
	}
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, tr); err != nil {
		return "", fmt.Errorf("read file tar read: %w", err)
	}
	return buf.String(), nil
}

// Destroy kills and removes the container. Always call this (defer it).
func (s *Sandbox) Destroy(ctx context.Context) error {
	err := s.docker.ContainerRemove(ctx, s.ID, container.RemoveOptions{
		Force: true,
	})
	if err != nil {
		return fmt.Errorf("sandbox destroy: %w", err)
	}
	return nil
}

// Status returns the current container status string (e.g. "running", "exited").
func (s *Sandbox) Status(ctx context.Context) (string, error) {
	info, err := s.docker.ContainerInspect(ctx, s.ID)
	if err != nil {
		return "", err
	}
	return info.State.Status, nil
}

// ensureImage pulls image if it is not already present locally.
func ensureImage(ctx context.Context, docker *client.Client, image string) error {
	images, err := docker.ImageList(ctx, types.ImageListOptions{
		Filters: filters.NewArgs(filters.Arg("reference", image)),
	})
	if err != nil {
		return err
	}
	if len(images) > 0 {
		return nil
	}
	fmt.Printf("pulling image %s...\n", image)
	out, err := docker.ImagePull(ctx, image, types.ImagePullOptions{})
	if err != nil {
		return err
	}
	io.Copy(io.Discard, out)
	out.Close()
	return nil
}
