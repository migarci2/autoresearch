"""Single-pod supervisor for RunPod deployments."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_config import (
    build_runtime_env,
    ensure_shared_layout,
    load_swarm_config,
    runtime_env_allowlist,
    write_runtime_manifests,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _is_process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _signal_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        return


@dataclass
class ManagedProcess:
    name: str
    kind: str
    argv: list[str]
    cwd: Path
    env: dict[str, str]
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    restart_count: int = 0
    started_at: str | None = None
    last_exit_at: str | None = None
    last_exit_code: int | None = None
    next_restart_at: float = 0.0
    state: str = "idle"
    _log_handle: Any = field(default=None, repr=False)

    @property
    def pid(self) -> int | None:
        if self.process is None:
            return None
        return int(self.process.pid)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "argv": self.argv,
            "cwd": str(self.cwd),
            "pid": self.pid,
            "state": self.state,
            "restart_count": self.restart_count,
            "started_at": self.started_at,
            "last_exit_at": self.last_exit_at,
            "last_exit_code": self.last_exit_code,
            "next_restart_at": self.next_restart_at,
            "log_path": str(self.log_path),
        }


class RunpodSupervisor:
    def __init__(self, config_path: str) -> None:
        self.config = load_swarm_config(config_path)
        ensure_shared_layout(self.config)
        write_runtime_manifests(self.config)
        self.runtime_dir = self.config.paths.runtime_dir
        self.supervisor_status_path = self.runtime_dir / "runpod-supervisor.json"
        self.processes_status_path = self.runtime_dir / "runpod-processes.json"
        self.events_path = self.runtime_dir / "runpod-events.jsonl"
        self.log_path = self.config.paths.logs_dir / "runpod-supervisor.log"
        self.started_at = _now_iso()
        self.should_stop = False
        self.processes: dict[str, ManagedProcess] = self._build_processes()
        self.restart_policy = self.config.deploy.supervisor_restart_policy

    def _build_processes(self) -> dict[str, ManagedProcess]:
        processes: dict[str, ManagedProcess] = {}
        for spec in self.config.enabled_agents():
            runtime = self.config.runtime(spec.key)
            env = build_runtime_env(
                self.config,
                runtime,
                extra={
                    "RUNPOD_MODE": "1",
                    "AUTORESEARCH_DASHBOARD_PORT": str(self.config.deploy.dashboard_port),
                },
            )
            processes[spec.agent_id] = ManagedProcess(
                name=spec.agent_id,
                kind="agent",
                argv=[
                    "python3",
                    str(self.config.root / "scripts" / "run_agent.py"),
                    "run",
                    "--role",
                    spec.role,
                ],
                cwd=self.config.root,
                env=env,
                log_path=runtime.log_path,
            )

        dashboard_log = self.config.paths.logs_dir / "dashboard.log"
        dashboard_env = runtime_env_allowlist()
        dashboard_env.update(
            {
                "AUTORESEARCH_SHARED_DIR": str(self.config.paths.shared_dir),
                "AUTORESEARCH_CONFIG_FILE": str(self.config.source_path),
                "AUTORESEARCH_CONFIG_PATH": str(self.config.source_path),
                "SWARM_CONFIG_PATH": str(self.config.source_path),
                "RUNPOD_MODE": "1",
                "RUNPOD_POD_ID": os.environ.get("RUNPOD_POD_ID", ""),
                "RUNPOD_PUBLIC_IP": os.environ.get("RUNPOD_PUBLIC_IP", ""),
                "RUNPOD_TCP_PORT_22": os.environ.get("RUNPOD_TCP_PORT_22", ""),
                "RUNPOD_TCP_PORT_8080": os.environ.get("RUNPOD_TCP_PORT_8080", ""),
            }
        )
        processes["dashboard"] = ManagedProcess(
            name="dashboard",
            kind="dashboard",
            argv=[
                "python3",
                "-m",
                "uvicorn",
                "dashboard_app:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.config.deploy.dashboard_port),
            ],
            cwd=self.config.root,
            env=dashboard_env,
            log_path=dashboard_log,
        )
        return processes

    def _log(self, message: str, *, event_type: str | None = None, **payload: Any) -> None:
        line = f"[runpod-supervisor] {message}"
        print(line, flush=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")
        if event_type:
            _append_jsonl(
                self.events_path,
                {
                    "recorded_at": _now_iso(),
                    "event_type": event_type,
                    **payload,
                },
            )

    def _supervisor_payload(self) -> dict[str, Any]:
        pod_id = os.environ.get("RUNPOD_POD_ID")
        public_ip = os.environ.get("RUNPOD_PUBLIC_IP")
        tcp_port_22 = os.environ.get("RUNPOD_TCP_PORT_22")
        tcp_port_8080 = os.environ.get("RUNPOD_TCP_PORT_8080")
        running = sum(1 for proc in self.processes.values() if proc.state == "running")
        return {
            "state": "stopping" if self.should_stop else "running",
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
            "started_at": self.started_at,
            "last_seen_at": _now_iso(),
            "uptime_seconds": max(
                0.0,
                time.time() - datetime.fromisoformat(self.started_at).timestamp(),
            ),
            "restart_policy": self.restart_policy,
            "services_total": len(self.processes),
            "services_running": running,
            "workspace_dir": str(self.config.deploy.workspace_dir),
            "app_dir": str(self.config.deploy.app_dir),
            "dashboard_port": self.config.deploy.dashboard_port,
            "ssh_port": self.config.deploy.ssh_port,
            "pod_id": pod_id,
            "public_ip": public_ip,
            "tcp_port_22": tcp_port_22,
            "tcp_port_8080": tcp_port_8080,
            "runpod_mode": os.environ.get("RUNPOD_MODE", "0"),
        }

    def _write_status(self) -> None:
        _write_json(self.supervisor_status_path, self._supervisor_payload())
        _write_json(
            self.processes_status_path,
            {
                "generated_at": _now_iso(),
                "processes": [proc.as_dict() for proc in self.processes.values()],
            },
        )

    def _open_log(self, process: ManagedProcess):
        process.log_path.parent.mkdir(parents=True, exist_ok=True)
        return process.log_path.open("ab")

    def _start_process(self, process: ManagedProcess) -> None:
        if process.process and process.process.poll() is None:
            return
        process._log_handle = self._open_log(process)
        child = subprocess.Popen(  # noqa: S603
            process.argv,
            cwd=process.cwd,
            env=process.env,
            stdout=process._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process.process = child
        process.state = "running"
        process.started_at = _now_iso()
        process.next_restart_at = 0.0
        self._log(
            f"started {process.name} (pid={child.pid})",
            event_type="service_started",
            service=process.name,
            kind=process.kind,
            pid=child.pid,
        )

    def _stop_process(self, process: ManagedProcess, *, sig: int = signal.SIGTERM) -> None:
        if not process.process:
            return
        pid = process.process.pid
        _signal_process_group(pid, sig)

    def _finalize_process(self, process: ManagedProcess) -> None:
        if process._log_handle is not None:
            process._log_handle.close()
            process._log_handle = None

    def _check_processes(self) -> None:
        now = time.time()
        for process in self.processes.values():
            if process.process is None:
                if not self.should_stop and process.next_restart_at <= now:
                    self._start_process(process)
                continue

            returncode = process.process.poll()
            if returncode is None:
                process.state = "running"
                continue

            process.last_exit_code = int(returncode)
            process.last_exit_at = _now_iso()
            process.state = "exited"
            self._finalize_process(process)
            self._log(
                f"{process.name} exited with code {returncode}",
                event_type="service_exited",
                service=process.name,
                kind=process.kind,
                exit_code=returncode,
            )
            process.process = None

            if self.should_stop or self.restart_policy == "never":
                continue

            process.restart_count += 1
            backoff = min(60, 2 ** min(process.restart_count, 5))
            process.next_restart_at = now + backoff
            process.state = "restarting"
            self._log(
                f"scheduling restart for {process.name} in {backoff}s",
                event_type="service_restart_scheduled",
                service=process.name,
                kind=process.kind,
                backoff_seconds=backoff,
                restart_count=process.restart_count,
            )

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            self.should_stop = True
            self._log(
                f"received signal {signum}, stopping supervisor",
                event_type="supervisor_signal",
                signal=signum,
            )

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _shutdown(self) -> None:
        self.should_stop = True
        for process in self.processes.values():
            self._stop_process(process, sig=signal.SIGTERM)
        deadline = time.time() + 15
        while time.time() < deadline:
            active = False
            for process in self.processes.values():
                if process.process and process.process.poll() is None:
                    active = True
            if not active:
                break
            time.sleep(0.5)
        for process in self.processes.values():
            if process.process and process.process.poll() is None:
                self._stop_process(process, sig=signal.SIGKILL)
            self._finalize_process(process)
            process.process = None
            process.state = "stopped"
        self._write_status()
        self._log("supervisor stopped", event_type="supervisor_stopped")

    def run(self) -> int:
        self._install_signal_handlers()
        self._log("supervisor starting", event_type="supervisor_started")
        self._write_status()
        try:
            while not self.should_stop:
                self._check_processes()
                self._write_status()
                time.sleep(2.0)
        finally:
            self._shutdown()
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod single-pod supervisor")
    parser.add_argument(
        "--config",
        default=os.environ.get("SWARM_CONFIG_PATH", "config/swarm.yaml"),
        help="Path to the swarm YAML config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    supervisor = RunpodSupervisor(args.config)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
