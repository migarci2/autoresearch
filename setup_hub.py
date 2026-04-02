"""Operator entrypoint for the Codex MNIST swarm."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import shutil
import subprocess
import time
from pathlib import Path

from swarm_auth import (
    account_capacity_status,
    auth_doctor,
    capture_account_login,
    hydrate_account_to_home,
    import_auth_payload,
    list_active_leases,
    list_vault_accounts,
    revoke_account,
    sync_remote_vault,
    verify_account,
)
from swarm_config import (
    AgentConfig,
    SwarmConfig,
    build_codex_exec_command,
    build_codex_login_command,
    build_runtime_env,
    codex_login_status,
    compose_command,
    compose_ready,
    copy_codex_auth_payload,
    ensure_shared_layout,
    load_swarm_config,
    repo_root,
    write_runtime_manifests,
)


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _git_worktree_exists(path: Path) -> bool:
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return path.exists()
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            candidate = Path(line.split(" ", 1)[1].strip())
            if candidate.resolve() == path.resolve():
                return True
    return False


def _branch_exists(branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo_root(),
        check=False,
    )
    return result.returncode == 0


def _has_git_metadata(root: Path) -> bool:
    if (root / ".git").exists():
        return True
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and "true" in (result.stdout or "").lower()


def _seed_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "homes",
        "shared",
        "worktrees",
        "swarm_logs",
        "runtime",
        "runs",
        "results",
        "queue",
        "dev",
        "secrets",
    }
    return {name for name in names if name in ignored}


def _seed_worktree_copy(config: SwarmConfig, spec: AgentConfig) -> None:
    runtime = config.runtime(spec.key)
    if runtime.worktree.exists() and any(runtime.worktree.iterdir()):
        return
    runtime.worktree.parent.mkdir(parents=True, exist_ok=True)
    if runtime.worktree.exists():
        shutil.rmtree(runtime.worktree)
    shutil.copytree(config.root, runtime.worktree, ignore=_seed_ignore, symlinks=False)


def _ensure_worktree(config: SwarmConfig, spec: AgentConfig, base_ref: str) -> None:
    runtime = config.runtime(spec.key)
    if runtime.worktree.exists() and _git_worktree_exists(runtime.worktree):
        return
    if not _has_git_metadata(config.root):
        _seed_worktree_copy(config, spec)
        return
    runtime.worktree.parent.mkdir(parents=True, exist_ok=True)
    if _branch_exists(spec.branch):
        subprocess.run(
            ["git", "worktree", "add", str(runtime.worktree), spec.branch],
            cwd=repo_root(),
            check=True,
        )
    else:
        subprocess.run(
            ["git", "worktree", "add", "-b", spec.branch, str(runtime.worktree), base_ref],
            cwd=repo_root(),
            check=True,
        )


def _pid_from_file(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _stop_pid(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _best_results(config: SwarmConfig) -> dict[str, object]:
    best_results_path = config.paths.shared_dir / "best_results.json"
    data = _read_json(
        best_results_path,
        {"global_best": None, "by_role": {}, "by_family": {}, "updated_at": None},
    )
    return data if isinstance(data, dict) else {"global_best": None, "by_role": {}, "by_family": {}, "updated_at": None}


def _ensure_account_capacity(config: SwarmConfig, requested_agents: int | None = None) -> None:
    requested = requested_agents if requested_agents is not None else len(config.enabled_agents())
    capacity = sum(
        int(row["free_capacity"])
        for row in account_capacity_status(config)
        if row["enabled"] and row["present_in_vault"] and row["healthy"]
    )
    if requested > 0 and capacity < requested:
        raise RuntimeError(
            f"Insufficient usable Codex account capacity: need {requested}, usable {capacity}. "
            "Add/import more vaulted accounts or increase capacity on existing ones."
        )


def ensure_layout(config: SwarmConfig) -> None:
    ensure_shared_layout(config)
    for spec in config.agents.values():
        runtime = config.runtime(spec.key)
        runtime.home.mkdir(parents=True, exist_ok=True)
        runtime.worktree.parent.mkdir(parents=True, exist_ok=True)


def bootstrap(config: SwarmConfig, base_ref: str) -> None:
    ensure_layout(config)
    for spec in config.agents.values():
        _ensure_worktree(config, spec, base_ref)
    manifests = write_runtime_manifests(config)
    print("Bootstrap complete.")
    print(f"Config: {config.source_path}")
    print(f"Shared dir: {config.paths.shared_dir}")
    print(f"Vault dir: {config.auth.vault_path}")
    print(f"Runtime manifests: {manifests['summary']}")
    print()
    print("Next steps:")
    print("  python scripts/swarm auth add <account-id>")
    print("  python scripts/swarm auth verify")
    if config.deploy.runpod_enabled:
        print("  python scripts/swarm runpod start")
    else:
        print("  python scripts/swarm up")


def _runpod_supervisor_paths(config: SwarmConfig) -> dict[str, Path]:
    return {
        "status": config.paths.runtime_dir / "runpod-supervisor.json",
        "processes": config.paths.runtime_dir / "runpod-processes.json",
        "events": config.paths.runtime_dir / "runpod-events.jsonl",
        "log": config.paths.logs_dir / "runpod-supervisor.log",
    }


def _runpod_supervisor_cmd(config: SwarmConfig) -> list[str]:
    return [
        "python3",
        str(config.root / "scripts" / "runpod_supervisor.py"),
        "--config",
        str(config.source_path),
    ]


def runpod_bootstrap(config: SwarmConfig, *, base_ref: str = "master") -> None:
    ensure_layout(config)
    bootstrap(config, base_ref=base_ref)


def runpod_start(config: SwarmConfig, *, foreground: bool = False) -> None:
    _ensure_account_capacity(config, requested_agents=len(config.enabled_agents()))
    runpod_bootstrap(config)
    status = _read_json(_runpod_supervisor_paths(config)["status"], default={})
    if isinstance(status, dict) and status.get("pid") and _pid_alive(int(status["pid"])):
        print(f"RunPod supervisor already running (pid={status['pid']})")
        return
    cmd = _runpod_supervisor_cmd(config)
    env = {
        **os.environ,
        "RUNPOD_MODE": "1",
        "RUNPOD_WORKSPACE_DIR": str(config.deploy.workspace_dir),
        "RUNPOD_APP_DIR": str(config.deploy.app_dir),
        "SWARM_CONFIG_PATH": str(config.source_path),
        "AUTORESEARCH_CONFIG_PATH": str(config.source_path),
        "AUTORESEARCH_CONFIG_FILE": str(config.source_path),
    }
    if foreground:
        subprocess.run(cmd, cwd=config.root, env=env, check=True)
        return

    paths = _runpod_supervisor_paths(config)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    with paths["log"].open("ab") as handle:
        process = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=config.root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"RunPod supervisor started (pid={process.pid})")


def runpod_stop(config: SwarmConfig) -> None:
    status = _read_json(_runpod_supervisor_paths(config)["status"], default={})
    pid = int(status.get("pid")) if isinstance(status, dict) and status.get("pid") else None
    if not pid:
        print("RunPod supervisor is not running.")
        return
    if not _stop_pid(pid):
        print(f"Unable to stop RunPod supervisor pid={pid}")
        return
    print(f"Sent SIGTERM to RunPod supervisor pid={pid}")


def runpod_status(config: SwarmConfig) -> None:
    paths = _runpod_supervisor_paths(config)
    status = _read_json(paths["status"], default={})
    processes = _read_json(paths["processes"], default={})
    if not isinstance(status, dict) or not status:
        print("RunPod supervisor: no status yet")
        return
    print("RunPod:")
    for key in (
        "state",
        "pid",
        "hostname",
        "started_at",
        "last_seen_at",
        "uptime_seconds",
        "workspace_dir",
        "app_dir",
        "dashboard_port",
        "ssh_port",
        "pod_id",
        "public_ip",
        "tcp_port_22",
        "tcp_port_8080",
    ):
        if key in status:
            print(f"  {key}: {status.get(key)}")
    print("Processes:")
    for process in (processes.get("processes", []) if isinstance(processes, dict) else []):
        print(
            f"  {process.get('name')}: state={process.get('state')} pid={process.get('pid')} "
            f"restarts={process.get('restart_count')} log={process.get('log_path')}"
        )


def runpod_doctor(config: SwarmConfig) -> int:
    issues: list[str] = []
    if not config.deploy.runpod_enabled:
        issues.append("config.deploy.target is not set to runpod")
    if not config.deploy.workspace_dir.exists():
        issues.append(f"workspace dir missing: {config.deploy.workspace_dir}")
    if not config.deploy.app_dir.exists():
        issues.append(f"app dir missing: {config.deploy.app_dir}")
    if not config.auth.vault_path.exists():
        issues.append(f"vault dir missing: {config.auth.vault_path}")
    doctor = auth_doctor(config)
    issues.extend(doctor.get("findings", []))
    status = _read_json(_runpod_supervisor_paths(config)["status"], default={})
    if not isinstance(status, dict) or not status:
        issues.append("runpod supervisor status file missing")
    elif not _pid_alive(int(status.get("pid")) if status.get("pid") else None):
        issues.append("runpod supervisor pid is not alive")

    if issues:
        print("RunPod doctor found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("RunPod doctor: healthy")
    return 0


def runpod_logs(config: SwarmConfig, service: str, tail: int = 100) -> None:
    paths = _runpod_supervisor_paths(config)
    if service in {"supervisor", "runpod-supervisor"}:
        log_path = paths["log"]
    elif service == "dashboard":
        log_path = config.paths.logs_dir / "dashboard.log"
    else:
        log_path = config.runtime(service).log_path
    if not log_path.exists():
        print(f"No log file yet: {log_path}")
        return
    lines = log_path.read_text(errors="replace").splitlines()
    for line in lines[-tail:]:
        print(line)


def launch_agent(
    config: SwarmConfig,
    spec: AgentConfig,
    *,
    model_override: str | None = None,
    dry_run: bool = False,
) -> None:
    runtime = config.runtime(spec.key)
    if not runtime.prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found for {spec.agent_id}: {runtime.prompt_path}")

    if not dry_run:
        rows = account_capacity_status(config)
        available = [
            row
            for row in rows
            if row["healthy"] and row["present_in_vault"] and row["free_capacity"] > 0
        ]
        if not available:
            raise RuntimeError("No healthy vaulted Codex accounts are available for launch")

    env = build_runtime_env(config, runtime)
    cmd = build_codex_exec_command(
        config,
        runtime,
        model_override=model_override,
        last_message_path=runtime.last_message_path,
    )

    if dry_run:
        print(
            f"[dry-run] {spec.agent_id}: {shlex.join(cmd)}\n"
            f"  HOME={runtime.home}\n"
            f"  WORKTREE={runtime.worktree}\n"
            f"  PROMPT={runtime.prompt_path}\n"
            f"  MODEL={spec.model}\n"
            f"  REASONING_EFFORT={spec.reasoning_effort}"
        )
        return

    runtime.log_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.pid_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime.prompt_path.open("rb") as prompt_handle, runtime.log_path.open("ab") as log_handle:
        process = subprocess.Popen(  # noqa: S603
            [
                "python3",
                str(config.root / "scripts" / "run_agent.py"),
                "run",
                "--role",
                spec.role,
                *([] if not model_override else ["--model", model_override]),
            ],
            cwd=config.root,
            env=env,
            stdin=prompt_handle,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    runtime.pid_path.write_text(f"{process.pid}\n")
    print(f"Launched {spec.agent_id} (pid={process.pid})")


def _compose_launch(config: SwarmConfig, services: list[str], *, dry_run: bool) -> None:
    if not config.compose.file.exists():
        raise FileNotFoundError(f"Compose file not found: {config.compose.file}")
    cmd = compose_command(config, "up", "-d", *services)
    if dry_run:
        print(f"[dry-run] {shlex.join(cmd)}")
        return
    subprocess.run(cmd, cwd=config.root, check=True)


def launch(
    config: SwarmConfig,
    *,
    role: str | None = None,
    model_override: str | None = None,
    dry_run: bool = False,
    use_compose: bool | None = None,
) -> None:
    if use_compose is None:
        use_compose = compose_ready(config)
    requested_agents = 1 if role else len(config.enabled_agents())
    _ensure_account_capacity(config, requested_agents=requested_agents)

    if role is not None:
        spec = config.agent(role)
        if use_compose:
            _compose_launch(config, [spec.service_name], dry_run=dry_run)
            return
        launch_agent(config, spec, model_override=model_override, dry_run=dry_run)
        return

    enabled_specs = config.enabled_agents()
    if use_compose:
        _compose_launch(
            config,
            [spec.service_name for spec in enabled_specs]
            + [config.compose.dashboard_service, config.compose.toolbox_service],
            dry_run=dry_run,
        )
        return

    for spec in enabled_specs:
        launch_agent(config, spec, model_override=model_override, dry_run=dry_run)


def auth_fanout(config: SwarmConfig, source_role: str) -> None:
    source = config.runtime(source_role)
    copied_total: list[Path] = []
    for spec in config.enabled_agents():
        if spec.key == config.agent(source_role).key:
            continue
        copied_total.extend(
            copy_codex_auth_payload(
                config,
                source_home=source.home,
                target_home=config.runtime(spec.key).home,
            )
        )
    print(f"Copied {len(copied_total)} auth payload files from {source.spec.agent_id}")


def print_logs(config: SwarmConfig, role: str, tail: int = 100) -> None:
    runtime = config.runtime(role)
    if compose_ready(config):
        subprocess.run(
            compose_command(config, "logs", "--tail", str(tail), runtime.spec.service_name),
            cwd=config.root,
            check=True,
        )
        return
    if not runtime.log_path.exists():
        print(f"No log file yet for {runtime.spec.agent_id}: {runtime.log_path}")
        return
    lines = runtime.log_path.read_text(errors="replace").splitlines()
    for line in lines[-tail:]:
        print(line)


def login(config: SwarmConfig, role: str, *, device_auth: bool = False) -> None:
    runtime = config.runtime(role)
    runtime.home.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        build_codex_login_command(config, device_auth=device_auth),
        cwd=config.root,
        env={**os.environ, "HOME": str(runtime.home), "CODEX_HOME": str(runtime.home / ".codex")},
        check=True,
    )


def down(config: SwarmConfig, *, use_compose: bool | None = None) -> None:
    if use_compose is None:
        use_compose = compose_ready(config)

    if use_compose:
        if not config.compose.file.exists():
            raise FileNotFoundError(f"Compose file not found: {config.compose.file}")
        subprocess.run(compose_command(config, "down"), cwd=config.root, check=True)
        return

    stopped = 0
    for spec in config.enabled_agents():
        runtime = config.runtime(spec.key)
        pid = _pid_from_file(runtime.pid_path)
        if _stop_pid(pid):
            stopped += 1
        runtime.pid_path.unlink(missing_ok=True)
    print(f"Stopped {stopped} local agent process(es)")


def _compose_status(config: SwarmConfig) -> None:
    if not compose_ready(config):
        print("Compose: unavailable")
        return
    result = subprocess.run(
        compose_command(config, "ps"),
        cwd=config.root,
        text=True,
        capture_output=True,
        check=False,
    )
    print("Compose status:")
    print(result.stdout.strip() or result.stderr.strip() or "(no compose output)")


def _print_account_summary(config: SwarmConfig) -> None:
    rows = account_capacity_status(config)
    leases = list_active_leases(config, include_stale=True)
    print("Accounts:")
    if not rows:
        print("  none configured")
        return
    for row in rows:
        health = row.get("health", {})
        print(
            f"  {row['account_id']}: enabled={row['enabled']} healthy={row['healthy']} "
            f"capacity={row['used_capacity']}/{row['capacity']} free={row['free_capacity']} "
            f"vault={row['present_in_vault']} state={health.get('state', 'unknown')}"
        )
        if health.get("last_error"):
            print(f"    last_error: {health['last_error']}")
        if health.get("suspect_until"):
            print(f"    suspect_until: {health['suspect_until']}")
    print("Leases:")
    if not leases:
        print("  none")
    for lease in leases:
        print(
            f"  {lease.get('session_id')}: account={lease.get('account_id')} "
            f"agent={lease.get('agent_id')} role={lease.get('role')} stale={lease.get('stale')}"
        )


def print_status(config: SwarmConfig, *, show_compose: bool = True) -> None:
    best_results = _best_results(config)
    print(f"Config: {config.source_path}")
    print(f"Deploy target: {config.deploy.target}")
    print(f"Compose file: {config.compose.file}")
    print(f"Vault dir: {config.auth.vault_path}")
    print(f"Configured account capacity: {config.total_enabled_account_capacity()}")
    if show_compose and compose_ready(config):
        _compose_status(config)
    else:
        print("Compose: unavailable")
    print()
    for spec in config.agents.values():
        runtime = config.runtime(spec.key)
        logged_in, login_output = codex_login_status(config, runtime.home)
        pid = _pid_from_file(runtime.pid_path)
        alive = _pid_alive(pid)
        print(
            f"{spec.agent_id}: role={spec.role} model={spec.model} eff={spec.reasoning_effort} "
            f"logged_in={logged_in} alive={alive} pid={pid} worktree={runtime.worktree}"
        )
        if not logged_in and login_output:
            print(f"  login: {login_output}")
    print()
    _print_account_summary(config)
    print()
    print("Shared state:")
    print(f"  best_results: {config.paths.shared_dir / 'best_results.json'}")
    print(f"  experiment_log: {config.paths.shared_dir / 'experiment_log.jsonl'}")
    print(f"  claims_dir: {config.paths.shared_dir / 'claims'}")
    global_best = best_results.get("global_best")
    if isinstance(global_best, dict):
        metrics = global_best.get("metrics", {})
        print("  global_best:")
        print(f"    experiment_key: {global_best.get('experiment_key')}")
        print(f"    role: {global_best.get('role')}")
        print(f"    agent_id: {global_best.get('agent_id')}")
        for key in ("val_errors", "val_accuracy", "val_loss", "model_family", "run_mode"):
            if key in metrics:
                print(f"    {key}: {metrics[key]}")
    else:
        print("  global_best: none")


def smoke(config: SwarmConfig) -> None:
    ensure_layout(config)
    write_runtime_manifests(config)
    print("Smoke check:")
    print(f"  config: {config.source_path}")
    print(f"  compose_ready: {compose_ready(config)}")
    print(f"  enabled_agents: {[spec.agent_id for spec in config.enabled_agents()]}")
    print(f"  enabled_account_capacity: {config.total_enabled_account_capacity()}")
    _print_account_summary(config)
    if config.enabled_agents():
        spec = config.enabled_agents()[0]
        launch_agent(config, spec, dry_run=True)
    if compose_ready(config):
        subprocess.run(compose_command(config, "config"), cwd=config.root, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operator entrypoint for the Codex MNIST swarm")
    parser.add_argument(
        "--config",
        default=os.environ.get("SWARM_CONFIG_PATH", "config/swarm.yaml"),
        help="Path to the swarm YAML config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Create shared state, worktrees, homes and runtime manifests")
    bootstrap_parser.add_argument("--base-ref", default="master", help="Git ref used for new worktrees")

    def add_launch_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--role", help="Launch only one agent role, e.g. A")
        target.add_argument("--model", help="Override the Codex model for the launched agent(s)")
        target.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
        target.add_argument("--compose", action="store_true", help="Force compose mode when a compose file is present")
        target.add_argument("--no-compose", action="store_true", help="Force host-based launching even if compose exists")

    launch_parser = subparsers.add_parser("launch", help="Launch all agents or one selected role")
    add_launch_args(launch_parser)
    up_parser = subparsers.add_parser("up", help="Alias for launch")
    add_launch_args(up_parser)

    down_parser = subparsers.add_parser("down", help="Stop agents or bring compose down")
    down_parser.add_argument("--compose", action="store_true", help="Force compose mode when a compose file is present")
    down_parser.add_argument("--no-compose", action="store_true", help="Force host-based shutdown")

    status_parser = subparsers.add_parser("status", help="Show agent, compose and shared-state status")
    status_parser.add_argument("--compose", action="store_true", help="Force compose status if available")
    status_parser.add_argument("--no-compose", action="store_true", help="Hide compose status")

    login_parser = subparsers.add_parser("login", help="Run guided Codex login for one agent home")
    login_parser.add_argument("role", help="Agent role or agent id, e.g. A or agent-a")
    login_parser.add_argument("--device-auth", action="store_true", help="Use device code auth")

    auth_parser = subparsers.add_parser("auth", help="Auth helpers")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", required=True)
    add_parser = auth_subparsers.add_parser("add", help="Capture one ChatGPT-backed Codex account into the vault")
    add_parser.add_argument("account_id", help="Account id from config/swarm.yaml")
    add_parser.add_argument("--device-auth", action="store_true", help="Use device code auth")
    import_parser = auth_subparsers.add_parser("import", help="Import a local auth.json into the vault")
    import_parser.add_argument("account_id", help="Account id from config/swarm.yaml")
    import_parser.add_argument("--from", dest="source_path", required=True, help="Path to auth.json")
    verify_parser = auth_subparsers.add_parser("verify", help="Verify vaulted accounts with codex login status")
    verify_parser.add_argument("--account", dest="account_id", help="Optional account id")
    auth_subparsers.add_parser("list", help="List vaulted accounts and active leases")
    auth_subparsers.add_parser("doctor", help="Diagnose auth/vault issues")
    revoke_parser = auth_subparsers.add_parser("revoke", help="Remove one vaulted account")
    revoke_parser.add_argument("account_id", help="Account id from config/swarm.yaml")
    sync_parser = auth_subparsers.add_parser("sync-remote", help="Sync the encrypted vault to a remote SSH host")
    sync_parser.add_argument("--host", required=True, help="SSH target, e.g. user@host")
    hydrate_parser = auth_subparsers.add_parser("hydrate", help="Materialize one vaulted account into one agent home")
    hydrate_parser.add_argument("account_id", help="Account id from config/swarm.yaml")
    hydrate_parser.add_argument("--role", required=True, help="Target role or agent id")

    fanout_parser = auth_subparsers.add_parser("fanout", help="Legacy helper to copy auth payloads from one home to others")
    fanout_parser.add_argument("--from", dest="source_role", required=True, help="Source role or agent id")
    auth_fanout_parser = subparsers.add_parser("auth-fanout", help="Alias for auth fanout")
    auth_fanout_parser.add_argument("--from", dest="source_role", required=True, help="Source role or agent id")

    logs_parser = subparsers.add_parser("logs", help="Tail a single agent log")
    logs_parser.add_argument("role", help="Agent role or agent id, e.g. A or agent-a")
    logs_parser.add_argument("--tail", type=int, default=100, help="Number of lines to print")

    runpod_parser = subparsers.add_parser("runpod", help="RunPod-specific deployment helpers")
    runpod_subparsers = runpod_parser.add_subparsers(dest="runpod_command", required=True)
    runpod_bootstrap_parser = runpod_subparsers.add_parser("bootstrap", help="Prepare the RunPod workspace")
    runpod_bootstrap_parser.add_argument("--base-ref", default="master", help="Git ref used for new worktrees")
    runpod_start_parser = runpod_subparsers.add_parser("start", help="Start the RunPod supervisor")
    runpod_start_parser.add_argument("--foreground", action="store_true", help="Run the supervisor in the foreground")
    runpod_subparsers.add_parser("stop", help="Stop the RunPod supervisor")
    runpod_subparsers.add_parser("status", help="Show RunPod supervisor state")
    runpod_subparsers.add_parser("doctor", help="Run RunPod deployment checks")
    runpod_logs_parser = runpod_subparsers.add_parser("logs", help="Tail one RunPod service log")
    runpod_logs_parser.add_argument("service", help="Service name: supervisor, dashboard, A, agent-a, etc.")
    runpod_logs_parser.add_argument("--tail", type=int, default=100, help="Number of lines to print")
    runpod_sync_parser = runpod_subparsers.add_parser("sync-vault", help="Sync the encrypted vault to a RunPod host over SSH")
    runpod_sync_parser.add_argument("--host", required=True, help="SSH target, e.g. root@1.2.3.4")

    subparsers.add_parser("leases", help="Print active account leases")
    subparsers.add_parser("smoke", help="Validate config, manifests and compose if available")
    subparsers.add_parser("test", help="Alias for smoke")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = load_swarm_config(args.config)

    if args.command == "bootstrap":
        bootstrap(config, base_ref=args.base_ref)
        return

    if args.command in {"launch", "up"}:
        if args.compose:
            use_compose = True
        elif args.no_compose:
            use_compose = False
        else:
            use_compose = None
        launch(
            config,
            role=args.role,
            model_override=args.model,
            dry_run=args.dry_run,
            use_compose=use_compose,
        )
        return

    if args.command == "down":
        if args.compose:
            use_compose = True
        elif args.no_compose:
            use_compose = False
        else:
            use_compose = None
        down(config, use_compose=use_compose)
        return

    if args.command == "status":
        print_status(config, show_compose=not args.no_compose)
        return

    if args.command == "login":
        login(config, args.role, device_auth=args.device_auth)
        return

    if args.command == "auth-fanout":
        auth_fanout(config, args.source_role)
        return

    if args.command == "smoke":
        smoke(config)
        return

    if args.command == "test":
        smoke(config)
        return

    if args.command == "logs":
        print_logs(config, args.role, tail=args.tail)
        return

    if args.command == "leases":
        for lease in list_active_leases(config, include_stale=True):
            print(json.dumps(lease, sort_keys=True))
        return

    if args.command == "runpod":
        if args.runpod_command == "bootstrap":
            runpod_bootstrap(config, base_ref=args.base_ref)
            return
        if args.runpod_command == "start":
            runpod_start(config, foreground=args.foreground)
            return
        if args.runpod_command == "stop":
            runpod_stop(config)
            return
        if args.runpod_command == "status":
            runpod_status(config)
            return
        if args.runpod_command == "doctor":
            raise SystemExit(runpod_doctor(config))
        if args.runpod_command == "logs":
            runpod_logs(config, args.service, tail=args.tail)
            return
        if args.runpod_command == "sync-vault":
            sync_remote_vault(config, args.host)
            print(f"Synced vault to {args.host}")
            return

    if args.command == "auth":
        if args.auth_command == "fanout":
            auth_fanout(config, args.source_role)
            return
        if args.auth_command == "add":
            print(json.dumps(capture_account_login(config, args.account_id, device_auth=args.device_auth), indent=2, sort_keys=True))
            return
        if args.auth_command == "import":
            print(json.dumps(import_auth_payload(config, args.account_id, Path(args.source_path), source="manual_import"), indent=2, sort_keys=True))
            return
        if args.auth_command == "verify":
            account_ids = [args.account_id] if args.account_id else [row["account_id"] for row in list_vault_accounts(config) if row["present_in_vault"]]
            results = [verify_account(config, account_id) for account_id in account_ids]
            print(json.dumps(results, indent=2, sort_keys=True))
            if any(not row["ok"] for row in results):
                raise SystemExit(1)
            return
        if args.auth_command == "list":
            print(
                json.dumps(
                    {
                        "accounts": list_vault_accounts(config),
                        "capacity": account_capacity_status(config),
                        "leases": list_active_leases(config, include_stale=True),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if args.auth_command == "doctor":
            result = auth_doctor(config)
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["ok"]:
                raise SystemExit(1)
            return
        if args.auth_command == "revoke":
            revoke_account(config, args.account_id)
            print(f"Revoked {args.account_id}")
            return
        if args.auth_command == "sync-remote":
            sync_remote_vault(config, args.host)
            print(f"Synced vault to {args.host}")
            return
        if args.auth_command == "hydrate":
            print(json.dumps(hydrate_account_to_home(config, args.account_id, config.runtime(args.role).home), indent=2, sort_keys=True))
            return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
