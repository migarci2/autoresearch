"""Foreground Codex agent entrypoint.

Use this inside a Compose service or a dedicated container. It resolves the
agent identity from CLI flags or environment variables, then launches the
Codex loop in the foreground so container logs remain the source of truth.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_auth import (
    acquire_account_lease,
    account_capacity_status,
    auth_doctor,
    hydrate_account_to_home,
    import_auth_payload,
    list_active_leases,
    list_vault_accounts,
    mark_account_suspect,
    refresh_account_lease,
    release_account_lease,
    sync_remote_vault,
    verify_account,
)
from swarm_config import (
    build_codex_exec_command,
    build_codex_login_command,
    build_runtime_env,
    codex_login_status,
    ensure_shared_layout,
    load_swarm_config,
    write_runtime_manifests,
)


def _agent_key(args: argparse.Namespace) -> str:
    value = args.role or args.agent or os.environ.get("AUTORESEARCH_ROLE") or os.environ.get("AUTORESEARCH_AGENT_ID")
    if not value:
        raise SystemExit("Missing agent identity. Pass --role/--agent or set AUTORESEARCH_ROLE/AUTORESEARCH_AGENT_ID.")
    return value


def _print_runtime(config, runtime) -> None:
    print(f"agent_id={runtime.spec.agent_id}")
    print(f"role={runtime.spec.role}")
    print(f"model={runtime.spec.model}")
    print(f"reasoning_effort={runtime.spec.reasoning_effort}")
    print(f"home={runtime.home}")
    print(f"worktree={runtime.worktree}")
    print(f"prompt_path={runtime.prompt_path}")
    print(f"config_path={config.source_path}")
    print(f"runtime_manifest={runtime.manifest_path}")
    print(f"enabled_account_capacity={config.total_enabled_account_capacity()}")


def run_agent(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    ensure_shared_layout(config)
    write_runtime_manifests(config)
    runtime = config.runtime(_agent_key(args))
    runtime.home.mkdir(parents=True, exist_ok=True)
    backoff = max(5, int(args.poll_interval))

    if args.dry_run:
        env = build_runtime_env(config, runtime)
        cmd = build_codex_exec_command(
            config,
            runtime,
            model_override=args.model,
            last_message_path=runtime.last_message_path,
        )
        print(shlex.join(cmd))
        _print_runtime(config, runtime)
        return 0

    attempts = 0
    while True:
        attempts += 1
        lease = None
        stop_event = threading.Event()
        heartbeat_thread = None
        try:
            lease = acquire_account_lease(config, runtime, timeout=args.account_timeout, poll_interval=args.poll_interval)
            account_id = str(lease["account_id"])
            hydrate_account_to_home(config, account_id, runtime.home)
            logged_in, login_output = codex_login_status(config, runtime.home)
            if not logged_in and not args.allow_unauthed:
                mark_account_suspect(config, account_id, reason="hydrated_login_invalid", detail=login_output)
                release_account_lease(config, lease["session_id"])
                lease = None
                if attempts >= config.auth.max_auth_retries:
                    print(f"[{runtime.spec.agent_id}] account {account_id} failed login validation", file=sys.stderr)
                    if login_output:
                        print(login_output, file=sys.stderr)
                    return 1
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            def heartbeat_loop() -> None:
                while not stop_event.wait(config.auth.lease_heartbeat_seconds):
                    try:
                        refreshed = refresh_account_lease(config, lease["session_id"])
                        if not refreshed:
                            return
                    except Exception:
                        return

            heartbeat_thread = threading.Thread(
                target=heartbeat_loop,
                name=f"auth-lease-{runtime.spec.agent_id}",
                daemon=True,
            )
            heartbeat_thread.start()

            env = build_runtime_env(
                config,
                runtime,
                extra={
                    "AUTORESEARCH_ACCOUNT_ID": account_id,
                    "AUTORESEARCH_ACCOUNT_SESSION": str(lease["session_id"]),
                },
            )
            cmd = build_codex_exec_command(
                config,
                runtime,
                model_override=args.model,
                last_message_path=runtime.last_message_path,
            )
            print(
                f"Launching foreground agent {runtime.spec.agent_id} with account {account_id}",
                file=sys.stderr,
            )
            with runtime.prompt_path.open("rb") as prompt_handle:
                process = subprocess.run(
                    cmd,
                    cwd=config.root,
                    env=env,
                    stdin=prompt_handle,
                    check=False,
                )
            returncode = int(process.returncode)
            if returncode == 0:
                return 0

            logged_in, login_output = codex_login_status(config, runtime.home)
            if not logged_in and not args.allow_unauthed:
                mark_account_suspect(
                    config,
                    account_id,
                    reason="post_exec_auth_invalid",
                    detail=login_output or f"codex exec exited {returncode}",
                )
                if attempts < config.auth.max_auth_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
            return returncode
        finally:
            stop_event.set()
            if heartbeat_thread and heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=1.0)
            if lease is not None:
                release_account_lease(config, lease["session_id"])


def login(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    runtime = config.runtime(_agent_key(args))
    runtime.home.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        build_codex_login_command(config, device_auth=args.device_auth),
        cwd=config.root,
        env={**os.environ, "HOME": str(runtime.home), "CODEX_HOME": str(runtime.home / ".codex")},
        check=False,
    )
    return int(result.returncode)


def login_status(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    runtime = config.runtime(_agent_key(args))
    logged_in, login_output = codex_login_status(config, runtime.home)
    print(f"agent_id={runtime.spec.agent_id}")
    print(f"home={runtime.home}")
    print(f"logged_in={logged_in}")
    print(f"enabled_account_capacity={config.total_enabled_account_capacity()}")
    if config.accounts:
        rows = account_capacity_status(config)
        print(f"available_accounts={sum(1 for row in rows if row['healthy'] and row['present_in_vault'] and row['free_capacity'] > 0)}")
    if login_output:
        print(login_output)
    return 0 if logged_in else 1


def auth_import(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    result = import_auth_payload(config, args.account, Path(args.path), source="manual_import")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def auth_verify(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    result = verify_account(config, args.account)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def auth_list(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    payload = {
        "accounts": list_vault_accounts(config),
        "leases": list_active_leases(config, include_stale=True),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def auth_doctor_cmd(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    result = auth_doctor(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def auth_sync_remote(args: argparse.Namespace) -> int:
    config = load_swarm_config(args.config)
    sync_remote_vault(config, args.host)
    print(f"Synced vault to {args.host}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foreground Codex agent entrypoint")
    parser.add_argument(
        "--config",
        default=os.environ.get("SWARM_CONFIG_PATH", "config/swarm.yaml"),
        help="Path to the swarm config",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="Launch the selected agent in the foreground")
    run_parser.add_argument("--role", help="Agent role, e.g. A")
    run_parser.add_argument("--agent", help="Agent id, e.g. agent-a")
    run_parser.add_argument("--model", help="Override the agent model for this run")
    run_parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    run_parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between lease retries")
    run_parser.add_argument("--account-timeout", type=float, help="Optional timeout for waiting on an account lease")
    run_parser.add_argument(
        "--allow-unauthed",
        action="store_true",
        help="Skip login validation and launch anyway",
    )

    login_parser = subparsers.add_parser("login", help="Run guided Codex login for the selected agent home")
    login_parser.add_argument("--role", help="Agent role, e.g. A")
    login_parser.add_argument("--agent", help="Agent id, e.g. agent-a")
    login_parser.add_argument("--device-auth", action="store_true", help="Use device-code authentication")

    status_parser = subparsers.add_parser("login-status", help="Check whether the selected agent home is logged in")
    status_parser.add_argument("--role", help="Agent role, e.g. A")
    status_parser.add_argument("--agent", help="Agent id, e.g. agent-a")

    auth_parser = subparsers.add_parser("auth-import", help="Import an auth.json file into the encrypted vault")
    auth_parser.add_argument("--account", required=True, help="Account id from config/swarm.yaml")
    auth_parser.add_argument("--path", required=True, help="Path to auth.json")

    verify_parser = subparsers.add_parser("auth-verify", help="Verify one vaulted account")
    verify_parser.add_argument("--account", required=True, help="Account id from config/swarm.yaml")

    subparsers.add_parser("auth-list", help="List vaulted accounts and active leases")
    subparsers.add_parser("auth-doctor", help="Diagnose auth/vault issues")

    sync_parser = subparsers.add_parser("auth-sync-remote", help="Sync the encrypted vault to a remote SSH target")
    sync_parser.add_argument("--host", required=True, help="SSH target, e.g. user@host")

    subparsers.add_parser("print-runtime", help="Print the resolved runtime configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command = args.command or "run"

    if command == "run":
        return run_agent(args)
    if command == "login":
        return login(args)
    if command == "login-status":
        return login_status(args)
    if command == "auth-import":
        return auth_import(args)
    if command == "auth-verify":
        return auth_verify(args)
    if command == "auth-list":
        return auth_list(args)
    if command == "auth-doctor":
        return auth_doctor_cmd(args)
    if command == "auth-sync-remote":
        return auth_sync_remote(args)
    if command == "print-runtime":
        config = load_swarm_config(args.config)
        ensure_shared_layout(config)
        write_runtime_manifests(config)
        runtime = config.runtime(os.environ.get("AUTORESEARCH_ROLE") or os.environ.get("AUTORESEARCH_AGENT_ID") or "A")
        _print_runtime(config, runtime)
        return 0
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
