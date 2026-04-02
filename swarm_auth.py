"""Vault, account-pool, and Codex auth helpers for the MNIST swarm."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from swarm_config import (
    AccountConfig,
    AgentRuntime,
    SwarmConfig,
    build_codex_login_command,
    codex_login_status,
    ensure_shared_layout,
    runtime_env_allowlist,
)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@contextlib.contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def auth_root(config: SwarmConfig) -> Path:
    return config.paths.shared_dir / "accounts"


def leases_dir(config: SwarmConfig) -> Path:
    return auth_root(config) / "leases"


def runtime_cache_dir(config: SwarmConfig) -> Path:
    return auth_root(config) / "runtime_cache"


def health_path(config: SwarmConfig) -> Path:
    return auth_root(config) / "health.json"


def events_path(config: SwarmConfig) -> Path:
    return auth_root(config) / "events.jsonl"


def vault_manifest_path(config: SwarmConfig) -> Path:
    return config.auth.vault_path / "manifest.json"


def vault_account_path(config: SwarmConfig, account_id: str) -> Path:
    return config.auth.vault_path / f"{account_id}.vault"


def lease_path(config: SwarmConfig, session_id: str) -> Path:
    return leases_dir(config) / f"{session_id}.json"


def _passphrase(config: SwarmConfig) -> str:
    file_path = os.environ.get(config.auth.passphrase_file_env)
    if file_path:
        return Path(file_path).read_text().strip()
    value = os.environ.get(config.auth.passphrase_env)
    if value:
        return value.strip()
    raise RuntimeError(
        "Vault passphrase missing. Set "
        f"{config.auth.passphrase_env} or {config.auth.passphrase_file_env}."
    )


def _derive_keys(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    material = hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        200_000,
        dklen=64,
    )
    return material[:32], material[32:]


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    counter = 0
    while sum(len(block) for block in blocks) < length:
        counter_bytes = counter.to_bytes(8, "big")
        blocks.append(hmac.new(key, nonce + counter_bytes, hashlib.sha256).digest())
        counter += 1
    return b"".join(blocks)[:length]


def _seal_payload(config: SwarmConfig, payload: dict[str, Any]) -> dict[str, Any]:
    plain = json.dumps(payload, sort_keys=True).encode("utf-8")
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    enc_key, mac_key = _derive_keys(_passphrase(config), salt)
    cipher = bytes(a ^ b for a, b in zip(plain, _keystream(enc_key, nonce, len(plain))))
    tag = hmac.new(mac_key, salt + nonce + cipher, hashlib.sha256).digest()
    return {
        "version": 1,
        "kdf": "pbkdf2-sha256",
        "cipher": "xor-hmac-stream",
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ciphertext_b64": base64.b64encode(cipher).decode("ascii"),
        "tag_b64": base64.b64encode(tag).decode("ascii"),
    }


def _open_payload(config: SwarmConfig, envelope: dict[str, Any]) -> dict[str, Any]:
    salt = base64.b64decode(envelope["salt_b64"])
    nonce = base64.b64decode(envelope["nonce_b64"])
    cipher = base64.b64decode(envelope["ciphertext_b64"])
    tag = base64.b64decode(envelope["tag_b64"])
    enc_key, mac_key = _derive_keys(_passphrase(config), salt)
    expected = hmac.new(mac_key, salt + nonce + cipher, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, tag):
        raise RuntimeError("Vault decrypt failed: invalid passphrase or corrupted payload")
    plain = bytes(a ^ b for a, b in zip(cipher, _keystream(enc_key, nonce, len(cipher))))
    return json.loads(plain.decode("utf-8"))


def ensure_codex_file_storage(config: SwarmConfig, home: Path) -> Path:
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    config_path = codex_home / "config.toml"
    lines: list[str] = []
    if config_path.exists():
        lines = config_path.read_text().splitlines()
    wanted = f'cli_auth_credentials_store = "{config.auth.force_credentials_store}"'
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith("cli_auth_credentials_store"):
            lines[index] = wanted
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(wanted)
    config_path.write_text("\n".join(lines).rstrip() + "\n")
    return config_path


def codex_auth_env(config: SwarmConfig, home: Path) -> dict[str, str]:
    env = runtime_env_allowlist()
    env.update(
        {
            "HOME": str(home),
            "CODEX_HOME": str(home / ".codex"),
        }
    )
    return env


def _collect_auth_payloads(config: SwarmConfig, home: Path) -> dict[str, str]:
    codex_home = home / ".codex"
    payloads: dict[str, str] = {}
    for relative in config.codex.auth_payload_files:
        src = codex_home / relative
        if not src.exists() or src.is_dir():
            continue
        payloads[relative] = base64.b64encode(src.read_bytes()).decode("ascii")
    return payloads


def _write_auth_payloads(config: SwarmConfig, payloads: dict[str, str], home: Path) -> list[Path]:
    codex_home = home / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for relative, encoded in payloads.items():
        dst = codex_home / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(encoded))
        written.append(dst)
    return written


def _account_manifest(config: SwarmConfig) -> dict[str, Any]:
    payload = _read_json(vault_manifest_path(config), default={"accounts": {}, "updated_at": None})
    if not isinstance(payload, dict):
        payload = {"accounts": {}, "updated_at": None}
    payload.setdefault("accounts", {})
    payload.setdefault("updated_at", None)
    return payload


def _save_account_manifest(config: SwarmConfig, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now_iso()
    _write_json(vault_manifest_path(config), manifest)


def record_auth_event(config: SwarmConfig, event_type: str, **payload: Any) -> None:
    _append_jsonl(
        events_path(config),
        {
            "recorded_at": _now_iso(),
            "event_type": event_type,
            **payload,
        },
    )


def _health_payload(config: SwarmConfig) -> dict[str, Any]:
    payload = _read_json(health_path(config), default={"accounts": {}, "updated_at": None})
    if not isinstance(payload, dict):
        payload = {"accounts": {}, "updated_at": None}
    payload.setdefault("accounts", {})
    payload.setdefault("updated_at", None)
    return payload


def _save_health_payload(config: SwarmConfig, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now_iso()
    _write_json(health_path(config), payload)


def _update_account_health(config: SwarmConfig, account_id: str, **updates: Any) -> dict[str, Any]:
    with _locked_file(health_path(config)):
        payload = _health_payload(config)
        account = payload["accounts"].setdefault(account_id, {})
        account.update(updates)
        if "failure_count" in updates and isinstance(account.get("failure_count"), int):
            account["failure_count"] = max(0, int(account["failure_count"]))
        payload["accounts"][account_id] = account
        _save_health_payload(config, payload)
        return dict(account)


def list_vault_accounts(config: SwarmConfig) -> list[dict[str, Any]]:
    ensure_shared_layout(config)
    manifest = _account_manifest(config)
    health = _health_payload(config).get("accounts", {})
    rows: list[dict[str, Any]] = []
    for account_id, spec in config.accounts.items():
        vault_info = manifest.get("accounts", {}).get(account_id, {})
        health_info = health.get(account_id, {})
        rows.append(
            {
                "account_id": account_id,
                "label": spec.label,
                "capacity": spec.capacity,
                "enabled": spec.enabled,
                "workspace_id": spec.workspace_id,
                "role_affinity": list(spec.role_affinity),
                "metadata": spec.metadata,
                "present_in_vault": vault_account_path(config, account_id).exists(),
                "vault": vault_info,
                "health": health_info,
            }
        )
    rows.sort(key=lambda row: row["account_id"])
    return rows


def import_auth_payload(config: SwarmConfig, account_id: str, auth_json_path: Path, *, source: str) -> dict[str, Any]:
    ensure_shared_layout(config)
    spec = config.account(account_id)
    auth_json_path = auth_json_path.resolve()
    if not auth_json_path.exists():
        raise FileNotFoundError(f"Auth file not found: {auth_json_path}")

    temp_home = Path(tempfile.mkdtemp(prefix=f"codex-import-{account_id}-"))
    try:
        ensure_codex_file_storage(config, temp_home)
        target = temp_home / ".codex" / "auth.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(auth_json_path, target)
        payloads = _collect_auth_payloads(config, temp_home)
    finally:
        shutil.rmtree(temp_home, ignore_errors=True)

    sha256 = hashlib.sha256(auth_json_path.read_bytes()).hexdigest()
    sealed = _seal_payload(
        config,
        {
            "account_id": account_id,
            "captured_at": _now_iso(),
            "source": source,
            "workspace_id": spec.workspace_id,
            "payloads": payloads,
        },
    )
    _write_json(vault_account_path(config, account_id), sealed)

    with _locked_file(vault_manifest_path(config)):
        manifest = _account_manifest(config)
        manifest["accounts"][account_id] = {
            "account_id": account_id,
            "label": spec.label,
            "capacity": spec.capacity,
            "enabled": spec.enabled,
            "workspace_id": spec.workspace_id,
            "source": source,
            "captured_at": _now_iso(),
            "auth_payload_count": len(payloads),
            "sha256": sha256,
            "last_verified_at": None,
        }
        _save_account_manifest(config, manifest)

    _update_account_health(
        config,
        account_id,
        state="healthy",
        suspect_until=None,
        last_error=None,
        failure_count=0,
        last_imported_at=_now_iso(),
    )
    record_auth_event(config, "account_imported", account_id=account_id, source=source)
    return {
        "account_id": account_id,
        "label": spec.label,
        "vault_path": str(vault_account_path(config, account_id)),
        "sha256": sha256,
        "auth_payload_count": len(payloads),
    }


def capture_account_login(config: SwarmConfig, account_id: str, *, device_auth: bool = False) -> dict[str, Any]:
    ensure_shared_layout(config)
    config.account(account_id)
    temp_home = Path(tempfile.mkdtemp(prefix=f"codex-capture-{account_id}-"))
    try:
        ensure_codex_file_storage(config, temp_home)
        result = subprocess.run(
            build_codex_login_command(config, device_auth=device_auth),
            cwd=config.root,
            env=codex_auth_env(config, temp_home),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"codex login failed for account {account_id}")
        auth_json = temp_home / ".codex" / "auth.json"
        if not auth_json.exists():
            raise RuntimeError(f"Codex login did not produce auth.json for account {account_id}")
        return import_auth_payload(
            config,
            account_id,
            auth_json,
            source="device_auth" if device_auth else "local_browser",
        )
    finally:
        shutil.rmtree(temp_home, ignore_errors=True)


def hydrate_account_to_home(config: SwarmConfig, account_id: str, home: Path) -> dict[str, Any]:
    ensure_shared_layout(config)
    envelope = _read_json(vault_account_path(config, account_id), default={})
    if not envelope:
        raise FileNotFoundError(f"Vault seed not found for account {account_id}")
    opened = _open_payload(config, envelope)
    ensure_codex_file_storage(config, home)
    written = _write_auth_payloads(config, opened.get("payloads", {}), home)
    record_auth_event(config, "account_hydrated", account_id=account_id, home=str(home))
    return {
        "account_id": account_id,
        "home": str(home),
        "files": [str(path) for path in written],
    }


def verify_account(config: SwarmConfig, account_id: str) -> dict[str, Any]:
    temp_home = Path(tempfile.mkdtemp(prefix=f"codex-verify-{account_id}-"))
    try:
        hydrate_account_to_home(config, account_id, temp_home)
        logged_in, output = codex_login_status(config, temp_home)
        now = _now_iso()
        with _locked_file(vault_manifest_path(config)):
            manifest = _account_manifest(config)
            entry = manifest.get("accounts", {}).setdefault(account_id, {})
            entry["last_verified_at"] = now
            entry["last_verify_ok"] = bool(logged_in)
            entry["last_verify_output"] = output[-800:]
            manifest["accounts"][account_id] = entry
            _save_account_manifest(config, manifest)
        if logged_in:
            _update_account_health(
                config,
                account_id,
                state="healthy",
                suspect_until=None,
                last_verified_at=now,
                last_error=None,
                failure_count=0,
            )
        else:
            mark_account_suspect(config, account_id, reason="verify_failed", detail=output[-800:])
        record_auth_event(config, "account_verified", account_id=account_id, ok=bool(logged_in))
        return {
            "account_id": account_id,
            "ok": bool(logged_in),
            "output": output,
        }
    finally:
        shutil.rmtree(temp_home, ignore_errors=True)


def revoke_account(config: SwarmConfig, account_id: str) -> None:
    vault_account_path(config, account_id).unlink(missing_ok=True)
    with _locked_file(vault_manifest_path(config)):
        manifest = _account_manifest(config)
        manifest.get("accounts", {}).pop(account_id, None)
        _save_account_manifest(config, manifest)
    _update_account_health(config, account_id, state="revoked", suspect_until=None)
    record_auth_event(config, "account_revoked", account_id=account_id)


def mark_account_suspect(config: SwarmConfig, account_id: str, *, reason: str, detail: str | None = None) -> None:
    now = time.time()
    until = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now + config.auth.suspect_cooldown_seconds))
    health = _health_payload(config).get("accounts", {}).get(account_id, {})
    failure_count = int(health.get("failure_count", 0)) + 1
    _update_account_health(
        config,
        account_id,
        state="suspect",
        suspect_until=until,
        last_error=reason,
        last_error_detail=(detail or "")[-1200:],
        failure_count=failure_count,
        last_failed_at=_now_iso(),
    )
    record_auth_event(config, "account_suspect", account_id=account_id, reason=reason)


def list_active_leases(config: SwarmConfig, *, include_stale: bool = False) -> list[dict[str, Any]]:
    ensure_shared_layout(config)
    leases: list[dict[str, Any]] = []
    for path in sorted(leases_dir(config).glob("*.json")):
        payload = _read_json(path, default={})
        if not payload:
            continue
        heartbeat_at = payload.get("heartbeat_at") or payload.get("leased_at")
        age = None
        if heartbeat_at:
            try:
                from datetime import datetime

                age = time.time() - datetime.fromisoformat(heartbeat_at).timestamp()
            except ValueError:
                age = None
        stale = age is None or age > payload.get("ttl_seconds", config.auth.lease_ttl_seconds)
        if not stale and payload.get("hostname") == socket.gethostname():
            pid = payload.get("pid")
            try:
                os.kill(int(pid), 0)
            except Exception:
                stale = True
        payload["stale"] = stale
        payload["lease_path"] = str(path)
        if stale and not include_stale:
            continue
        leases.append(payload)
    return leases


def _cleanup_stale_leases(config: SwarmConfig) -> None:
    for payload in list_active_leases(config, include_stale=True):
        if payload.get("stale"):
            Path(payload["lease_path"]).unlink(missing_ok=True)


def account_capacity_status(config: SwarmConfig) -> list[dict[str, Any]]:
    ensure_shared_layout(config)
    _cleanup_stale_leases(config)
    health = _health_payload(config).get("accounts", {})
    active_leases = list_active_leases(config, include_stale=False)
    usage: dict[str, list[dict[str, Any]]] = {}
    for lease in active_leases:
        usage.setdefault(str(lease.get("account_id")), []).append(lease)

    rows: list[dict[str, Any]] = []
    now = time.time()
    for spec in config.accounts.values():
        info = health.get(spec.account_id, {})
        suspect_until = info.get("suspect_until")
        healthy = True
        if info.get("state") == "suspect" and suspect_until:
            try:
                from datetime import datetime

                healthy = datetime.fromisoformat(suspect_until).timestamp() <= now
            except ValueError:
                healthy = False
        rows.append(
            {
                "account_id": spec.account_id,
                "label": spec.label,
                "capacity": spec.capacity,
                "enabled": spec.enabled,
                "workspace_id": spec.workspace_id,
                "role_affinity": list(spec.role_affinity),
                "leases": usage.get(spec.account_id, []),
                "used_capacity": len(usage.get(spec.account_id, [])),
                "free_capacity": max(0, spec.capacity - len(usage.get(spec.account_id, []))),
                "present_in_vault": vault_account_path(config, spec.account_id).exists(),
                "health": info,
                "healthy": healthy
                and info.get("state") != "revoked"
                and vault_account_path(config, spec.account_id).exists(),
            }
        )
    return rows


def acquire_account_lease(
    config: SwarmConfig,
    runtime: AgentRuntime,
    *,
    timeout: float | None = None,
    poll_interval: float = 5.0,
) -> dict[str, Any]:
    ensure_shared_layout(config)
    start = time.time()
    while True:
        with _locked_file(health_path(config)):
            _cleanup_stale_leases(config)
            rows = account_capacity_status(config)
            candidates = []
            for row in rows:
                spec = config.account(row["account_id"])
                affinity_bonus = 0 if runtime.spec.role in spec.role_affinity else 1
                if not spec.enabled or not row["present_in_vault"] or not row["healthy"]:
                    continue
                if row["free_capacity"] <= 0:
                    continue
                candidates.append((affinity_bonus, row["used_capacity"], row["account_id"], row))
            if candidates:
                _, _, account_id, row = sorted(candidates)[0]
                session_id = f"{runtime.spec.agent_id}-{secrets.token_hex(6)}"
                payload = {
                    "session_id": session_id,
                    "account_id": account_id,
                    "agent_id": runtime.spec.agent_id,
                    "role": runtime.spec.role,
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "leased_at": _now_iso(),
                    "heartbeat_at": _now_iso(),
                    "ttl_seconds": config.auth.lease_ttl_seconds,
                }
                _write_json(lease_path(config, session_id), payload)
                record_auth_event(
                    config,
                    "lease_acquired",
                    account_id=account_id,
                    session_id=session_id,
                    agent_id=runtime.spec.agent_id,
                    role=runtime.spec.role,
                )
                return payload
        if timeout is not None and time.time() - start > timeout:
            raise TimeoutError("Timed out waiting for an available Codex account")
        time.sleep(poll_interval)


def refresh_account_lease(config: SwarmConfig, session_id: str) -> dict[str, Any] | None:
    path = lease_path(config, session_id)
    if not path.exists():
        return None
    with _locked_file(path):
        payload = _read_json(path, default={})
        if not payload:
            return None
        payload["heartbeat_at"] = _now_iso()
        payload["pid"] = os.getpid()
        payload["hostname"] = socket.gethostname()
        _write_json(path, payload)
        return payload


def release_account_lease(config: SwarmConfig, session_id: str) -> None:
    path = lease_path(config, session_id)
    payload = _read_json(path, default={})
    path.unlink(missing_ok=True)
    if payload:
        record_auth_event(
            config,
            "lease_released",
            account_id=payload.get("account_id"),
            session_id=session_id,
            agent_id=payload.get("agent_id"),
            role=payload.get("role"),
        )


def sync_remote_vault(config: SwarmConfig, ssh_target: str) -> list[str]:
    ensure_shared_layout(config)
    remote_dir = os.environ.get("SWARM_REMOTE_SYNC_PATH", config.auth.remote_sync_path)
    cmds = [
        f"mkdir -p {remote_dir}",
        f"tar -C {config.auth.vault_path} -cf - . | ssh {ssh_target} 'tar -C {remote_dir} -xf -'",
    ]
    subprocess.run(
        ["ssh", ssh_target, "mkdir", "-p", remote_dir],
        cwd=config.root,
        check=True,
    )
    tar_proc = subprocess.Popen(
        ["tar", "-C", str(config.auth.vault_path), "-cf", "-", "."],
        cwd=config.root,
        stdout=subprocess.PIPE,
    )
    try:
        ssh_proc = subprocess.run(
            ["ssh", ssh_target, f"tar -C {remote_dir} -xf -"],
            cwd=config.root,
            stdin=tar_proc.stdout,
            check=True,
        )
        if tar_proc.stdout:
            tar_proc.stdout.close()
        tar_return = tar_proc.wait()
        if tar_return != 0:
            raise subprocess.CalledProcessError(tar_return, tar_proc.args)
    finally:
        if tar_proc.stdout:
            tar_proc.stdout.close()
        if tar_proc.poll() is None:
            tar_proc.kill()
    record_auth_event(config, "vault_synced_remote", ssh_target=ssh_target, remote_dir=remote_dir)
    return cmds


def auth_doctor(config: SwarmConfig) -> dict[str, Any]:
    ensure_shared_layout(config)
    rows = list_vault_accounts(config)
    findings: list[str] = []
    for row in rows:
        account_id = row["account_id"]
        if not row["present_in_vault"]:
            findings.append(f"{account_id}: missing vault seed")
            continue
        health = row.get("health", {})
        if health.get("state") == "suspect":
            findings.append(f"{account_id}: suspect ({health.get('last_error') or 'unknown error'})")
        runtime_homes = []
        for agent_spec in config.enabled_agents():
            home = config.runtime(agent_spec.key).home / ".codex" / "auth.json"
            if home.exists():
                runtime_homes.append(str(home))
        if not runtime_homes:
            findings.append(f"{account_id}: not hydrated to any agent home yet")
    return {
        "ok": not findings,
        "findings": findings,
        "accounts": rows,
        "leases": list_active_leases(config, include_stale=True),
    }
