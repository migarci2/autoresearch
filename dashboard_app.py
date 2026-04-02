from __future__ import annotations

import html
import json
import os
import re
import socket
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
SHARED_DIR = Path(os.environ.get("AUTORESEARCH_SHARED_DIR", str(BASE_DIR / "shared"))).resolve()
RUNTIME_DIR = SHARED_DIR / "runtime"
ACCOUNTS_DIR = SHARED_DIR / "accounts"
ACCOUNT_HEALTH_PATH = ACCOUNTS_DIR / "health.json"
ACCOUNT_LEASES_DIR = ACCOUNTS_DIR / "leases"
ACCOUNT_EVENTS_PATH = ACCOUNTS_DIR / "events.jsonl"
CLAIMS_DIR = SHARED_DIR / "claims"
LOCKS_DIR = SHARED_DIR / "locks"
AGENTS_DIR = SHARED_DIR / "agents"
BEST_RESULTS_PATH = SHARED_DIR / "best_results.json"
EXPERIMENT_LOG_PATH = SHARED_DIR / "experiment_log.jsonl"
GPU_LOCK_PATH = LOCKS_DIR / "gpu.lock"
RUNPOD_SUPERVISOR_PATH = RUNTIME_DIR / "runpod-supervisor.json"
RUNPOD_PROCESSES_PATH = RUNTIME_DIR / "runpod-processes.json"
RUNPOD_EVENTS_PATH = RUNTIME_DIR / "runpod-events.jsonl"
TEMPLATES_DIR = BASE_DIR / "dashboard" / "templates"
STATIC_DIR = BASE_DIR / "dashboard" / "static"

CLAIM_TTL_SECONDS = 30 * 60
GPU_LOCK_STALE_SECONDS = 2 * 60 * 60
GPU_HEARTBEAT_STALE_SECONDS = 2 * 60
AGENT_STALE_SECONDS = 2 * 60
TEXT_PREVIEW_LIMIT = 16_384
TIMELINE_POINT_LIMIT = 2_000
RECENT_EXPERIMENT_LIMIT = 100
RECENT_AGENT_LIMIT = 24

for path in [SHARED_DIR, CLAIMS_DIR, LOCKS_DIR, AGENTS_DIR, RUNTIME_DIR, TEMPLATES_DIR, STATIC_DIR]:
    path.mkdir(parents=True, exist_ok=True)
for path in [ACCOUNTS_DIR, ACCOUNT_LEASES_DIR]:
    path.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Autoresearch MNIST Swarm Dashboard",
    description="Read-only dashboard for the local Codex MNIST swarm.",
    docs_url=None,
    redoc_url=None,
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def _format_dt(value: Any) -> str:
    dt = _parse_iso(value)
    if not dt:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _format_short_dt(value: Any) -> str:
    dt = _parse_iso(value)
    if not dt:
        return "—"
    return dt.strftime("%m/%d %H:%M")


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _format_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _human_seconds(value: Any) -> str:
    if value is None:
        return "—"
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return "—"
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _lower_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_auth_status(value: Any) -> str:
    raw = _lower_text(value)
    if raw in {"healthy", "ok", "good", "ready", "active", "available"}:
        return "healthy"
    if raw in {"leased", "allocated", "in_use", "busy", "occupied"}:
        return "leased"
    if raw in {"suspect", "quarantine", "quarantined", "error", "invalid", "bad"}:
        return "suspect"
    if raw in {"expired", "stale", "dead", "revoked", "invalidated"}:
        return "expired"
    if raw in {"disabled", "off", "inactive"}:
        return "disabled"
    if raw in {"waiting", "pending", "blocked"}:
        return "waiting"
    if raw:
        return raw
    return "unknown"


def _candidate_list(payload: dict[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _candidate_dict(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _account_id(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    cleaned = str(value).strip()
    return cleaned or fallback


def _account_capacity(value: Any, default: int = 1) -> int:
    try:
        capacity = int(value)
        return capacity if capacity > 0 else default
    except (TypeError, ValueError):
        return default


def _load_account_events(limit: int = 200) -> list[dict[str, Any]]:
    rows = _load_jsonl(ACCOUNT_EVENTS_PATH)
    if limit > 0:
        rows = rows[-limit:]
    events: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_kind = _lower_text(row.get("event") or row.get("type") or row.get("status") or row.get("action"))
        severity = _lower_text(row.get("severity") or row.get("level"))
        events.append(
            {
                "event": row.get("event") or row.get("type") or row.get("action") or row.get("status"),
                "kind": event_kind or "event",
                "severity": severity or "info",
                "recorded_at": row.get("recorded_at") or row.get("timestamp") or row.get("created_at"),
                "recorded_label": _format_short_dt(row.get("recorded_at") or row.get("timestamp") or row.get("created_at")),
                "account_id": row.get("account_id") or row.get("account") or row.get("auth_account_id"),
                "agent_id": row.get("agent_id") or row.get("agent"),
                "role": row.get("role") or row.get("agent_role"),
                "message": row.get("message") or row.get("detail") or row.get("description"),
                "status": row.get("status"),
                "path": _path_rel(ACCOUNT_EVENTS_PATH, SHARED_DIR),
                "raw": row,
            }
        )
    return list(reversed(events))


def _load_account_leases() -> list[dict[str, Any]]:
    leases: list[dict[str, Any]] = []
    if not ACCOUNT_LEASES_DIR.exists():
        return leases

    now = datetime.now(timezone.utc)
    for path in sorted(ACCOUNT_LEASES_DIR.glob("*.json")):
        payload = _safe_json_load(path, default={})
        if not isinstance(payload, dict) or not payload:
            leases.append(
                {
                    "path": _path_rel(path, SHARED_DIR),
                    "status": "invalid",
                    "active": False,
                    "stale": True,
                    "account_id": path.stem,
                }
            )
            continue

        expires_at = _parse_iso(payload.get("expires_at"))
        heartbeat_at = _parse_iso(payload.get("heartbeat_at")) or _parse_iso(payload.get("claimed_at"))
        acquired_at = _parse_iso(payload.get("acquired_at")) or _parse_iso(payload.get("started_at"))
        age_seconds = max(0.0, (now - heartbeat_at).total_seconds()) if heartbeat_at else None
        remaining_seconds = max(0.0, (expires_at - now).total_seconds()) if expires_at else None
        state = _normalize_auth_status(payload.get("state") or payload.get("status"))
        terminal = {"expired", "revoked", "released", "closed", "done", "stale"}
        active = bool(payload.get("active", True)) and state not in terminal
        if expires_at and expires_at <= now:
            active = False
            state = "expired"
        if heartbeat_at is None and acquired_at is None and state in {"unknown", "waiting"}:
            active = False
        leases.append(
            {
                "lease_id": payload.get("lease_id") or path.stem,
                "account_id": _account_id(payload.get("account_id") or payload.get("account"), path.stem),
                "account_label": payload.get("account_label") or payload.get("label"),
                "agent_id": payload.get("agent_id") or payload.get("agent"),
                "role": payload.get("role") or payload.get("agent_role"),
                "status": state,
                "active": active,
                "stale": not active,
                "capacity": _account_capacity(payload.get("capacity") or payload.get("slots") or payload.get("size"), 1),
                "expires_at": payload.get("expires_at"),
                "heartbeat_at": payload.get("heartbeat_at") or payload.get("claimed_at"),
                "acquired_at": payload.get("acquired_at") or payload.get("started_at"),
                "remaining_seconds": remaining_seconds,
                "age_seconds": age_seconds,
                "hostname": payload.get("hostname"),
                "pid": payload.get("pid"),
                "message": payload.get("message") or payload.get("reason"),
                "path": _path_rel(path, SHARED_DIR),
                "raw": payload,
            }
        )

    leases.sort(key=lambda item: (not item.get("active"), item.get("account_id") or "", item.get("agent_id") or "", item.get("lease_id") or ""))
    return leases


def _load_auth_health() -> dict[str, Any]:
    payload = _safe_json_load(ACCOUNT_HEALTH_PATH, default={})
    if not isinstance(payload, dict):
        payload = {}

    raw_accounts = _candidate_list(payload, ("accounts", "items", "rows", "entries", "data"))
    raw_agent_maps = _candidate_dict(payload, ("agent_account_map", "agent_accounts", "assignments"))
    raw_agents = _candidate_list(payload, ("agents",))
    leases = _load_account_leases()
    events = _load_account_events()

    lease_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lease in leases:
        lease_groups[str(lease.get("account_id") or lease.get("lease_id") or "unknown")].append(lease)

    normalized_accounts: list[dict[str, Any]] = []
    seen_account_ids: set[str] = set()

    def add_account(account: dict[str, Any], fallback_id: str) -> None:
        account_id = _account_id(
            account.get("account_id") or account.get("id") or account.get("name") or account.get("label"),
            fallback_id,
        )
        seen_account_ids.add(account_id)
        account_leases = lease_groups.get(account_id, [])
        capacity = _account_capacity(
            account.get("capacity")
            or account.get("max_capacity")
            or account.get("max_agents")
            or account.get("slots"),
            default=max(len(account_leases), 1),
        )
        used_capacity = sum(
            _account_capacity(lease.get("capacity"), 1)
            for lease in account_leases
            if lease.get("active")
        )
        if "used_capacity" in account:
            used_capacity = _account_capacity(account.get("used_capacity"), used_capacity)
        elif "active_leases" in account:
            used_capacity = _account_capacity(account.get("active_leases"), used_capacity)
        free_capacity = max(capacity - used_capacity, 0)
        raw_status = account.get("status") or account.get("state") or account.get("health")
        status = _normalize_auth_status(raw_status)
        if status == "unknown":
            if any(lease.get("active") for lease in account_leases):
                status = "leased"
            elif free_capacity > 0:
                status = "healthy"
        agent_entries = []
        for lease in account_leases:
            if lease.get("agent_id") or lease.get("role"):
                agent_entries.append(
                    {
                        "agent_id": lease.get("agent_id"),
                        "role": lease.get("role"),
                        "lease_id": lease.get("lease_id"),
                        "status": lease.get("status"),
                        "expires_at": lease.get("expires_at"),
                        "heartbeat_at": lease.get("heartbeat_at"),
                    }
                )
        agent_entries.sort(key=lambda item: (item.get("role") or "", item.get("agent_id") or ""))
        normalized_accounts.append(
            {
                "account_id": account_id,
                "label": account.get("label") or account.get("display_name") or account_id,
                "status": status,
                "capacity": capacity,
                "used_capacity": used_capacity,
                "free_capacity": free_capacity,
                "enabled": bool(account.get("enabled", True)),
                "workspace_id": account.get("workspace_id") or account.get("workspace"),
                "provider": account.get("provider"),
                "last_verified_at": account.get("last_verified_at") or account.get("last_checked_at"),
                "last_error": account.get("last_error") or account.get("error"),
                "suspect_reason": account.get("suspect_reason") or account.get("reason"),
                "expires_at": account.get("expires_at"),
                "lease_count": len(account_leases),
                "active_lease_count": sum(1 for lease in account_leases if lease.get("active")),
                "agents": agent_entries,
                "leases": account_leases,
                "raw": account,
            }
        )

    for idx, account in enumerate(raw_accounts):
        if isinstance(account, dict):
            add_account(account, f"account-{idx + 1}")

    if not normalized_accounts and lease_groups:
        for account_id, account_leases in lease_groups.items():
            add_account(
                {
                    "account_id": account_id,
                    "label": account_id,
                    "status": "leased" if any(lease.get("active") for lease in account_leases) else "unknown",
                    "capacity": max(len(account_leases), 1),
                    "used_capacity": sum(
                        _account_capacity(lease.get("capacity"), 1) for lease in account_leases if lease.get("active")
                    ),
                },
                account_id,
            )

    agent_map: list[dict[str, Any]] = []
    if isinstance(raw_agent_maps, dict):
        for agent_id, account_ref in raw_agent_maps.items():
            if isinstance(account_ref, dict):
                agent_map.append(
                    {
                        "agent_id": agent_id,
                        "role": account_ref.get("role") or account_ref.get("agent_role"),
                        "account_id": account_ref.get("account_id") or account_ref.get("id") or account_ref.get("account"),
                        "account_label": account_ref.get("account_label") or account_ref.get("label"),
                        "status": account_ref.get("status") or account_ref.get("state"),
                        "message": account_ref.get("message") or account_ref.get("reason"),
                        "last_seen_at": account_ref.get("last_seen_at") or account_ref.get("heartbeat_at"),
                    }
                )
            else:
                agent_map.append({"agent_id": agent_id, "account_id": account_ref})
    for entry in raw_agents:
        if isinstance(entry, dict):
            agent_id = entry.get("agent_id") or entry.get("id") or entry.get("name")
            if not agent_id:
                continue
            agent_map.append(
                {
                    "agent_id": agent_id,
                    "role": entry.get("role") or entry.get("agent_role"),
                    "account_id": entry.get("account_id") or entry.get("account") or entry.get("lease_account_id"),
                    "account_label": entry.get("account_label") or entry.get("label"),
                    "status": entry.get("status") or entry.get("state"),
                    "message": entry.get("message") or entry.get("reason"),
                    "last_seen_at": entry.get("last_seen_at") or entry.get("heartbeat_at"),
                }
            )

    deduped_agent_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in agent_map:
        if not (item.get("agent_id") or item.get("role")):
            continue
        key = (str(item.get("agent_id") or ""), str(item.get("role") or ""), str(item.get("account_id") or ""))
        deduped_agent_map[key] = item
    agent_map = list(deduped_agent_map.values())
    agent_map.sort(key=lambda item: (item.get("role") or "", item.get("agent_id") or "", item.get("account_id") or ""))

    status_counts = defaultdict(int)
    capacity_total = 0
    capacity_used = 0
    capacity_free = 0
    waiting_agents = 0
    active_accounts = 0
    for account in normalized_accounts:
        status_counts[account["status"]] += 1
        capacity_total += int(account["capacity"])
        capacity_used += int(account["used_capacity"])
        capacity_free += int(account["free_capacity"])
        if account["enabled"]:
            active_accounts += 1
    for lease in leases:
        if lease.get("status") == "waiting":
            waiting_agents += 1

    auth_failures = [
        event
        for event in events
        if any(token in _lower_text(event.get("kind") or event.get("event") or event.get("status")) for token in ("fail", "error", "invalid", "deny", "expired"))
    ]
    suspect_accounts = status_counts.get("suspect", 0)
    expired_accounts = status_counts.get("expired", 0)

    alerts: list[dict[str, Any]] = []
    if waiting_agents > 0:
        alerts.append(
            {
                "level": "warn",
                "title": "Agents waiting for auth",
                "message": f"{waiting_agents} agent lease(s) are waiting for a free account slot.",
            }
        )
    if capacity_free <= 0 and active_accounts > 0:
        alerts.append(
            {
                "level": "warn",
                "title": "No free capacity",
                "message": "All enabled account capacity is currently allocated.",
            }
        )
    if suspect_accounts > 0:
        alerts.append(
            {
                "level": "danger",
                "title": "Suspect accounts",
                "message": f"{suspect_accounts} account(s) are marked suspect or invalid.",
            }
        )
    if expired_accounts > 0:
        alerts.append(
            {
                "level": "warn",
                "title": "Expired auth",
                "message": f"{expired_accounts} account(s) appear expired or stale.",
            }
        )
    if auth_failures:
        alerts.append(
            {
                "level": "warn",
                "title": "Recent auth failures",
                "message": f"{len(auth_failures)} recent auth-related event(s) were recorded.",
            }
        )
    if not normalized_accounts and not leases:
        alerts.append(
            {
                "level": "info",
                "title": "No auth inventory yet",
                "message": "Populate shared/accounts/health.json or shared/accounts/leases/*.json to track account capacity.",
            }
        )

    summary = {
        "path": _path_rel(ACCOUNT_HEALTH_PATH, SHARED_DIR),
        "generated_at": _now_iso(),
        "loaded": ACCOUNT_HEALTH_PATH.exists(),
        "counts": {
            "accounts": len(normalized_accounts),
            "healthy": status_counts.get("healthy", 0),
            "leased": status_counts.get("leased", 0),
            "suspect": suspect_accounts,
            "expired": expired_accounts,
            "disabled": status_counts.get("disabled", 0),
            "waiting_agents": waiting_agents,
            "active_leases": sum(1 for lease in leases if lease.get("active")),
            "capacity_total": capacity_total,
            "capacity_used": capacity_used,
            "capacity_free": capacity_free,
            "auth_failures": len(auth_failures),
        },
        "alerts": alerts,
        "top_line": {
            "healthy_accounts": status_counts.get("healthy", 0),
            "free_capacity": capacity_free,
            "waiting_agents": waiting_agents,
        },
    }

    return {
        "summary": summary,
        "accounts": normalized_accounts,
        "leases": leases,
        "events": events,
        "agent_map": agent_map,
        "health": payload,
    }


def _load_runpod_state() -> dict[str, Any]:
    supervisor = _safe_json_load(RUNPOD_SUPERVISOR_PATH, default={})
    if not isinstance(supervisor, dict):
        supervisor = {}
    processes_payload = _safe_json_load(RUNPOD_PROCESSES_PATH, default={"processes": []})
    if not isinstance(processes_payload, dict):
        processes_payload = {"processes": []}
    processes = processes_payload.get("processes", [])
    if not isinstance(processes, list):
        processes = []
    events = _load_jsonl(RUNPOD_EVENTS_PATH)
    if len(events) > 100:
        events = events[-100:]

    state = str(supervisor.get("state") or ("running" if supervisor else "unknown"))
    running = sum(1 for item in processes if isinstance(item, dict) and item.get("state") == "running")
    restarting = sum(1 for item in processes if isinstance(item, dict) and item.get("state") == "restarting")
    exited = sum(1 for item in processes if isinstance(item, dict) and item.get("state") == "exited")
    pod_id = supervisor.get("pod_id") or os.environ.get("RUNPOD_POD_ID")
    public_ip = supervisor.get("public_ip") or os.environ.get("RUNPOD_PUBLIC_IP")
    tcp_port_22 = supervisor.get("tcp_port_22") or os.environ.get("RUNPOD_TCP_PORT_22")
    tcp_port_8080 = supervisor.get("tcp_port_8080") or os.environ.get("RUNPOD_TCP_PORT_8080")
    ssh_endpoint = None
    if public_ip and tcp_port_22:
        ssh_endpoint = f"{public_ip}:{tcp_port_22}"

    alerts: list[dict[str, Any]] = []
    if not supervisor:
        alerts.append(
            {
                "level": "warn",
                "title": "Supervisor missing",
                "message": "No RunPod supervisor status file has been published yet.",
            }
        )
    if restarting > 0:
        alerts.append(
            {
                "level": "warn",
                "title": "Services restarting",
                "message": f"{restarting} service(s) are currently in restart backoff.",
            }
        )
    if exited > 0 and restarting == 0:
        alerts.append(
            {
                "level": "danger",
                "title": "Exited services",
                "message": f"{exited} service(s) have exited and are not currently running.",
            }
        )

    summary = {
        "loaded": bool(supervisor),
        "state": state,
        "generated_at": _now_iso(),
        "started_at": supervisor.get("started_at"),
        "last_seen_at": supervisor.get("last_seen_at"),
        "uptime_seconds": supervisor.get("uptime_seconds"),
        "restart_policy": supervisor.get("restart_policy"),
        "services_total": supervisor.get("services_total") or len(processes),
        "services_running": supervisor.get("services_running") or running,
        "services_restarting": restarting,
        "services_exited": exited,
        "workspace_dir": supervisor.get("workspace_dir"),
        "app_dir": supervisor.get("app_dir"),
        "dashboard_port": supervisor.get("dashboard_port"),
        "ssh_port": supervisor.get("ssh_port"),
        "pod_id": pod_id,
        "public_ip": public_ip,
        "tcp_port_22": tcp_port_22,
        "tcp_port_8080": tcp_port_8080,
        "ssh_endpoint": ssh_endpoint,
        "alerts": alerts,
        "links": {
            "supervisor": _browse_href("runtime/runpod-supervisor.json"),
            "processes": _browse_href("runtime/runpod-processes.json"),
            "events": _browse_href("runtime/runpod-events.jsonl"),
        },
    }

    return {
        "summary": summary,
        "processes": processes,
        "events": list(reversed(events)),
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = row.get("metrics", {})
    return (
        _to_float(metrics.get("val_errors")) or float("inf"),
        _to_float(metrics.get("val_loss")) or float("inf"),
        _to_float(metrics.get("training_seconds")) or float("inf"),
        str(row.get("recorded_at") or ""),
    )


def _experiment_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        _to_float(row.get("val_errors")) or float("inf"),
        _to_float(row.get("val_loss")) or float("inf"),
        _to_float(row.get("training_seconds")) or float("inf"),
        str(row.get("recorded_at") or ""),
    )


def _rank_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    metrics = row.get("metrics", {})
    return {
        "experiment_key": row.get("experiment_key"),
        "agent_id": row.get("agent_id"),
        "role": row.get("role"),
        "status": row.get("status"),
        "description": row.get("description"),
        "recorded_at": row.get("recorded_at"),
        "final_eval": bool(row.get("final_eval")),
        "model_family": metrics.get("model_family"),
        "run_mode": metrics.get("run_mode"),
        "val_errors": metrics.get("val_errors"),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_loss": metrics.get("val_loss"),
        "train_loss": metrics.get("train_loss"),
        "training_seconds": metrics.get("training_seconds"),
        "test_errors": metrics.get("test_errors"),
        "test_accuracy": metrics.get("test_accuracy"),
        "checkpoint_path": metrics.get("checkpoint_path"),
        "manifest_path": metrics.get("manifest_path"),
        "config_path": metrics.get("config_path"),
        "val_logits_path": metrics.get("val_logits_path"),
        "train_py_path": row.get("train_py_path"),
        "artifacts_dir": row.get("artifacts_dir"),
        "shared_artifact_dir": row.get("shared_artifact_dir"),
    }


def _compact_preview(text: str, limit: int = 120) -> str:
    limit = min(limit, TEXT_PREVIEW_LIMIT)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _path_rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def _browse_href(relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    cleaned = str(relative_path).replace("\\", "/").lstrip("/")
    return f"/browse/{quote(cleaned)}"


def _is_process_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except OSError:
        return False


def _file_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "name": path.name,
        "path": _path_rel(path, SHARED_DIR),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "modified_label": _format_dt(stat.st_mtime),
        "browse_url": _browse_href(_path_rel(path, SHARED_DIR)),
    }
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload["kind"] = "json"
            payload["content"] = json.loads(path.read_text())
            payload["summary"] = _compact_json_summary(payload["content"])
        except Exception:
            payload["kind"] = "text"
            payload["preview"] = _compact_preview(path.read_text(errors="ignore"), 280)
    elif suffix in {".jsonl", ".txt", ".md", ".py", ".yaml", ".yml", ".log"}:
        payload["kind"] = "text"
        payload["preview"] = _compact_preview(path.read_text(errors="ignore"), 280)
    else:
        payload["kind"] = "binary"
    return payload


def _compact_json_summary(value: Any) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())
        if not keys:
            return "Empty object"
        return f"{len(keys)} keys: {', '.join(map(str, keys[:5]))}"
    if isinstance(value, list):
        return f"{len(value)} items"
    return type(value).__name__


def _claim_payload(path: Path) -> dict[str, Any]:
    payload = _safe_json_load(path, default={})
    if not isinstance(payload, dict) or not payload:
        return {
            "path": _path_rel(path, SHARED_DIR),
            "status": "invalid",
            "active": False,
        }

    heartbeat_at = _parse_iso(payload.get("heartbeat_at")) or _parse_iso(payload.get("claimed_at"))
    claimed_at = _parse_iso(payload.get("claimed_at"))
    age_seconds = None
    if heartbeat_at:
        age_seconds = max(0.0, (datetime.now(timezone.utc) - heartbeat_at).total_seconds())
    ttl_seconds = payload.get("ttl_seconds", CLAIM_TTL_SECONDS)
    stale = False
    if age_seconds is None:
        stale = True
    else:
        stale = age_seconds > float(ttl_seconds)
        if not stale and payload.get("hostname") == socket.gethostname():
            stale = not _is_process_alive(payload.get("pid"))

    return {
        "experiment_key": payload.get("experiment_key"),
        "description": payload.get("description"),
        "normalized_description": payload.get("normalized_description"),
        "agent_id": payload.get("agent_id"),
        "role": payload.get("role"),
        "pid": payload.get("pid"),
        "hostname": payload.get("hostname"),
        "claimed_at": payload.get("claimed_at"),
        "heartbeat_at": payload.get("heartbeat_at"),
        "age_seconds": age_seconds,
        "ttl_seconds": ttl_seconds,
        "expires_in_seconds": None if age_seconds is None else max(0.0, float(ttl_seconds) - age_seconds),
        "stale": stale,
        "active": not stale,
        "path": _path_rel(path, SHARED_DIR),
    }


def _gpu_lock_payload() -> dict[str, Any]:
    payload = _safe_json_load(GPU_LOCK_PATH, default={})
    if not isinstance(payload, dict) or not payload:
        return {"locked": False, "stale": False, "path": _path_rel(GPU_LOCK_PATH, SHARED_DIR)}

    acquired_at = _parse_iso(payload.get("acquired_at"))
    heartbeat_at = _parse_iso(payload.get("heartbeat_at")) or acquired_at
    age_seconds = None
    heartbeat_age_seconds = None
    if acquired_at:
        age_seconds = max(0.0, (datetime.now(timezone.utc) - acquired_at).total_seconds())
    if heartbeat_at:
        heartbeat_age_seconds = max(0.0, (datetime.now(timezone.utc) - heartbeat_at).total_seconds())
    stale = False
    if age_seconds is None or heartbeat_age_seconds is None:
        stale = True
    else:
        stale = heartbeat_age_seconds > GPU_HEARTBEAT_STALE_SECONDS or age_seconds > GPU_LOCK_STALE_SECONDS
        if not stale and payload.get("hostname") == socket.gethostname():
            stale = not _is_process_alive(payload.get("pid"))

    return {
        "locked": True,
        "stale": stale,
        "agent_id": payload.get("agent_id"),
        "role": payload.get("role"),
        "pid": payload.get("pid"),
        "hostname": payload.get("hostname"),
        "acquired_at": payload.get("acquired_at"),
        "heartbeat_at": payload.get("heartbeat_at"),
        "age_seconds": age_seconds,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "path": _path_rel(GPU_LOCK_PATH, SHARED_DIR),
    }


def _load_best_results() -> dict[str, Any]:
    payload = _safe_json_load(
        BEST_RESULTS_PATH,
        default={"global_best": None, "by_role": {}, "by_family": {}, "updated_at": None},
    )
    if not isinstance(payload, dict):
        payload = {"global_best": None, "by_role": {}, "by_family": {}, "updated_at": None}
    payload.setdefault("global_best", None)
    payload.setdefault("by_role", {})
    payload.setdefault("by_family", {})
    payload.setdefault("updated_at", None)
    return payload


def _load_agent_manifests() -> list[dict[str, Any]]:
    if not AGENTS_DIR.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for path in sorted(AGENTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        payload = _file_payload(path)
        payload["source"] = "shared/agents"
        try:
            data = json.loads(path.read_text())
            payload["kind"] = "json"
            payload["content"] = data
            payload["agent_id"] = data.get("agent_id") or data.get("id") or path.stem
            payload["role"] = data.get("role") or data.get("agent_role") or data.get("name")
            payload["display_name"] = data.get("display_name") or data.get("name") or payload["agent_id"]
            payload["model"] = data.get("model") or data.get("codex_model")
            payload["summary"] = _compact_json_summary(data)
        except Exception:
            payload["agent_id"] = path.stem
            payload["display_name"] = path.stem
        manifests.append(payload)
    return manifests


def _collect_claims() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    if not CLAIMS_DIR.exists():
        return active, stale
    for path in sorted(CLAIMS_DIR.glob("*.json")):
        payload = _claim_payload(path)
        if payload.get("active"):
            active.append(payload)
        else:
            stale.append(payload)
    active.sort(key=lambda item: (item.get("role") or "", item.get("claimed_at") or ""))
    stale.sort(key=lambda item: (item.get("role") or "", item.get("claimed_at") or ""))
    return active, stale


def _collect_experiments() -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(_load_jsonl(EXPERIMENT_LOG_PATH)):
        metrics = row.get("metrics", {})
        record = {
            "index": idx,
            "experiment_key": row.get("experiment_key"),
            "agent_id": row.get("agent_id"),
            "role": row.get("role"),
            "status": row.get("status"),
            "description": row.get("description"),
            "recorded_at": row.get("recorded_at"),
            "recorded_label": _format_short_dt(row.get("recorded_at")),
            "final_eval": bool(row.get("final_eval")),
            "model_family": metrics.get("model_family"),
            "run_mode": metrics.get("run_mode"),
            "val_errors": metrics.get("val_errors"),
            "val_accuracy": metrics.get("val_accuracy"),
            "val_loss": metrics.get("val_loss"),
            "train_loss": metrics.get("train_loss"),
            "training_seconds": metrics.get("training_seconds"),
            "test_errors": metrics.get("test_errors"),
            "test_accuracy": metrics.get("test_accuracy"),
            "checkpoint_path": metrics.get("checkpoint_path"),
            "manifest_path": metrics.get("manifest_path"),
            "config_path": metrics.get("config_path"),
            "val_logits_path": metrics.get("val_logits_path"),
            "train_py_path": row.get("train_py_path"),
            "artifacts_dir": row.get("artifacts_dir"),
            "shared_artifact_dir": row.get("shared_artifact_dir"),
            "record_path": _browse_href(f"results/{row.get('experiment_key')}.json"),
            "snapshot_path": _browse_href(f"snapshots/{row.get('experiment_key')}/train.py"),
            "artifact_path": _browse_href(f"best_checkpoints/{row.get('experiment_key')}") if row.get("shared_artifact_dir") else None,
            "source": "experiment_log",
        }
        rows.append(record)
    rows.sort(key=lambda item: (_parse_iso(item.get("recorded_at")) or datetime.min.replace(tzinfo=timezone.utc), item["index"]))
    return rows


def _best_prefix_series(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    if len(experiments) > TIMELINE_POINT_LIMIT:
        experiments = experiments[-TIMELINE_POINT_LIMIT:]
    labels: list[str] = []
    best_val_errors: list[float | None] = []
    best_val_loss: list[float | None] = []
    best_roles: list[str | None] = []
    best_families: list[str | None] = []
    best_records: list[dict[str, Any] | None] = []

    best_row: dict[str, Any] | None = None
    for row in experiments:
        if row.get("status") == "keep":
            if best_row is None or _experiment_sort_key(row) < _experiment_sort_key(best_row):
                best_row = row
        labels.append(row.get("recorded_label") or row.get("experiment_key") or f"#{row['index']}")
        best_val_errors.append(_to_float(best_row.get("val_errors")) if best_row else None)
        best_val_loss.append(_to_float(best_row.get("val_loss")) if best_row else None)
        best_roles.append(best_row.get("role") if best_row else None)
        best_families.append(best_row.get("model_family") if best_row else None)
        best_records.append(best_row)

    return {
        "labels": labels,
        "best_val_errors": best_val_errors,
        "best_val_loss": best_val_loss,
        "best_roles": best_roles,
        "best_families": best_families,
        "best_records": [row.get("experiment_key") if row else None for row in best_records],
    }


def _build_agent_roster(
    manifests: list[dict[str, Any]],
    claims_active: list[dict[str, Any]],
    claims_stale: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    bests: dict[str, Any],
    account_map: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    roster: dict[str, dict[str, Any]] = {}
    account_lookup: dict[str, dict[str, Any]] = {}
    for entry in account_map or []:
        for key in (entry.get("agent_id"), entry.get("role")):
            if key:
                account_lookup[str(key)] = entry

    def map_status(raw_state: Any, stale: bool) -> str:
        if stale:
            return "stale-claim"
        state = str(raw_state or "").strip().lower()
        if state in {"running", "waiting_for_gpu", "planning"}:
            return "working"
        if state in {"crash", "crashed"}:
            return "crashed"
        if state in {"discard", "review"}:
            return "review"
        if state in {"idle", "ready"}:
            return "idle"
        if state in {"stopped", "offline"}:
            return "offline"
        if state:
            return "recent"
        return "offline"

    def get_entry(agent_id: str | None, role: str | None = None) -> dict[str, Any]:
        key = agent_id or role or "unknown"
        if key not in roster:
            roster[key] = {
                "agent_id": agent_id or key,
                "role": role,
                "display_name": agent_id or role or key,
                "source": [],
                "manifest_files": [],
                "log_count": 0,
                "latest_run": None,
                "best_run": None,
                "active_claim": None,
                "stale_claim": None,
                "status": "offline",
                "last_seen": None,
                "model_family": None,
                "run_mode": None,
                "account_id": None,
                "account_label": None,
                "account_status": None,
                "account_lease_state": None,
                "account_free_capacity": None,
                "account_capacity": None,
                "summary": None,
            }
        return roster[key]

    for manifest in manifests:
        agent_id = manifest.get("agent_id")
        role = manifest.get("role")
        entry = get_entry(agent_id, role)
        entry["display_name"] = manifest.get("display_name") or entry["display_name"]
        if manifest.get("model"):
            entry["model_family"] = manifest.get("model")
        entry["manifest_files"].append(
            {
                "path": manifest.get("path"),
                "browse_url": manifest.get("browse_url"),
                "kind": manifest.get("kind"),
                "summary": manifest.get("summary"),
            }
        )
        entry["source"].append("shared/agents")
        if isinstance(manifest.get("content"), dict):
            data = manifest.get("content")
            entry["summary"] = data
            extra = data.get("extra", {}) if isinstance(data.get("extra"), dict) else {}
            last_seen_dt = _parse_iso(data.get("last_seen_at"))
            if last_seen_dt:
                entry["last_seen"] = data.get("last_seen_at")
            if extra.get("model_family"):
                entry["model_family"] = extra.get("model_family")
            last_result = extra.get("last_result", {}) if isinstance(extra.get("last_result"), dict) else {}
            if last_result.get("model_family") and not entry.get("model_family"):
                entry["model_family"] = last_result.get("model_family")
            if last_result.get("run_mode"):
                entry["run_mode"] = last_result.get("run_mode")
            stale = True
            if last_seen_dt:
                stale = (datetime.now(timezone.utc) - last_seen_dt).total_seconds() > AGENT_STALE_SECONDS
            entry["status"] = map_status(data.get("state"), stale)

    for row in experiments:
        entry = get_entry(row.get("agent_id"), row.get("role"))
        entry["log_count"] += 1
        entry["source"].append("experiment_log")
        if row.get("recorded_at"):
            if not entry["last_seen"] or _parse_iso(row.get("recorded_at")) > _parse_iso(entry["last_seen"]):
                entry["last_seen"] = row.get("recorded_at")
                entry["latest_run"] = row
                entry["model_family"] = row.get("model_family") or entry["model_family"]
                entry["run_mode"] = row.get("run_mode") or entry["run_mode"]
        if row.get("status") == "keep":
            best_for_agent = entry.get("best_run")
            if best_for_agent is None or _experiment_sort_key(row) < _experiment_sort_key(best_for_agent):
                entry["best_run"] = row

    claim_lookup: dict[str, dict[str, Any]] = {}
    for claim in claims_active:
        claim_lookup[str(claim.get("agent_id") or claim.get("role") or claim.get("experiment_key"))] = claim
    stale_lookup: dict[str, dict[str, Any]] = {}
    for claim in claims_stale:
        stale_lookup[str(claim.get("agent_id") or claim.get("role") or claim.get("experiment_key"))] = claim

    for key, entry in roster.items():
        claim = claim_lookup.get(str(entry.get("agent_id"))) or claim_lookup.get(str(entry.get("role")))
        stale_claim = stale_lookup.get(str(entry.get("agent_id"))) or stale_lookup.get(str(entry.get("role")))
        account_entry = account_lookup.get(str(entry.get("agent_id"))) or account_lookup.get(str(entry.get("role")))
        if account_entry:
            entry["account_id"] = account_entry.get("account_id")
            entry["account_label"] = account_entry.get("account_label") or account_entry.get("account_id")
            entry["account_status"] = account_entry.get("status")
            entry["account_lease_state"] = account_entry.get("status")
            entry["account_free_capacity"] = account_entry.get("free_capacity")
            entry["account_capacity"] = account_entry.get("capacity")
        if claim:
            entry["active_claim"] = claim
            entry["source"].append("claims")
        if stale_claim:
            entry["stale_claim"] = stale_claim
            entry["source"].append("claims")

        latest = entry.get("latest_run")
        active = bool(claim)
        if active:
            entry["status"] = "working"
        elif latest and latest.get("status") == "crash":
            entry["status"] = "crashed"
        elif latest and latest.get("status") == "discard":
            entry["status"] = "review"
        elif latest and latest.get("status") == "keep":
            entry["status"] = "idle"
        elif latest:
            entry["status"] = "recent"
        else:
            entry["status"] = "offline"
        if isinstance(entry.get("summary"), dict) and entry.get("summary", {}).get("state"):
            summary = entry["summary"]
            last_seen = _parse_iso(summary.get("last_seen_at"))
            stale = not last_seen or (datetime.now(timezone.utc) - last_seen).total_seconds() > AGENT_STALE_SECONDS
            entry["status"] = map_status(summary.get("state"), stale)
        if stale_claim and not claim and entry["status"] == "offline":
            entry["status"] = "stale-claim"

        if latest and latest.get("recorded_at"):
            entry["last_seen"] = latest.get("recorded_at")

        best_by_role = bests.get("by_role", {}).get(entry.get("role"))
        if best_by_role and not entry.get("best_run"):
            entry["best_run"] = _rank_row(best_by_role)
        if best_by_role and not entry.get("model_family"):
            entry["model_family"] = best_by_role.get("metrics", {}).get("model_family")

        entry["source"] = sorted(set(entry["source"]))

    return sorted(
        roster.values(),
        key=lambda item: (
            item.get("role") or "",
            item.get("agent_id") or "",
        ),
    )


def _build_leaderboard(bests: dict[str, Any], experiments: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = []
    for role, row in sorted(bests.get("by_role", {}).items()):
        by_role.append({"role": role, **_rank_row(row)})

    by_family = []
    for family, row in sorted(bests.get("by_family", {}).items()):
        by_family.append({"family": family, **_rank_row(row)})

    top_keeps = sorted((row for row in experiments if row.get("status") == "keep"), key=_experiment_sort_key)[:RECENT_AGENT_LIMIT]
    top_keeps = [_rank_row(row) for row in top_keeps]
    return {
        "global_best": _rank_row(bests.get("global_best")),
        "by_role": by_role,
        "by_family": by_family,
        "top_keeps": top_keeps,
    }


def _build_summary() -> dict[str, Any]:
    bests = _load_best_results()
    experiments_timeline = _collect_experiments()
    active_claims, stale_claims = _collect_claims()
    gpu_lock = _gpu_lock_payload()
    manifests = _load_agent_manifests()
    auth_state = _load_auth_health()
    runpod_state = _load_runpod_state()
    leaderboard = _build_leaderboard(bests, experiments_timeline)
    chart = _best_prefix_series(experiments_timeline)
    agents = _build_agent_roster(
        manifests,
        active_claims,
        stale_claims,
        experiments_timeline,
        bests,
        auth_state.get("agent_map", []),
    )
    recent_experiments = list(reversed(experiments_timeline))[:RECENT_EXPERIMENT_LIMIT]

    counts = {
        "experiments": len(experiments_timeline),
        "keeps": sum(1 for row in experiments_timeline if row.get("status") == "keep"),
        "discards": sum(1 for row in experiments_timeline if row.get("status") == "discard"),
        "crashes": sum(1 for row in experiments_timeline if row.get("status") == "crash"),
        "active_claims": len(active_claims),
        "stale_claims": len(stale_claims),
        "agents": len(agents),
        "manifest_files": len(manifests),
        "final_evals": sum(1 for row in experiments_timeline if row.get("final_eval")),
        "accounts": auth_state["summary"]["counts"]["accounts"],
        "auth_healthy": auth_state["summary"]["counts"]["healthy"],
        "auth_leased": auth_state["summary"]["counts"]["leased"],
        "auth_suspect": auth_state["summary"]["counts"]["suspect"],
        "auth_expired": auth_state["summary"]["counts"]["expired"],
        "auth_waiting": auth_state["summary"]["counts"]["waiting_agents"],
        "auth_capacity_free": auth_state["summary"]["counts"]["capacity_free"],
        "auth_capacity_total": auth_state["summary"]["counts"]["capacity_total"],
        "runpod_services_total": runpod_state["summary"].get("services_total", 0),
        "runpod_services_running": runpod_state["summary"].get("services_running", 0),
        "runpod_services_restarting": runpod_state["summary"].get("services_restarting", 0),
        "runpod_services_exited": runpod_state["summary"].get("services_exited", 0),
    }

    latest_keep = next((row for row in reversed(experiments_timeline) if row.get("status") == "keep"), None)
    latest_failure = next((row for row in reversed(experiments_timeline) if row.get("status") in {"discard", "crash"}), None)
    global_best = bests.get("global_best")
    global_best_metrics = (global_best or {}).get("metrics", {})

    summary = {
        "generated_at": _now_iso(),
        "shared_dir": str(SHARED_DIR),
        "counts": counts,
        "best_results_path": _path_rel(BEST_RESULTS_PATH, SHARED_DIR),
        "experiment_log_path": _path_rel(EXPERIMENT_LOG_PATH, SHARED_DIR),
        "gpu_lock_path": _path_rel(GPU_LOCK_PATH, SHARED_DIR),
        "global_best": _rank_row(global_best),
        "global_best_error": _to_float(global_best_metrics.get("val_errors")),
        "global_best_loss": _to_float(global_best_metrics.get("val_loss")),
        "global_best_accuracy": _to_float(global_best_metrics.get("val_accuracy")),
        "global_best_test_errors": _to_float(global_best_metrics.get("test_errors")),
        "global_best_test_accuracy": _to_float(global_best_metrics.get("test_accuracy")),
        "best_by_role_count": len(bests.get("by_role", {})),
        "best_by_family_count": len(bests.get("by_family", {})),
        "latest_keep": _rank_row(latest_keep),
        "latest_failure": _rank_row(latest_failure),
        "active_claims": active_claims,
        "stale_claims": stale_claims,
        "auth_health": auth_state["summary"],
        "auth_accounts": auth_state["accounts"],
        "auth_leases": auth_state["leases"],
        "auth_events": auth_state["events"],
        "runpod": runpod_state["summary"],
        "gpu_lock": gpu_lock,
        "top_line": {
            "best_role": (global_best or {}).get("role"),
            "best_family": global_best_metrics.get("model_family"),
            "run_mode": global_best_metrics.get("run_mode"),
            "auth_free_capacity": auth_state["summary"]["counts"]["capacity_free"],
            "auth_waiting_agents": auth_state["summary"]["counts"]["waiting_agents"],
            "runpod_state": runpod_state["summary"].get("state"),
            "runpod_pod_id": runpod_state["summary"].get("pod_id"),
        },
        "links": {
            "experiment_log": _browse_href("experiment_log.jsonl"),
            "best_results": _browse_href("best_results.json"),
            "claims": _browse_href("claims"),
            "locks": _browse_href("locks"),
            "agents": _browse_href("agents"),
            "accounts": _browse_href("accounts"),
            "account_health": _browse_href("accounts/health.json"),
            "account_leases": _browse_href("accounts/leases"),
            "account_events": _browse_href("accounts/events.jsonl"),
            "runtime": _browse_href("runtime"),
            "runpod_supervisor": _browse_href("runtime/runpod-supervisor.json"),
            "runpod_processes": _browse_href("runtime/runpod-processes.json"),
            "runpod_events": _browse_href("runtime/runpod-events.jsonl"),
        },
        "updated_at": bests.get("updated_at"),
    }

    return {
        "summary": summary,
        "leaderboard": leaderboard,
        "experiments": recent_experiments,
        "agents": agents,
        "chart": chart,
        "best_results": bests,
        "auth": auth_state,
        "runpod": runpod_state,
    }


def _dashboard_payload() -> dict[str, Any]:
    state = _build_summary()
    summary = state["summary"]
    experiments = state["experiments"]
    leaderboard = state["leaderboard"]
    chart = state["chart"]
    agents = state["agents"]
    auth = state["auth"]
    runpod = state["runpod"]

    latest_rows = experiments[:10]
    active_claims = summary.get("active_claims", [])
    stale_claims = summary.get("stale_claims", [])

    payload = {
        "summary": summary,
        "leaderboard": leaderboard,
        "experiments": experiments,
        "agents": agents,
        "chart": chart,
        "latest_rows": latest_rows,
        "active_claims": active_claims,
        "stale_claims": stale_claims,
        "auth": auth,
        "runpod": runpod,
    }
    return payload


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    payload = _dashboard_payload()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "state": payload,
            "generated_at": payload["summary"]["generated_at"],
        },
    )


@app.get("/api/summary")
def api_summary() -> JSONResponse:
    return JSONResponse(_dashboard_payload()["summary"])


@app.get("/api/leaderboard")
def api_leaderboard() -> JSONResponse:
    return JSONResponse(_dashboard_payload()["leaderboard"])


@app.get("/api/experiments")
def api_experiments() -> JSONResponse:
    payload = _dashboard_payload()
    return JSONResponse(
        {
            "experiments": payload["experiments"],
            "counts": payload["summary"]["counts"],
            "summary": payload["summary"],
        }
    )


@app.get("/api/agents")
def api_agents() -> JSONResponse:
    payload = _dashboard_payload()
    return JSONResponse(
        {
            "agents": payload["agents"],
            "active_claims": payload["active_claims"],
            "stale_claims": payload["stale_claims"],
            "summary": payload["summary"],
        }
    )


@app.get("/api/accounts")
def api_accounts() -> JSONResponse:
    payload = _dashboard_payload()
    return JSONResponse(
        {
            "accounts": payload["auth"]["accounts"],
            "leases": payload["auth"]["leases"],
            "events": payload["auth"]["events"],
            "health": payload["auth"]["summary"],
            "summary": payload["summary"],
        }
    )


@app.get("/api/leases")
def api_leases() -> JSONResponse:
    payload = _dashboard_payload()
    return JSONResponse(
        {
            "leases": payload["auth"]["leases"],
            "summary": payload["auth"]["summary"],
            "agent_map": payload["auth"]["agent_map"],
        }
    )


@app.get("/api/health/auth")
def api_health_auth() -> JSONResponse:
    payload = _dashboard_payload()
    return JSONResponse(
        {
            "health": payload["auth"]["summary"],
            "accounts": payload["auth"]["accounts"],
            "alerts": payload["auth"]["summary"]["alerts"],
            "events": payload["auth"]["events"][:50],
            "summary": payload["summary"],
        }
    )


@app.get("/api/runpod")
def api_runpod() -> JSONResponse:
    payload = _dashboard_payload()
    return JSONResponse(
        {
            "runpod": payload["runpod"],
            "summary": payload["summary"],
        }
    )


@app.get("/api/charts/best")
def api_charts_best() -> JSONResponse:
    return JSONResponse(_dashboard_payload()["chart"])


def _safe_shared_target(requested_path: str) -> Path:
    cleaned = requested_path.strip().lstrip("/")
    target = (SHARED_DIR / cleaned).resolve()
    try:
        target.relative_to(SHARED_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc
    return target


def _directory_listing(target: Path) -> HTMLResponse:
    entries = []
    if target != SHARED_DIR:
        entries.append(
            {
                "name": "..",
                "href": _browse_href(_path_rel(target.parent, SHARED_DIR)),
                "kind": "parent",
                "size": "",
                "modified": "",
            }
        )

    for child in sorted(target.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
        stat = child.stat()
        rel = _path_rel(child, SHARED_DIR)
        entries.append(
            {
                "name": child.name + ("/" if child.is_dir() else ""),
                "href": _browse_href(rel),
                "kind": "dir" if child.is_dir() else child.suffix.lstrip(".") or "file",
                "size": f"{stat.st_size:,} B" if child.is_file() else "",
                "modified": _format_dt(stat.st_mtime),
            }
        )

    rows = "\n".join(
        f"""
        <tr>
          <td><a href="{html.escape(entry['href'] or '#')}" class="file-link">{html.escape(entry['name'])}</a></td>
          <td>{html.escape(entry['kind'])}</td>
          <td>{html.escape(entry['size'])}</td>
          <td>{html.escape(entry['modified'])}</td>
        </tr>
        """
        for entry in entries
    )

    body = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Shared Browser</title>
        <style>
          body {{
            margin: 0;
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f172a;
            color: #e2e8f0;
          }}
          .wrap {{
            max-width: 1080px;
            margin: 0 auto;
            padding: 32px;
          }}
          .panel {{
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 20px 60px rgba(2, 6, 23, 0.35);
          }}
          a {{
            color: #86efac;
            text-decoration: none;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
          }}
          th, td {{
            padding: 12px 10px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            text-align: left;
            vertical-align: top;
          }}
          th {{
            color: #94a3b8;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .08em;
          }}
          .crumb {{
            margin: 0 0 18px;
            color: #94a3b8;
          }}
          pre {{
            white-space: pre-wrap;
            word-break: break-word;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 18px;
            overflow: auto;
          }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="panel">
            <p class="crumb">shared/{html.escape(_path_rel(target, SHARED_DIR)) or "."}</p>
            <table>
              <thead>
                <tr><th>Name</th><th>Type</th><th>Size</th><th>Modified</th></tr>
              </thead>
              <tbody>{rows or "<tr><td colspan='4'>Empty directory</td></tr>"}</tbody>
            </table>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(body)


@app.get("/browse/{requested_path:path}")
def browse(requested_path: str):
    target = _safe_shared_target(requested_path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if target.is_dir():
        return _directory_listing(target)
    return FileResponse(target)
