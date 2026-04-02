"""
Local filesystem coordinator for collaborative autoresearch on MNIST.

This replaces the original Ensue-backed coordinator with a shared-directory
implementation so multiple Codex agents can collaborate on a single machine.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import fcntl

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SHARED_DIR_ENV = "AUTORESEARCH_SHARED_DIR"
DEFAULT_WORKSPACE_ENV = "AUTORESEARCH_WORKSPACE_ROOT"
DEFAULT_AGENT_ENV = "AUTORESEARCH_AGENT_ID"
DEFAULT_ROLE_ENV = "AUTORESEARCH_ROLE"

CLAIM_TTL_SECONDS = 30 * 60
GPU_LOCK_STALE_SECONDS = 2 * 60 * 60
GPU_HEARTBEAT_STALE_SECONDS = 2 * 60
AGENT_HEARTBEAT_SECONDS = 15
AGENT_STALE_SECONDS = 2 * 60
SYNC_EVERY_N = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    slug = slug.strip("-")
    return slug[:max_len].rstrip("-") or "exp"


def _normalize_description(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _description_hash(role: str, description: str) -> str:
    base = f"{role.lower()}::{_normalize_description(description)}"
    return hashlib.sha256(base.encode()).hexdigest()[:10]


def _git_output(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _git_branch() -> Optional[str]:
    return _git_output("git", "branch", "--show-current")


def _git_commit_short() -> Optional[str]:
    return _git_output("git", "rev-parse", "--short", "HEAD")


def _git_remote_url() -> Optional[str]:
    url = _git_output("git", "remote", "get-url", "origin")
    if not url:
        return None
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:") :]
    if url.endswith(".git"):
        url = url[:-4]
    return url


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _timestamp_age_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        return time.time() - datetime.fromisoformat(timestamp).timestamp()
    except ValueError:
        return None


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _record_sort_key(record: dict[str, Any]) -> tuple[float, float, float]:
    metrics = record.get("metrics", {})
    return (
        float(metrics.get("val_errors", float("inf"))),
        float(metrics.get("val_loss", float("inf"))),
        float(metrics.get("training_seconds", float("inf"))),
    )


def _is_better(candidate: dict[str, Any], incumbent: Optional[dict[str, Any]]) -> bool:
    if incumbent is None:
        return True
    return _record_sort_key(candidate) < _record_sort_key(incumbent)


@contextlib.contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class LocalCoordinator:
    def __init__(
        self,
        shared_dir: str | os.PathLike[str] | None = None,
        workspace_root: str | os.PathLike[str] | None = None,
        agent_id: str | None = None,
        role: str | None = None,
    ) -> None:
        workspace_default = Path(os.environ.get(DEFAULT_WORKSPACE_ENV, str(Path.cwd())))
        self.workspace_root = Path(workspace_root or workspace_default).resolve()
        shared_default = Path(
            os.environ.get(DEFAULT_SHARED_DIR_ENV, str(self.workspace_root / "shared"))
        )
        self.shared_dir = Path(shared_dir or shared_default).resolve()
        self.agent_id = agent_id or os.environ.get(DEFAULT_AGENT_ENV, "local-agent")
        self.role = role or os.environ.get(DEFAULT_ROLE_ENV, "solo")
        self.experiment_count = 0

        self.claims_dir = self.shared_dir / "claims"
        self.locks_dir = self.shared_dir / "locks"
        self.agents_dir = self.shared_dir / "agents"
        self.snapshots_dir = self.shared_dir / "snapshots"
        self.results_dir = self.shared_dir / "results"
        self.best_checkpoints_dir = self.shared_dir / "best_checkpoints"
        self.best_results_path = self.shared_dir / "best_results.json"
        self.experiment_log_path = self.shared_dir / "experiment_log.jsonl"
        self.insights_path = self.shared_dir / "insights.jsonl"
        self.hypotheses_path = self.shared_dir / "hypotheses.jsonl"
        self.state_lock_path = self.locks_dir / "state.lock"
        self.gpu_lock_path = self.locks_dir / "gpu.lock"

        for path in [
            self.shared_dir,
            self.claims_dir,
            self.locks_dir,
            self.agents_dir,
            self.snapshots_dir,
            self.results_dir,
            self.best_checkpoints_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.agent_status_path = self.agents_dir / (
            f"{_slugify(self.role, max_len=12)}--{_slugify(self.agent_id, max_len=32)}.json"
        )
        self._status_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._closed = False
        self._status_state = "idle"
        self._status_message = "ready"
        self._status_since = _now_iso()
        self._status_extra: dict[str, Any] = {}
        self._owned_claims: set[str] = set()
        self._holding_gpu_lease = False

        self._write_agent_status()
        self._start_heartbeat()
        atexit.register(self.close)

    @property
    def connected(self) -> bool:
        return True

    def _log(self, message: str) -> None:
        print(f"[{self.role}:{self.agent_id}] {message}")

    def _make_key(self, description: str) -> str:
        slug = _slugify(description)
        digest = _description_hash(self.role, description)
        role_slug = _slugify(self.role, max_len=12)
        return f"{role_slug}--{slug}--{digest}"

    def _claim_path(self, experiment_key: str) -> Path:
        return self.claims_dir / f"{experiment_key}.json"

    def _load_best_results(self) -> dict[str, Any]:
        return _read_json(
            self.best_results_path,
            default={"global_best": None, "by_role": {}, "by_family": {}, "updated_at": None},
        )

    def _save_best_results(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        _write_json(self.best_results_path, payload)

    def announce(self) -> None:
        analysis = self.analyze_swarm()
        print(analysis["summary"])

    def _status_payload(self) -> dict[str, Any]:
        with self._status_lock:
            payload = {
                "agent_id": self.agent_id,
                "role": self.role,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "workspace_root": str(self.workspace_root),
                "shared_dir": str(self.shared_dir),
                "state": self._status_state,
                "message": self._status_message,
                "state_since": self._status_since,
                "last_seen_at": _now_iso(),
                "claims_count": len(self._owned_claims),
                "claim_keys": sorted(self._owned_claims),
                "gpu_lease_held": self._holding_gpu_lease,
                "extra": _json_safe(dict(self._status_extra)),
            }
        return payload

    def _write_agent_status(self) -> dict[str, Any]:
        payload = self._status_payload()
        _write_json(self.agent_status_path, payload)
        return payload

    def touch_agent_status(
        self,
        state: str | None = None,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
        *,
        replace_extra: bool = False,
        drop_extra_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._status_lock:
            if state and state != self._status_state:
                self._status_state = state
                self._status_since = _now_iso()
            if message is not None:
                self._status_message = message
            if replace_extra:
                self._status_extra = {}
            for key in drop_extra_keys or []:
                self._status_extra.pop(key, None)
            if extra:
                self._status_extra.update(_json_safe(extra))
        return self._write_agent_status()

    def _refresh_claim_heartbeats(self) -> None:
        now = _now_iso()
        with self._status_lock:
            owned = list(self._owned_claims)
        if not owned:
            return
        with _locked_file(self.state_lock_path):
            for experiment_key in owned:
                claim_path = self._claim_path(experiment_key)
                payload = _read_json(claim_path, default={})
                if payload.get("agent_id") != self.agent_id:
                    with self._status_lock:
                        self._owned_claims.discard(experiment_key)
                    continue
                payload["heartbeat_at"] = now
                payload["pid"] = os.getpid()
                payload["hostname"] = socket.gethostname()
                _write_json(claim_path, payload)

    def _refresh_gpu_heartbeat(self) -> None:
        with self._status_lock:
            holding_gpu_lease = self._holding_gpu_lease
        if not holding_gpu_lease:
            return
        with _locked_file(self.state_lock_path):
            payload = _read_json(self.gpu_lock_path, default={})
            if payload.get("agent_id") != self.agent_id or payload.get("role") != self.role:
                with self._status_lock:
                    self._holding_gpu_lease = False
                return
            payload["heartbeat_at"] = _now_iso()
            payload["pid"] = os.getpid()
            payload["hostname"] = socket.gethostname()
            _write_json(self.gpu_lock_path, payload)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(AGENT_HEARTBEAT_SECONDS):
            try:
                self._refresh_claim_heartbeats()
                self._refresh_gpu_heartbeat()
                self._write_agent_status()
            except Exception:
                # Heartbeats must never crash the main experiment loop.
                pass

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"coord-heartbeat-{self.agent_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread and thread.is_alive() and thread.ident != threading.get_ident():
            thread.join(timeout=1.0)
        self.touch_agent_status(
            state="stopped",
            message="process exited",
            extra={"stopped_at": _now_iso()},
        )

    # ------------------------------------------------------------------
    # Claiming
    # ------------------------------------------------------------------

    def _claim_payload(self, experiment_key: str, description: str) -> dict[str, Any]:
        now = _now_iso()
        return {
            "experiment_key": experiment_key,
            "description": description,
            "normalized_description": _normalize_description(description),
            "agent_id": self.agent_id,
            "role": self.role,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "claimed_at": now,
            "heartbeat_at": now,
            "ttl_seconds": CLAIM_TTL_SECONDS,
        }

    def _is_claim_stale(self, payload: dict[str, Any]) -> bool:
        claimed_at = payload.get("heartbeat_at") or payload.get("claimed_at")
        pid = payload.get("pid")
        age = _timestamp_age_seconds(claimed_at)
        if age is None:
            return True
        if age > payload.get("ttl_seconds", CLAIM_TTL_SECONDS):
            return True
        if payload.get("hostname") == socket.gethostname():
            return not _is_process_alive(pid)
        return False

    def check_claimed(self, experiment_key: str) -> bool:
        claim_path = self._claim_path(experiment_key)
        if not claim_path.exists():
            return False
        payload = _read_json(claim_path, default={})
        if self._is_claim_stale(payload):
            try:
                claim_path.unlink()
            except FileNotFoundError:
                pass
            return False
        return True

    def claim_experiment(self, description: str) -> Optional[str]:
        experiment_key = self._make_key(description)
        claim_path = self._claim_path(experiment_key)
        with _locked_file(self.state_lock_path):
            if claim_path.exists():
                payload = _read_json(claim_path, default={})
                if not self._is_claim_stale(payload):
                    self._log(f"Experiment already claimed: {experiment_key}")
                    return None
                claim_path.unlink(missing_ok=True)

            payload = self._claim_payload(experiment_key, description)
            _write_json(claim_path, payload)
            with self._status_lock:
                self._owned_claims.add(experiment_key)
        self._log(f"Claimed experiment: {experiment_key}")
        self.touch_agent_status(
            state="planning",
            message=f"claimed {experiment_key}",
            extra={
                "current_experiment_key": experiment_key,
                "current_description": description,
            },
        )
        return experiment_key

    def release_claim(self, experiment_key: str) -> None:
        claim_path = self._claim_path(experiment_key)
        with _locked_file(self.state_lock_path):
            payload = _read_json(claim_path, default={})
            if payload.get("agent_id") == self.agent_id:
                claim_path.unlink(missing_ok=True)
                with self._status_lock:
                    self._owned_claims.discard(experiment_key)
        self.touch_agent_status(
            state="idle",
            message="ready",
            drop_extra_keys=["current_experiment_key", "current_description"],
        )

    # ------------------------------------------------------------------
    # GPU lease
    # ------------------------------------------------------------------

    def _gpu_lock_payload(self) -> dict[str, Any]:
        now = _now_iso()
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": now,
            "heartbeat_at": now,
        }

    def _gpu_lock_is_stale(self) -> bool:
        payload = _read_json(self.gpu_lock_path, default={})
        if not payload:
            return False
        heartbeat_age = _timestamp_age_seconds(payload.get("heartbeat_at") or payload.get("acquired_at"))
        acquired_age = _timestamp_age_seconds(payload.get("acquired_at"))
        if heartbeat_age is None or acquired_age is None:
            return True
        if heartbeat_age > GPU_HEARTBEAT_STALE_SECONDS or acquired_age > GPU_LOCK_STALE_SECONDS:
            return True
        if payload.get("hostname") == socket.gethostname():
            return not _is_process_alive(payload.get("pid"))
        return False

    @contextlib.contextmanager
    def gpu_lease(self, poll_interval: float = 5.0, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        start = time.time()
        self.touch_agent_status(state="waiting_for_gpu", message="waiting for shared GPU lease")
        while True:
            with _locked_file(self.state_lock_path):
                if self.gpu_lock_path.exists() and self._gpu_lock_is_stale():
                    self.gpu_lock_path.unlink(missing_ok=True)

                if not self.gpu_lock_path.exists():
                    payload = self._gpu_lock_payload()
                    _write_json(self.gpu_lock_path, payload)
                    with self._status_lock:
                        self._holding_gpu_lease = True
                    self._log("Acquired GPU lease")
                    self.touch_agent_status(state="running", message="holding shared GPU lease")
                    try:
                        yield payload
                    finally:
                        with _locked_file(self.state_lock_path):
                            current = _read_json(self.gpu_lock_path, default={})
                            if current.get("pid") == os.getpid() and current.get("agent_id") == self.agent_id:
                                self.gpu_lock_path.unlink(missing_ok=True)
                                self._log("Released GPU lease")
                        with self._status_lock:
                            self._holding_gpu_lease = False
                        self.touch_agent_status(state="idle", message="ready")
                    return

            if timeout is not None and time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for the shared GPU lease")
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def publish_result(
        self,
        experiment_key: str,
        metrics: dict[str, Any],
        status: str,
        description: str,
        train_py_source: str,
        artifacts_dir: str | os.PathLike[str] | None = None,
        final_eval: bool = False,
    ) -> dict[str, Any]:
        metrics = _json_safe(dict(metrics))
        snapshot_dir = self.snapshots_dir / experiment_key
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        train_snapshot_path = snapshot_dir / "train.py"
        train_snapshot_path.write_text(train_py_source)

        local_artifacts_dir = Path(artifacts_dir).resolve() if artifacts_dir else None
        shared_artifact_dir = None
        if status == "keep" and local_artifacts_dir and local_artifacts_dir.exists():
            shared_artifact_dir = self.best_checkpoints_dir / experiment_key
            if shared_artifact_dir.exists():
                shutil.rmtree(shared_artifact_dir)
            shutil.copytree(local_artifacts_dir, shared_artifact_dir)
            for key in ["checkpoint_path", "val_logits_path", "metadata_path", "config_path", "manifest_path"]:
                if metrics.get(key):
                    path = Path(metrics[key])
                    if path.exists():
                        metrics[key] = str(shared_artifact_dir / path.name)

        record = {
            "experiment_key": experiment_key,
            "agent_id": self.agent_id,
            "role": self.role,
            "status": status,
            "description": description,
            "workspace_root": str(self.workspace_root),
            "branch": _git_branch(),
            "commit": _git_commit_short(),
            "repo_url": _git_remote_url(),
            "recorded_at": _now_iso(),
            "train_py_path": str(train_snapshot_path),
            "artifacts_dir": str(local_artifacts_dir) if local_artifacts_dir else None,
            "shared_artifact_dir": str(shared_artifact_dir) if shared_artifact_dir else None,
            "final_eval": bool(final_eval),
            "metrics": metrics,
        }

        with _locked_file(self.state_lock_path):
            _append_jsonl(self.experiment_log_path, record)
            _write_json(self.results_dir / f"{experiment_key}.json", record)
            bests = self._load_best_results()

            if status == "keep":
                if _is_better(record, bests.get("global_best")):
                    bests["global_best"] = record

                role_best = bests.setdefault("by_role", {}).get(self.role)
                if _is_better(record, role_best):
                    bests["by_role"][self.role] = record

                family = str(metrics.get("model_family", "unknown"))
                family_best = bests.setdefault("by_family", {}).get(family)
                if _is_better(record, family_best):
                    bests["by_family"][family] = record

                self._save_best_results(bests)

            claim_path = self._claim_path(experiment_key)
            if claim_path.exists():
                claim_payload = _read_json(claim_path, default={})
                if claim_payload.get("agent_id") == self.agent_id:
                    claim_path.unlink(missing_ok=True)
                    with self._status_lock:
                        self._owned_claims.discard(experiment_key)

        self._log(
            "Published result "
            f"{experiment_key}: status={status} val_errors={metrics.get('val_errors')} val_loss={metrics.get('val_loss')}"
        )
        self.touch_agent_status(
            state="idle",
            message=f"last result: {status}",
            extra={
                "last_result": {
                    "experiment_key": experiment_key,
                    "status": status,
                    "val_errors": metrics.get("val_errors"),
                    "val_loss": metrics.get("val_loss"),
                    "model_family": metrics.get("model_family"),
                    "run_mode": metrics.get("run_mode"),
                    "recorded_at": record["recorded_at"],
                    "final_eval": bool(final_eval),
                }
            },
            drop_extra_keys=["current_experiment_key", "current_description"],
        )
        return record

    def get_ranked_results(
        self,
        limit: int | None = None,
        *,
        status: str | None = "keep",
        run_mode: str | None = None,
        families: list[str] | None = None,
        roles: list[str] | None = None,
        final_eval: bool | None = None,
    ) -> list[dict[str, Any]]:
        rows = _load_jsonl(self.experiment_log_path)
        filtered = []
        family_set = set(families or [])
        role_set = set(roles or [])
        for row in rows:
            metrics = row.get("metrics", {})
            if status and row.get("status") != status:
                continue
            if run_mode and metrics.get("run_mode") != run_mode:
                continue
            if family_set and metrics.get("model_family") not in family_set:
                continue
            if role_set and row.get("role") not in role_set:
                continue
            if final_eval is not None and bool(row.get("final_eval")) != final_eval:
                continue
            filtered.append(row)
        filtered.sort(key=_record_sort_key)
        return filtered[:limit] if limit is not None else filtered

    def pull_best_config(
        self,
        scope: str = "global",
        *,
        role: str | None = None,
        family: str | None = None,
    ) -> Optional[dict[str, Any]]:
        bests = self._load_best_results()
        record = None
        if scope == "global":
            record = bests.get("global_best")
        elif scope == "role":
            record = bests.get("by_role", {}).get(role or self.role)
        elif scope == "family":
            if not family:
                raise ValueError("family is required when scope='family'")
            record = bests.get("by_family", {}).get(family)
        else:
            raise ValueError(f"Unknown scope: {scope}")

        if not record:
            return None

        train_py_path = Path(record["train_py_path"])
        train_py_source = train_py_path.read_text() if train_py_path.exists() else None
        metrics = record.get("metrics", {})
        result = {
            "record": record,
            "train_py_path": str(train_py_path),
            "train_py_source": train_py_source,
            "artifact_dir": record.get("shared_artifact_dir") or record.get("artifacts_dir"),
            "checkpoint_path": metrics.get("checkpoint_path"),
            "manifest_path": metrics.get("manifest_path"),
            "config_path": metrics.get("config_path"),
            "val_logits_path": metrics.get("val_logits_path"),
        }
        self._log(
            f"Pulled best config scope={scope} "
            f"val_errors={metrics.get('val_errors')} role={record.get('role')} family={metrics.get('model_family')}"
        )
        return result

    def get_gpu_lease_status(self) -> dict[str, Any] | None:
        payload = _read_json(self.gpu_lock_path, default={})
        if not payload:
            return None
        heartbeat_age = _timestamp_age_seconds(payload.get("heartbeat_at") or payload.get("acquired_at"))
        stale = heartbeat_age is None or heartbeat_age > GPU_HEARTBEAT_STALE_SECONDS
        result = dict(payload)
        result["stale"] = stale
        result["heartbeat_age_seconds"] = heartbeat_age
        return result

    def list_agent_statuses(self, include_stale: bool = False) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.agents_dir.glob("*.json")):
            payload = _read_json(path, default={})
            if not payload:
                continue
            age = _timestamp_age_seconds(payload.get("last_seen_at"))
            stale = age is None or age > AGENT_STALE_SECONDS
            payload["status_path"] = str(path)
            payload["stale"] = stale
            payload["last_seen_age_seconds"] = age
            if stale and not include_stale:
                continue
            rows.append(payload)
        rows.sort(
            key=lambda row: (
                bool(row.get("stale")),
                str(row.get("role", "")),
                str(row.get("agent_id", "")),
            )
        )
        return rows

    # ------------------------------------------------------------------
    # Thinking utilities
    # ------------------------------------------------------------------

    def should_sync(self) -> bool:
        self.experiment_count += 1
        return self.experiment_count % SYNC_EVERY_N == 0

    def post_insight(self, insight: str, evidence_keys: list[str] | None = None) -> None:
        payload = {
            "agent_id": self.agent_id,
            "role": self.role,
            "insight": insight,
            "evidence_keys": evidence_keys or [],
            "recorded_at": _now_iso(),
        }
        with _locked_file(self.state_lock_path):
            _append_jsonl(self.insights_path, payload)
        self._log(f"Posted insight: {insight}")

    def publish_hypothesis(
        self,
        title: str,
        hypothesis: str,
        suggested_config: dict[str, Any] | None = None,
        evidence_keys: list[str] | None = None,
        priority: int = 3,
    ) -> None:
        payload = {
            "agent_id": self.agent_id,
            "role": self.role,
            "title": title,
            "hypothesis": hypothesis,
            "suggested_config": suggested_config or {},
            "evidence_keys": evidence_keys or [],
            "priority": priority,
            "recorded_at": _now_iso(),
        }
        with _locked_file(self.state_lock_path):
            _append_jsonl(self.hypotheses_path, payload)
        self._log(f"Published hypothesis: {title}")

    def get_swarm_insights(self, topic: str, limit: int = 10) -> list[dict[str, Any]]:
        topic_norm = topic.lower()
        matches = []
        for row in _load_jsonl(self.insights_path):
            if topic_norm in row.get("insight", "").lower():
                matches.append(row)
        return matches[:limit]

    def get_unclaimed_hypotheses(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = _load_jsonl(self.hypotheses_path)
        rows.sort(key=lambda row: (-int(row.get("priority", 0)), row.get("recorded_at", "")))
        return rows[:limit]

    def analyze_swarm(self) -> dict[str, Any]:
        bests = self._load_best_results()
        recent_results = self.get_ranked_results(limit=10, status="keep")
        failures = [
            row
            for row in reversed(_load_jsonl(self.experiment_log_path))
            if row.get("status") in {"discard", "crash"}
        ][:10]

        active_claims = []
        for claim_file in sorted(self.claims_dir.glob("*.json")):
            payload = _read_json(claim_file, default={})
            if not payload:
                continue
            if self._is_claim_stale(payload):
                claim_file.unlink(missing_ok=True)
                continue
            active_claims.append(payload)

        gpu_lease = self.get_gpu_lease_status()
        active_agents = self.list_agent_statuses()

        lines = [
            "=" * 56,
            "LOCAL AUTORESEARCH SWARM",
            "=" * 56,
            f"Shared dir: {self.shared_dir}",
            f"Agent: {self.agent_id} ({self.role})",
        ]
        global_best = bests.get("global_best")
        if global_best:
            metrics = global_best.get("metrics", {})
            lines.append(
                "Global best: "
                f"val_errors={metrics.get('val_errors')} "
                f"val_loss={metrics.get('val_loss'):.6f} "
                f"role={global_best.get('role')} family={metrics.get('model_family')}"
            )
        else:
            lines.append("Global best: none yet")

        if gpu_lease and not gpu_lease.get("stale"):
            lines.append(
                "GPU lease: "
                f"held by {gpu_lease.get('role')}:{gpu_lease.get('agent_id')} "
                f"for {gpu_lease.get('heartbeat_age_seconds', 0):.0f}s since last heartbeat"
            )
        else:
            lines.append("GPU lease: free")

        lines.append(f"Active agents: {len(active_agents)}")
        for agent in active_agents[:5]:
            lines.append(
                f"  [{agent.get('role')}] {agent.get('agent_id')} "
                f"{agent.get('state')} - {agent.get('message')}"
            )

        lines.append(f"Active claims: {len(active_claims)}")
        for claim in active_claims[:5]:
            lines.append(f"  [{claim.get('role')}] {claim.get('description')}")

        lines.append(f"Recent keeps: {len(recent_results)}")
        for row in recent_results[:5]:
            metrics = row.get("metrics", {})
            lines.append(
                f"  [{row.get('role')}] val_errors={metrics.get('val_errors')} "
                f"val_loss={metrics.get('val_loss'):.6f} {row.get('description')}"
            )

        lines.append(f"Recent failures: {len(failures)}")
        for row in failures[:5]:
            lines.append(f"  [{row.get('role')}] {row.get('status')} {row.get('description')}")

        by_role = bests.get("by_role", {})
        if by_role:
            lines.append("Best by role:")
            for role_name, row in sorted(by_role.items()):
                metrics = row.get("metrics", {})
                lines.append(
                    f"  {role_name}: val_errors={metrics.get('val_errors')} "
                    f"family={metrics.get('model_family')} {row.get('description')}"
                )

        by_family = bests.get("by_family", {})
        if by_family:
            lines.append("Best by family:")
            for family_name, row in sorted(by_family.items()):
                metrics = row.get("metrics", {})
                lines.append(
                    f"  {family_name}: val_errors={metrics.get('val_errors')} "
                    f"role={row.get('role')} {row.get('description')}"
                )

        lines.append("=" * 56)
        return {
            "global_best": global_best,
            "by_role": by_role,
            "by_family": by_family,
            "recent_keeps": recent_results,
            "recent_failures": failures,
            "active_claims": active_claims,
            "active_agents": active_agents,
            "gpu_lease": gpu_lease,
            "summary": "\n".join(lines),
        }


Coordinator = LocalCoordinator
