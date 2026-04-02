"""Shared swarm configuration and runtime helpers.

This module keeps the launcher, agent entrypoint, and runtime manifests on the
same central source of truth: ``config/swarm.yaml``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any

import yaml

ALLOWED_PASSTHROUGH_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TZ",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "GIT_SSH_COMMAND",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "SWARM_VAULT_PASSPHRASE",
    "SWARM_VAULT_PASSPHRASE_FILE",
    "SWARM_REMOTE_SYNC_PATH",
    "RUNPOD_MODE",
    "RUNPOD_POD_ID",
    "RUNPOD_PUBLIC_IP",
    "RUNPOD_TCP_PORT_22",
    "RUNPOD_TCP_PORT_8080",
    "RUNPOD_WORKSPACE_DIR",
    "RUNPOD_APP_DIR",
    "RUNPOD_AUTO_BOOTSTRAP",
    "AUTORESEARCH_IMAGE_APP_DIR",
    "AUTORESEARCH_IMAGE_REVISION",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def normalize_agent_key(value: str) -> str:
    return value.strip().upper()


def resolve_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Swarm config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Swarm config must be a mapping: {path}")
    return data


@dataclass(frozen=True)
class PathLayout:
    shared_dir: Path
    homes_dir: Path
    worktrees_dir: Path
    logs_dir: Path
    runtime_dir: Path
    prompt_root: Path
    compose_file: Path


@dataclass(frozen=True)
class ComposeLayout:
    enabled: bool
    project_name: str
    file: Path
    agent_service_prefix: str
    toolbox_service: str
    dashboard_service: str


@dataclass(frozen=True)
class DeployLayout:
    target: str
    workspace_dir: Path
    app_dir: Path
    image_app_dir: Path
    dashboard_port: int
    ssh_port: int
    auto_bootstrap: bool
    code_sync_mode: str
    supervisor_restart_policy: str

    @property
    def runpod_enabled(self) -> bool:
        return self.target.lower() == "runpod"


@dataclass(frozen=True)
class CodexLayout:
    login_command: tuple[str, ...]
    status_command: tuple[str, ...]
    auth_payload_files: tuple[str, ...]
    exec_base_args: tuple[str, ...]


@dataclass(frozen=True)
class AuthLayout:
    mode: str
    capture_preferred: str
    capture_fallbacks: tuple[str, ...]
    vault_path: Path
    remote_sync_path: str
    force_credentials_store: str
    passphrase_env: str
    passphrase_file_env: str
    lease_ttl_seconds: int
    lease_heartbeat_seconds: int
    suspect_cooldown_seconds: int
    max_auth_retries: int


@dataclass(frozen=True)
class AccountConfig:
    account_id: str
    label: str
    capacity: int
    enabled: bool = True
    workspace_id: str | None = None
    role_affinity: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    key: str
    agent_id: str
    role: str
    label: str
    family: str
    branch: str
    prompt_file: str
    model: str
    reasoning_effort: str
    enabled: bool = True
    extra_codex_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    @property
    def service_name(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentRuntime:
    spec: AgentConfig
    home: Path
    worktree: Path
    prompt_path: Path
    log_path: Path
    last_message_path: Path
    pid_path: Path
    manifest_path: Path
    env_path: Path


@dataclass(frozen=True)
class SwarmConfig:
    source_path: Path
    version: int
    defaults: dict[str, Any]
    paths: PathLayout
    deploy: DeployLayout
    compose: ComposeLayout
    codex: CodexLayout
    auth: AuthLayout
    dashboard: dict[str, Any]
    agents: dict[str, AgentConfig]
    accounts: dict[str, AccountConfig]

    @property
    def root(self) -> Path:
        return self.source_path.parent.parent

    def agent(self, key_or_id: str) -> AgentConfig:
        key = normalize_agent_key(key_or_id)
        if key in self.agents:
            return self.agents[key]
        for spec in self.agents.values():
            if spec.agent_id == key_or_id or spec.agent_id.lower() == key_or_id.lower():
                return spec
        raise KeyError(f"Unknown agent or role: {key_or_id}")

    def enabled_agents(self) -> list[AgentConfig]:
        return [spec for spec in self.agents.values() if spec.enabled]

    def account(self, account_id: str) -> AccountConfig:
        if account_id not in self.accounts:
            raise KeyError(f"Unknown account id: {account_id}")
        return self.accounts[account_id]

    def enabled_accounts(self) -> list[AccountConfig]:
        return [spec for spec in self.accounts.values() if spec.enabled]

    def total_enabled_account_capacity(self) -> int:
        return sum(max(0, spec.capacity) for spec in self.enabled_accounts())

    def agent_keys(self) -> list[str]:
        return list(self.agents.keys())

    def runtime(self, key_or_id: str) -> AgentRuntime:
        spec = self.agent(key_or_id)
        return AgentRuntime(
            spec=spec,
            home=self.paths.homes_dir / spec.agent_id,
            worktree=self.paths.worktrees_dir / spec.agent_id,
            prompt_path=(self.paths.prompt_root / spec.prompt_file).resolve()
            if not Path(spec.prompt_file).is_absolute()
            else Path(spec.prompt_file),
            log_path=self.paths.logs_dir / f"{spec.agent_id}.log",
            last_message_path=self.paths.logs_dir / f"{spec.agent_id}.last.txt",
            pid_path=self.paths.logs_dir / f"{spec.agent_id}.pid",
            manifest_path=self.paths.runtime_dir / f"{spec.agent_id}.json",
            env_path=self.paths.runtime_dir / f"{spec.agent_id}.env",
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config_path": str(self.source_path),
            "repo_root": str(self.root),
            "defaults": self.defaults,
            "paths": {
                "shared_dir": str(self.paths.shared_dir),
                "homes_dir": str(self.paths.homes_dir),
                "worktrees_dir": str(self.paths.worktrees_dir),
                "logs_dir": str(self.paths.logs_dir),
                "runtime_dir": str(self.paths.runtime_dir),
                "prompt_root": str(self.paths.prompt_root),
                "compose_file": str(self.paths.compose_file),
            },
            "deploy": {
                "target": self.deploy.target,
                "workspace_dir": str(self.deploy.workspace_dir),
                "app_dir": str(self.deploy.app_dir),
                "image_app_dir": str(self.deploy.image_app_dir),
                "dashboard_port": self.deploy.dashboard_port,
                "ssh_port": self.deploy.ssh_port,
                "auto_bootstrap": self.deploy.auto_bootstrap,
                "code_sync_mode": self.deploy.code_sync_mode,
                "supervisor_restart_policy": self.deploy.supervisor_restart_policy,
            },
            "compose": {
                "enabled": self.compose.enabled,
                "project_name": self.compose.project_name,
                "file": str(self.compose.file),
                "agent_service_prefix": self.compose.agent_service_prefix,
                "toolbox_service": self.compose.toolbox_service,
                "dashboard_service": self.compose.dashboard_service,
            },
            "dashboard": self.dashboard,
            "auth": {
                "mode": self.auth.mode,
                "capture_preferred": self.auth.capture_preferred,
                "capture_fallbacks": list(self.auth.capture_fallbacks),
                "vault_path": str(self.auth.vault_path),
                "remote_sync_path": self.auth.remote_sync_path,
                "force_credentials_store": self.auth.force_credentials_store,
                "lease_ttl_seconds": self.auth.lease_ttl_seconds,
                "lease_heartbeat_seconds": self.auth.lease_heartbeat_seconds,
                "suspect_cooldown_seconds": self.auth.suspect_cooldown_seconds,
                "max_auth_retries": self.auth.max_auth_retries,
            },
            "agents": {
                key: self._agent_manifest(spec) for key, spec in self.agents.items()
            },
            "accounts": {
                account_id: {
                    "account_id": spec.account_id,
                    "label": spec.label,
                    "capacity": spec.capacity,
                    "enabled": spec.enabled,
                    "workspace_id": spec.workspace_id,
                    "role_affinity": list(spec.role_affinity),
                    "metadata": spec.metadata,
                }
                for account_id, spec in self.accounts.items()
            },
        }

    def _agent_manifest(self, spec: AgentConfig) -> dict[str, Any]:
        runtime = self.runtime(spec.key)
        return {
            "key": spec.key,
            "agent_id": spec.agent_id,
            "role": spec.role,
            "label": spec.label,
            "family": spec.family,
            "branch": spec.branch,
            "enabled": spec.enabled,
            "model": spec.model,
            "reasoning_effort": spec.reasoning_effort,
            "prompt_file": spec.prompt_file,
            "prompt_path": str(runtime.prompt_path),
            "home": str(runtime.home),
            "worktree": str(runtime.worktree),
            "log_path": str(runtime.log_path),
            "last_message_path": str(runtime.last_message_path),
            "pid_path": str(runtime.pid_path),
            "manifest_path": str(runtime.manifest_path),
            "env_path": str(runtime.env_path),
            "extra_codex_args": list(spec.extra_codex_args),
            "env": spec.env,
            "compose_service": self.compose.agent_service_prefix + spec.role.lower(),
        }


def load_swarm_config(path: str | os.PathLike[str] | None = None) -> SwarmConfig:
    config_path = Path(path) if path is not None else repo_root() / "config" / "swarm.yaml"
    if not config_path.is_absolute():
        config_path = (repo_root() / config_path).resolve()

    data = _load_yaml(config_path)
    base = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent

    paths_cfg = data.get("paths", {})
    defaults_cfg = data.get("defaults", {})
    deploy_cfg = data.get("deploy", {})
    compose_cfg = data.get("compose", {})
    codex_cfg = data.get("codex", {})
    dashboard_cfg = data.get("dashboard", {})
    auth_cfg = data.get("auth", {})
    agents_cfg = data.get("agents", {})
    accounts_cfg = data.get("accounts", {})

    shared_dir = resolve_path(paths_cfg.get("shared_dir", "shared"), base)
    homes_dir = resolve_path(paths_cfg.get("homes_dir", "homes"), base)
    worktrees_dir = resolve_path(paths_cfg.get("worktrees_dir", "worktrees"), base)
    logs_dir = resolve_path(paths_cfg.get("logs_dir", "swarm_logs"), base)
    runtime_dir = resolve_path(paths_cfg.get("runtime_dir", "shared/runtime"), base)
    prompt_root = resolve_path(paths_cfg.get("prompt_root", "."), base)
    compose_file = resolve_path(paths_cfg.get("compose_file", "compose.yaml"), base)

    paths = PathLayout(
        shared_dir=shared_dir,
        homes_dir=homes_dir,
        worktrees_dir=worktrees_dir,
        logs_dir=logs_dir,
        runtime_dir=runtime_dir,
        prompt_root=prompt_root,
        compose_file=compose_file,
    )

    runpod_cfg = deploy_cfg.get("runpod", {}) if isinstance(deploy_cfg.get("runpod", {}), dict) else {}
    deploy = DeployLayout(
        target=str(deploy_cfg.get("target", "runpod")),
        workspace_dir=resolve_path(runpod_cfg.get("workspace_dir", "/workspace"), base),
        app_dir=resolve_path(runpod_cfg.get("app_dir", "/workspace/autoresearch"), base),
        image_app_dir=resolve_path(
            runpod_cfg.get("image_app_dir", "/opt/autoresearch/app"),
            base,
        ),
        dashboard_port=int(runpod_cfg.get("dashboard_port", 8080)),
        ssh_port=int(runpod_cfg.get("ssh_port", 22)),
        auto_bootstrap=bool(runpod_cfg.get("auto_bootstrap", True)),
        code_sync_mode=str(runpod_cfg.get("code_sync_mode", "image_to_workspace")),
        supervisor_restart_policy=str(
            runpod_cfg.get("supervisor_restart_policy", "unless-stopped")
        ),
    )

    compose = ComposeLayout(
        enabled=bool(compose_cfg.get("enabled", True)),
        project_name=str(compose_cfg.get("project_name", "autoresearch")),
        file=compose_file,
        agent_service_prefix=str(compose_cfg.get("agent_service_prefix", "agent-")),
        toolbox_service=str(compose_cfg.get("toolbox_service", "toolbox")),
        dashboard_service=str(compose_cfg.get("dashboard_service", "dashboard")),
    )

    codex = CodexLayout(
        login_command=tuple(codex_cfg.get("login_command", ["codex", "login"])),
        status_command=tuple(codex_cfg.get("status_command", ["codex", "login", "status"])),
        auth_payload_files=tuple(
            codex_cfg.get(
                "auth_payload_files",
                ["config.toml", "auth.json", "credentials.json", "device.json"],
            )
        ),
        exec_base_args=tuple(
            codex_cfg.get(
                "exec_base_args",
                ["codex", "exec", "-s", "danger-full-access", "-a", "never"],
            )
        ),
    )

    vault_path = resolve_path(auth_cfg.get("vault_path", "secrets/codex-vault"), base)
    auth = AuthLayout(
        mode=str(auth_cfg.get("mode", "chatgpt_cache")),
        capture_preferred=str(auth_cfg.get("capture", {}).get("preferred", "local_browser")),
        capture_fallbacks=tuple(
            str(item)
            for item in auth_cfg.get("capture", {}).get(
                "fallbacks", ["device_auth", "ssh_callback_tunnel"]
            )
        ),
        vault_path=vault_path,
        remote_sync_path=str(auth_cfg.get("remote_sync_path", "secrets/codex-vault")),
        force_credentials_store=str(auth_cfg.get("credentials_store", "file")),
        passphrase_env=str(auth_cfg.get("passphrase_env", "SWARM_VAULT_PASSPHRASE")),
        passphrase_file_env=str(
            auth_cfg.get("passphrase_file_env", "SWARM_VAULT_PASSPHRASE_FILE")
        ),
        lease_ttl_seconds=int(auth_cfg.get("lease_ttl_seconds", 15 * 60)),
        lease_heartbeat_seconds=int(auth_cfg.get("lease_heartbeat_seconds", 15)),
        suspect_cooldown_seconds=int(auth_cfg.get("suspect_cooldown_seconds", 15 * 60)),
        max_auth_retries=int(auth_cfg.get("max_auth_retries", 3)),
    )

    agents: dict[str, AgentConfig] = {}
    for raw_key, raw_spec in agents_cfg.items():
        key = normalize_agent_key(str(raw_key))
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Agent {raw_key!r} must be a mapping")
        agent_id = str(raw_spec.get("agent_id", f"agent-{key.lower()}"))
        role = str(raw_spec.get("role", key))
        label = str(raw_spec.get("label", agent_id))
        family = str(raw_spec.get("family", defaults_cfg.get("model_family", "generic")))
        branch = str(raw_spec.get("branch", f"autoresearch/mnist-{agent_id}"))
        prompt_file = str(raw_spec.get("prompt_file", f"program_{agent_id}.md"))
        model = str(raw_spec.get("model", defaults_cfg.get("model", "gpt-5.4")))
        reasoning_effort = str(
            raw_spec.get("reasoning_effort", defaults_cfg.get("reasoning_effort", "medium"))
        )
        enabled = bool(raw_spec.get("enabled", True))
        extra_codex_args = tuple(str(item) for item in raw_spec.get("extra_codex_args", []))
        env = {str(k): str(v) for k, v in raw_spec.get("env", {}).items()}
        agents[key] = AgentConfig(
            key=key,
            agent_id=agent_id,
            role=role,
            label=label,
            family=family,
            branch=branch,
            prompt_file=prompt_file,
            model=model,
            reasoning_effort=reasoning_effort,
            enabled=enabled,
            extra_codex_args=extra_codex_args,
            env=env,
        )

    accounts: dict[str, AccountConfig] = {}
    for raw_account_id, raw_spec in accounts_cfg.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Account {raw_account_id!r} must be a mapping")
        account_id = str(raw_spec.get("id", raw_account_id))
        capacity = max(1, int(raw_spec.get("capacity", 1)))
        role_affinity = tuple(str(item).upper() for item in raw_spec.get("role_affinity", []))
        metadata = {str(k): str(v) for k, v in raw_spec.get("metadata", {}).items()}
        accounts[account_id] = AccountConfig(
            account_id=account_id,
            label=str(raw_spec.get("label", account_id)),
            capacity=capacity,
            enabled=bool(raw_spec.get("enabled", True)),
            workspace_id=str(raw_spec["workspace_id"]) if raw_spec.get("workspace_id") else None,
            role_affinity=role_affinity,
            metadata=metadata,
        )

    return SwarmConfig(
        source_path=config_path,
        version=int(data.get("version", 1)),
        defaults={str(k): v for k, v in defaults_cfg.items()},
        paths=paths,
        deploy=deploy,
        compose=compose,
        codex=codex,
        auth=auth,
        dashboard={str(k): v for k, v in dashboard_cfg.items()},
        agents=agents,
        accounts=accounts,
    )


def ensure_shared_layout(config: SwarmConfig) -> None:
    for path in (
        config.paths.shared_dir,
        config.paths.homes_dir,
        config.paths.worktrees_dir,
        config.paths.logs_dir,
        config.paths.runtime_dir,
        config.paths.shared_dir / "agents",
        config.paths.shared_dir / "claims",
        config.paths.shared_dir / "best_checkpoints",
        config.paths.shared_dir / "results",
        config.paths.shared_dir / "snapshots",
        config.paths.shared_dir / "locks",
        config.paths.shared_dir / "accounts",
        config.paths.shared_dir / "accounts" / "leases",
        config.paths.shared_dir / "accounts" / "runtime_cache",
        config.auth.vault_path,
    ):
        path.mkdir(parents=True, exist_ok=True)

    best_results_path = config.paths.shared_dir / "best_results.json"
    if not best_results_path.exists():
        best_results_path.write_text(
            json.dumps(
                {
                    "global_best": None,
                    "by_role": {},
                    "by_family": {},
                    "updated_at": None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    experiment_log_path = config.paths.shared_dir / "experiment_log.jsonl"
    experiment_log_path.touch(exist_ok=True)

    for relative in ("insights.jsonl", "hypotheses.jsonl"):
        (config.paths.shared_dir / relative).touch(exist_ok=True)

    (config.paths.shared_dir / "accounts" / "events.jsonl").touch(exist_ok=True)
    health_path = config.paths.shared_dir / "accounts" / "health.json"
    if not health_path.exists():
        health_path.write_text(json.dumps({"accounts": {}, "updated_at": None}, indent=2) + "\n")

    for lock_name in ("gpu.lock", "state.lock"):
        (config.paths.shared_dir / "locks" / lock_name).touch(exist_ok=True)


def runtime_env_allowlist() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ALLOWED_PASSTHROUGH_ENV:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


def build_runtime_env(
    config: SwarmConfig,
    runtime: AgentRuntime,
    *,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env = runtime_env_allowlist()
    env.update(
        {
            "HOME": str(runtime.home),
            "CODEX_HOME": str(runtime.home / ".codex"),
            "AUTORESEARCH_SHARED_DIR": str(config.paths.shared_dir),
            "AUTORESEARCH_WORKSPACE_ROOT": str(runtime.worktree),
            "AUTORESEARCH_AGENT_ID": runtime.spec.agent_id,
            "AUTORESEARCH_ROLE": runtime.spec.role,
            "AUTORESEARCH_AGENT_LABEL": runtime.spec.label,
            "AUTORESEARCH_MODEL": runtime.spec.model,
            "AUTORESEARCH_REASONING_EFFORT": runtime.spec.reasoning_effort,
            "AUTORESEARCH_PROMPT_FILE": str(runtime.prompt_path),
            "AUTORESEARCH_CONFIG_FILE": str(config.source_path),
            "AUTORESEARCH_CONFIG_PATH": str(config.source_path),
            "AUTORESEARCH_TIME_BUDGET": str(config.defaults.get("time_budget_seconds", 45)),
            "AUTORESEARCH_FINAL_TIME_BUDGET": str(config.defaults.get("final_time_budget_seconds", 300)),
            "AUTORESEARCH_FINAL_EVAL": str(config.defaults.get("final_eval", False)).lower(),
            "AUTORESEARCH_RUNTIME_MANIFEST": str(runtime.manifest_path),
            "AUTORESEARCH_RUNTIME_ENV": str(runtime.env_path),
            "AUTORESEARCH_MODEL_FAMILY": runtime.spec.family,
            "SWARM_CONFIG_PATH": str(config.source_path),
        }
    )
    env.update(runtime.spec.env)
    if extra:
        env.update(extra)
    return env


def codex_login_status(config: SwarmConfig, home: Path) -> tuple[bool, str]:
    env = runtime_env_allowlist()
    env["HOME"] = str(home)
    env["CODEX_HOME"] = str(home / ".codex")
    result = subprocess.run(
        list(config.codex.status_command),
        cwd=config.root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    logged_in = result.returncode == 0 and "Logged in" in output
    return logged_in, output.strip()


def build_codex_login_command(config: SwarmConfig, *, device_auth: bool = False) -> list[str]:
    command = list(config.codex.login_command)
    if device_auth and "--device-auth" not in command:
        command.append("--device-auth")
    return command


def build_codex_exec_command(
    config: SwarmConfig,
    runtime: AgentRuntime,
    *,
    model_override: str | None = None,
    last_message_path: Path | None = None,
) -> list[str]:
    model = model_override or runtime.spec.model or str(config.defaults.get("model", "gpt-5.4"))
    command = list(config.codex.exec_base_args)
    if model:
        command.extend(["-m", model])
    command.extend(["-C", str(runtime.worktree)])
    command.extend(["-o", str(last_message_path or runtime.last_message_path)])
    command.extend(runtime.spec.extra_codex_args)
    command.append("-")
    return command


def write_runtime_manifests(config: SwarmConfig) -> dict[str, Path]:
    ensure_shared_layout(config)
    summary = config.manifest()
    summary_path = config.paths.runtime_dir / "swarm.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    for spec in config.agents.values():
        runtime = config.runtime(spec.key)
        payload = summary["agents"][spec.key]
        runtime.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        runtime.env_path.write_text(
            "\n".join(
                f"{key}={quote(value)}"
                for key, value in build_runtime_env(config, runtime).items()
            )
            + "\n"
        )

    compose_env_path = config.paths.runtime_dir / "compose.env"
    compose_env_path.write_text(
        "\n".join(
            [
                f"SWARM_CONFIG_PATH={quote(str(config.source_path))}",
                f"SWARM_SHARED_DIR={quote(str(config.paths.shared_dir))}",
                f"SWARM_RUNTIME_DIR={quote(str(config.paths.runtime_dir))}",
                f"SWARM_HOMES_DIR={quote(str(config.paths.homes_dir))}",
                f"SWARM_WORKTREES_DIR={quote(str(config.paths.worktrees_dir))}",
                f"SWARM_LOGS_DIR={quote(str(config.paths.logs_dir))}",
                f"SWARM_COMPOSE_FILE={quote(str(config.compose.file))}",
                f"SWARM_COMPOSE_PROJECT={quote(config.compose.project_name)}",
                f"SWARM_DEPLOY_TARGET={quote(config.deploy.target)}",
                f"SWARM_RUNPOD_WORKSPACE={quote(str(config.deploy.workspace_dir))}",
                f"SWARM_RUNPOD_APP_DIR={quote(str(config.deploy.app_dir))}",
                f"SWARM_RUNPOD_DASHBOARD_PORT={quote(str(config.deploy.dashboard_port))}",
                f"SWARM_RUNPOD_SSH_PORT={quote(str(config.deploy.ssh_port))}",
                f"SWARM_VAULT_PATH={quote(str(config.auth.vault_path))}",
                f"SWARM_VAULT_PASSPHRASE_ENV={quote(config.auth.passphrase_env)}",
                f"SWARM_VAULT_PASSPHRASE_FILE_ENV={quote(config.auth.passphrase_file_env)}",
            ]
        )
        + "\n"
    )
    return {
        "summary": summary_path,
        "compose_env": compose_env_path,
    }


def copy_codex_auth_payload(
    config: SwarmConfig,
    *,
    source_home: Path,
    target_home: Path,
) -> list[Path]:
    source_codex = source_home / ".codex"
    target_codex = target_home / ".codex"
    if not source_codex.exists():
        raise FileNotFoundError(f"Source Codex home does not exist: {source_codex}")

    target_codex.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for relative in config.codex.auth_payload_files:
        src = source_codex / relative
        dst = target_codex / relative
        if not src.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied.append(dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def compose_command(config: SwarmConfig, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(config.compose.file),
        "--project-name",
        config.compose.project_name,
        *args,
    ]


def compose_ready(config: SwarmConfig) -> bool:
    if not config.compose.enabled or not config.compose.file.exists():
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        cwd=config.root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def format_agent_runtime(runtime: AgentRuntime, logged_in: bool, login_output: str, pid: int | None, alive: bool) -> dict[str, Any]:
    return {
        "agent_id": runtime.spec.agent_id,
        "role": runtime.spec.role,
        "label": runtime.spec.label,
        "family": runtime.spec.family,
        "model": runtime.spec.model,
        "reasoning_effort": runtime.spec.reasoning_effort,
        "home": str(runtime.home),
        "worktree": str(runtime.worktree),
        "prompt_path": str(runtime.prompt_path),
        "log_path": str(runtime.log_path),
        "last_message_path": str(runtime.last_message_path),
        "pid_path": str(runtime.pid_path),
        "manifest_path": str(runtime.manifest_path),
        "env_path": str(runtime.env_path),
        "compose_service": runtime.spec.service_name,
        "logged_in": logged_in,
        "login_output": login_output,
        "pid": pid,
        "alive": alive,
    }
