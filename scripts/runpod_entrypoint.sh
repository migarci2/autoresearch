#!/usr/bin/env bash
set -euo pipefail

RUNPOD_MODE="${RUNPOD_MODE:-0}"

if [[ "$RUNPOD_MODE" != "1" ]]; then
  exec "$@"
fi

IMAGE_APP_DIR="${AUTORESEARCH_IMAGE_APP_DIR:-/opt/autoresearch/app}"
WORKSPACE_DIR="${RUNPOD_WORKSPACE_DIR:-/workspace}"
APP_DIR="${RUNPOD_APP_DIR:-${WORKSPACE_DIR}/autoresearch}"
SECRETS_DIR="${APP_DIR}/secrets"
PASSFILE_DEFAULT="${SECRETS_DIR}/runpod-vault.passphrase"
SSH_PORT="${RUNPOD_SSH_PORT:-22}"
AUTO_BOOTSTRAP="${RUNPOD_AUTO_BOOTSTRAP:-1}"
IMAGE_REVISION="${AUTORESEARCH_IMAGE_REVISION:-dev}"
SYNC_STAMP="${APP_DIR}/.image-revision"

mkdir -p "${WORKSPACE_DIR}" "${APP_DIR}" "${SECRETS_DIR}" /var/run/sshd /root/.ssh
chmod 700 /root/.ssh

sync_app_tree() {
  if [[ ! -d "${IMAGE_APP_DIR}" ]]; then
    echo "Missing image app dir: ${IMAGE_APP_DIR}" >&2
    exit 1
  fi

  if [[ ! -f "${SYNC_STAMP}" ]] || [[ "$(<"${SYNC_STAMP}")" != "${IMAGE_REVISION}" ]]; then
    rsync -a --delete \
      --exclude '.git/' \
      --exclude '.venv/' \
      --exclude '__pycache__/' \
      --exclude '.pytest_cache/' \
      --exclude '.mypy_cache/' \
      --exclude '.ruff_cache/' \
      --exclude 'homes/' \
      --exclude 'shared/' \
      --exclude 'worktrees/' \
      --exclude 'swarm_logs/' \
      --exclude 'runs/' \
      --exclude 'results/' \
      --exclude 'secrets/' \
      "${IMAGE_APP_DIR}/" "${APP_DIR}/"
    printf '%s\n' "${IMAGE_REVISION}" > "${SYNC_STAMP}"
  fi
}

configure_ssh() {
  local keys_file="/root/.ssh/authorized_keys"
  : > "${keys_file}"
  chmod 600 "${keys_file}"

  for key_env in PUBLIC_KEY SSH_PUBLIC_KEY RUNPOD_PUBLIC_KEY AUTHORIZED_KEYS; do
    if [[ -n "${!key_env:-}" ]]; then
      printf '%s\n' "${!key_env}" >> "${keys_file}"
    fi
  done

  if [[ -n "${SSH_PUBLIC_KEYS_FILE:-}" ]] && [[ -f "${SSH_PUBLIC_KEYS_FILE}" ]]; then
    cat "${SSH_PUBLIC_KEYS_FILE}" >> "${keys_file}"
  fi

  if [[ -s "${keys_file}" ]]; then
    awk '!seen[$0]++' "${keys_file}" > "${keys_file}.tmp"
    mv "${keys_file}.tmp" "${keys_file}"
    chmod 600 "${keys_file}"
  fi

  ssh-keygen -A >/dev/null 2>&1 || true

  cat >/etc/ssh/sshd_config.d/runpod-autoresearch.conf <<EOF
Port ${SSH_PORT}
PermitRootLogin yes
PasswordAuthentication no
PubkeyAuthentication yes
PermitEmptyPasswords no
ChallengeResponseAuthentication no
UsePAM no
X11Forwarding no
PrintMotd no
ClientAliveInterval 60
ClientAliveCountMax 3
EOF
}

sync_app_tree
configure_ssh

export RUNPOD_MODE=1
export RUNPOD_WORKSPACE_DIR="${WORKSPACE_DIR}"
export RUNPOD_APP_DIR="${APP_DIR}"
export SWARM_CONFIG_PATH="${SWARM_CONFIG_PATH:-${APP_DIR}/config/swarm.yaml}"
export AUTORESEARCH_CONFIG_PATH="${AUTORESEARCH_CONFIG_PATH:-${SWARM_CONFIG_PATH}}"
export AUTORESEARCH_CONFIG_FILE="${AUTORESEARCH_CONFIG_FILE:-${SWARM_CONFIG_PATH}}"

if [[ -z "${SWARM_VAULT_PASSPHRASE_FILE:-}" ]] && [[ -f "${PASSFILE_DEFAULT}" ]]; then
  export SWARM_VAULT_PASSPHRASE_FILE="${PASSFILE_DEFAULT}"
fi

cd "${APP_DIR}"

if [[ "${AUTO_BOOTSTRAP}" == "1" ]]; then
  python3 setup_hub.py runpod bootstrap
fi

/usr/sbin/sshd -D -e &
SSHD_PID=$!
trap 'kill "${SSHD_PID}" 2>/dev/null || true' EXIT

exec python3 setup_hub.py runpod start --foreground
