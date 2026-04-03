# RunPod deployment

This repository is designed to run primarily as **one RunPod Pod** with **one GPU**, **one network volume**, **five local Codex agents**, **one supervisor**, and **one dashboard**.

## Architecture

Inside the Pod:
- the immutable image code lives at `/opt/autoresearch/app`
- the persistent working copy lives at `/workspace/autoresearch`
- `scripts/runpod_entrypoint.sh` syncs image code into `/workspace/autoresearch`
- `scripts/runpod_supervisor.py` starts and monitors `agent-a` to `agent-e` plus the dashboard

Persistent directories:

```text
/workspace/autoresearch/shared
/workspace/autoresearch/homes
/workspace/autoresearch/worktrees
/workspace/autoresearch/swarm_logs
/workspace/autoresearch/secrets
```

## Before you create the Pod

### 1. Publish the image

Push the repo to GitHub and let the GHCR workflow publish the image, or build manually:

```bash
docker build \
  -f Dockerfile.base \
  --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  -t ghcr.io/YOUR_ORG/autoresearch-mnist-base:master .
docker push ghcr.io/YOUR_ORG/autoresearch-mnist-base:master

docker build \
  -f Dockerfile \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=ghcr.io/YOUR_ORG/autoresearch-mnist-base:master \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  -t ghcr.io/YOUR_ORG/autoresearch-mnist:master .
docker push ghcr.io/YOUR_ORG/autoresearch-mnist:master
```

The heavy CUDA/PyTorch layers now live in the base image so the app image rebuild stays much faster.

### 2. Capture Codex auth locally

```bash
export SWARM_VAULT_PASSPHRASE='choose-a-strong-passphrase'
python scripts/swarm auth add primary
python scripts/swarm auth verify
```

If you prefer device auth:

```bash
python scripts/swarm auth add primary --device-auth
```

### 3. Create a passphrase file for the Pod

```bash
mkdir -p secrets
printf '%s\n' 'choose-a-strong-passphrase' > secrets/runpod-vault.passphrase
chmod 600 secrets/runpod-vault.passphrase
```

That file should end up at:

```text
/workspace/autoresearch/secrets/runpod-vault.passphrase
```

## Create the RunPod Pod

### Required choices

- GPU Pod
- one GPU
- network volume mounted at `/workspace`
- ports `22/tcp` and `8080/http`
- public IP enabled
- custom image pointing at your GHCR image

Use [runpod/pod-template.json](/home/dark/Desktop/Projects/autoresearch/runpod/pod-template.json) as the baseline request body or as a console checklist.

For normal iteration, keep the Pod on a branch tag such as `master`.
Once a workflow finishes and you want a fully reproducible deployment, replace that tag with the published digest from GHCR.

Recommended environment variables on the Pod:

```text
RUNPOD_MODE=1
RUNPOD_AUTO_BOOTSTRAP=1
RUNPOD_WORKSPACE_DIR=/workspace
RUNPOD_APP_DIR=/workspace/autoresearch
SWARM_CONFIG_PATH=/workspace/autoresearch/config/swarm.yaml
SWARM_VAULT_PASSPHRASE_FILE=/workspace/autoresearch/secrets/runpod-vault.passphrase
SSH_PUBLIC_KEY=<your ssh public key>
```

## First boot

The entrypoint will:
1. sync the image into `/workspace/autoresearch`
2. configure `sshd`
3. run `python3 setup_hub.py runpod bootstrap`
4. start the RunPod supervisor in the foreground

You can watch the container logs from RunPod, or attach by SSH once the Pod is ready.

## Sync the vault to the Pod

From your local machine:

```bash
python scripts/swarm auth sync-remote --host root@RUNPOD_PUBLIC_IP
```

If your SSH port is remapped, use the mapped TCP port:

```bash
ssh -p RUNPOD_TCP_PORT_22 root@RUNPOD_PUBLIC_IP
scp -P RUNPOD_TCP_PORT_22 -r secrets/codex-vault root@RUNPOD_PUBLIC_IP:/workspace/autoresearch/secrets/
```

## Operator commands inside the Pod

```bash
cd /workspace/autoresearch

python scripts/swarm runpod status
python scripts/swarm runpod doctor
python scripts/swarm runpod logs supervisor --tail 200
python scripts/swarm runpod logs dashboard --tail 200
python scripts/swarm runpod logs A --tail 200
```

If you disabled auto bootstrap in the Pod env:

```bash
python scripts/swarm runpod bootstrap
python scripts/swarm runpod start
```

## Dashboard access

Expose port `8080/http` on the Pod and open the RunPod HTTP endpoint, or tunnel over SSH:

```bash
ssh -L 8080:localhost:8080 -p RUNPOD_TCP_PORT_22 root@RUNPOD_PUBLIC_IP
```

Then open:

```text
http://localhost:8080
```

## Recovery flow

If the Pod is recreated with the same network volume:
- `/workspace/autoresearch/shared` is preserved
- `/workspace/autoresearch/homes` is preserved
- `/workspace/autoresearch/worktrees` is preserved
- `/workspace/autoresearch/secrets` is preserved

So recovery is usually just:

```bash
python scripts/swarm runpod doctor
python scripts/swarm runpod start
```

If an agent home is missing or stale, the runtime will hydrate it again from the encrypted vault.

## Useful paths

```text
/workspace/autoresearch/shared/runtime/runpod-supervisor.json
/workspace/autoresearch/shared/runtime/runpod-processes.json
/workspace/autoresearch/shared/runtime/runpod-events.jsonl
/workspace/autoresearch/shared/accounts/health.json
/workspace/autoresearch/swarm_logs/runpod-supervisor.log
```

## Notes

- The Pod is intentionally single-node and single-GPU.
- RunPod is the primary deploy target; Compose is now only a compatibility path.
- The supervisor will restart crashed agent processes with backoff according to `config/swarm.yaml`.
- The image runs `tini` in subreaper mode to avoid zombie-process warnings under RunPod's container wrapper. If you still see old `tini` warnings, rebuild and republish the image so the Pod picks up the latest `Dockerfile`.
