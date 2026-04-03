# autoresearch-mnist

RunPod-first Codex swarm for MNIST research.

Five Codex agents collaborate through shared local state:
- A: CNN architecture search
- B: ViT / hybrid search
- C: augmentation and preprocessing
- D: optimizer and schedule
- E: ensemble construction

They can reason and edit in parallel, but only one training job holds the GPU lease at a time.

## Recommended deployment

The primary target is **one RunPod Pod** with:
- one custom image published to GHCR
- one reusable base image published to GHCR for heavy CUDA/PyTorch layers
- one network volume mounted at `/workspace`
- one supervisor process managing `agent-a` to `agent-e` plus the dashboard
- one encrypted Codex account vault synced from your local machine

See [RUNPOD.md](RUNPOD.md) for the full deployment walkthrough.

## Repository map

```text
prepare.py                    MNIST dataset prep and fixed splits
train.py                      single-model and ensemble entrypoint
coordinator.py                claims, best-results ledger and GPU lease
setup_hub.py                  operator CLI for auth, bootstrap and runpod
swarm_auth.py                 encrypted account vault, leases and remote sync
swarm_config.py               central config loader and runtime manifest generator
scripts/run_agent.py          per-agent runtime wrapper
scripts/runpod_supervisor.py  single-Pod supervisor
scripts/runpod_entrypoint.sh  RunPod container entrypoint
dashboard_app.py              read-only dashboard
config/swarm.yaml             models, roles, accounts and budgets
runpod/pod-template.json      RunPod Pod creation template
```

## Local operator flow

## Environment files

There are only two env templates you should care about:
- [.env.example](/home/dark/Desktop/Projects/autoresearch/.env.example): local Compose fallback only
- [.env.secrets.example](/home/dark/Desktop/Projects/autoresearch/.env.secrets.example): secrets and optional credentials

Rule of thumb:
- if you are deploying to RunPod, `config/swarm.yaml` and `runpod/pod-template.json` matter more than `.env`
- if you are running locally with Compose, copy both examples once and then mostly forget about them
- if you are just editing models, roles or budgets, change `config/swarm.yaml`, not the env files

```bash
cp .env.example .env
cp .env.secrets.example .env.secrets
$EDITOR config/swarm.yaml
```

Capture Codex auth locally:

```bash
export SWARM_VAULT_PASSPHRASE='choose-a-strong-passphrase'
python scripts/swarm auth add primary
python scripts/swarm auth verify
```

Sync the vault to the GPU host:

```bash
python scripts/swarm auth sync-remote --host root@YOUR_HOST
```

Inside the Pod:

```bash
python scripts/swarm runpod bootstrap
python scripts/swarm runpod start
python scripts/swarm runpod status
python scripts/swarm runpod doctor
python scripts/swarm runpod logs supervisor --tail 200
```

The dashboard listens on port `8080`.

## RunPod runtime layout

```text
/workspace/autoresearch/
  shared/
  homes/
  worktrees/
  swarm_logs/
  secrets/
```

Important shared files:

```text
shared/best_results.json
shared/experiment_log.jsonl
shared/accounts/health.json
shared/accounts/leases/*.json
shared/runtime/runpod-supervisor.json
shared/runtime/runpod-processes.json
shared/runtime/runpod-events.jsonl
```

## Selection protocol

- Selection is based on `val_errors` over a fixed `50k/10k` split from the training set.
- `val_loss` is the tie-breaker.
- `test_errors` are reserved for frozen finalists and reproducibility checks.
- Ensemble search only uses published validation artifacts, not the test set.

## Fallback modes

Compose fallback:

```bash
python scripts/swarm bootstrap
python scripts/swarm up
python scripts/swarm status
```

Host-only fallback:

```bash
uv sync
python setup_hub.py bootstrap
python setup_hub.py up --no-compose
```

## Dashboard API

- `/api/summary`
- `/api/leaderboard`
- `/api/experiments`
- `/api/agents`
- `/api/accounts`
- `/api/leases`
- `/api/health/auth`
- `/api/runpod`
- `/api/charts/best`

## Image publishing

The repo includes a GHCR publishing workflow at [.github/workflows/publish-image.yml](.github/workflows/publish-image.yml).

It now publishes two images:
- `ghcr.io/<owner>/autoresearch-mnist-base`
- `ghcr.io/<owner>/autoresearch-mnist`

The base image contains the heavy CUDA, Python, Node, Codex CLI and PyTorch dependency layers.
The app image is intentionally thin and mostly copies repo code on top of that base.

Published tags include:
- branch name like `master` or `main`
- short `sha`
- release tags like `v1.2.3`

## Non-negotiable rules

- Never tune directly on the MNIST test split.
- Do not bypass the GPU lease in `coordinator.py`.
- Keep secrets out of git; use the encrypted vault.
- Treat RunPod as the primary target and Compose as compatibility mode.
