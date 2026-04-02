# Local Collaboration Protocol

This repository uses filesystem-based shared memory instead of Ensue.

The recommended runtime is a single RunPod Pod with one supervisor process managing the five agents plus the dashboard.

Docker Compose remains a compatibility fallback, with every service bind-mounting the same repo root so `shared/`, `homes/` and `worktrees/` stay local and inspectable.

## Shared workspace

All agents point to the same `shared/` directory through `AUTORESEARCH_SHARED_DIR`.

On RunPod, the persistent root is `/workspace/autoresearch`, so all of the paths below survive Pod restarts when the same network volume is reused.

```text
shared/
  best_results.json
  experiment_log.jsonl
  insights.jsonl
  hypotheses.jsonl
  agents/
  claims/
  best_checkpoints/
  snapshots/
  locks/
```

## Claims

- Claims are deterministic by role plus normalized description.
- The claim key format is:

```text
<role>--<slug>--<hash>
```

- Claims live in `shared/claims/*.json`.
- Claims expire after 30 minutes or when the owning process disappears on the same host namespace.
- Each active owner refreshes a claim heartbeat so both RunPod-supervised workers and Compose-based workers do not depend only on local PID visibility.

Always claim before running a new experiment.

## GPU lease

There is a single GPU lease shared by the whole swarm:
- lock path: `shared/locks/gpu.lock`
- only one run can train or evaluate at a time
- `train.py` acquires the lease automatically through `Coordinator.gpu_lease()`

Do not bypass this mechanism.

## Agent heartbeats

- Each agent publishes its presence under `shared/agents/*.json`.
- Files include:
  - `state`
  - `message`
  - `last_seen_at`
  - `claim_keys`
  - `gpu_lease_held`
- This is the read-only source for dashboard and operator status views.

On RunPod, the supervisor additionally publishes:
- `shared/runtime/runpod-supervisor.json`
- `shared/runtime/runpod-processes.json`
- `shared/runtime/runpod-events.jsonl`

## Best tracking

`shared/best_results.json` stores:
- `global_best`
- `by_role`
- `by_family`

Best selection uses:
1. lower `val_errors`
2. lower `val_loss`
3. lower `training_seconds`

Only `keep` runs can update bests.

## Publication contract

Every experiment should publish:
- `coord.publish_result(...)`
- `coord.post_insight(...)`
- `coord.publish_hypothesis(...)`

Each result must include a metrics dict with at least:
- `val_errors`
- `val_accuracy`
- `val_loss`
- `train_loss`
- `training_seconds`
- `checkpoint_path`
- `config_sha`
- `run_mode`
- `model_family`

For final evaluations, also include:
- `test_errors`
- `test_accuracy`

## Pulling prior work

Use:
- `coord.pull_best_config(scope="global")`
- `coord.pull_best_config(scope="role")`
- `coord.pull_best_config(scope="family", family="cnn")`

Returned data includes the result record, the saved `train.py` snapshot and the shared artifact directory.

## Ensemble role

Role E should:
- read the best published single-model checkpoints from `shared/best_checkpoints/`
- run `train.py` with `RUN_MODE["kind"] = "ensemble"`
- optimize weights and temperature on validation only
- reserve test evaluation for frozen candidates
