# Local Autoresearch Loop

This repo runs a local Codex swarm for MNIST.

The primary production runtime is a single RunPod Pod where a local supervisor launches all five agents plus the dashboard. Compose and host-only execution remain fallback modes.

## Objective

Minimize `val_errors` on the fixed 50k/10k split from the MNIST training set.

Tie-breakers:
1. lower `val_loss`
2. lower `training_seconds`

The MNIST test set is not for online search. Use it only when `RUN_MODE["final_eval"] = True` for frozen candidates.

## Files in scope

- `config/swarm.yaml`: central operator config for agent models, role metadata and runtime defaults. Do not edit it during a search run unless the operator explicitly asks for a swarm-wide change.
- `prepare.py`: fixed dataset preparation and evaluation helpers. Do not modify during search.
- `train.py`: main search surface. Agents primarily modify config sections here.
- `coordinator.py`: local claims, bests, shared log and GPU lease.
- `collab.md`: shared-state protocol.

## Core loop

1. THINK
   - Read `coord.analyze_swarm()`.
   - Check `shared/best_results.json`.
   - Inspect the most relevant failures and recent keeps.
2. CLAIM
   - `exp_key = coord.claim_experiment("short description")`
   - If it returns `None`, choose another idea.
3. EDIT
   - Change only the config sections allowed by your role.
4. RUN
   - `uv run train.py > run.log 2>&1`
   - `train.py` acquires and releases the GPU lease automatically.
5. PARSE
   - Read the final block from `run.log`.
   - Extract at least `val_errors`, `val_accuracy`, `val_loss`, `train_loss`, `training_seconds`, `peak_vram_mb`, `run_mode`, `model_family`, `config_sha`.
6. DECIDE
   - `keep` if the run improves the relevant baseline by `val_errors` or ties on errors and improves `val_loss`.
   - Otherwise `discard`.
   - If the run crashes, publish `crash`.
7. PUBLISH
   - `coord.publish_result(...)`
   - `coord.post_insight(...)`
   - `coord.publish_hypothesis(...)`

## Keep / discard policy

- Primary metric: `val_errors`
- Secondary metric: `val_loss`
- Tertiary metric: `training_seconds`

Do not keep a change just because test accuracy looked better. The test set is not the search objective.

## Train output

`train.py` prints a parseable summary block:

```text
---
val_errors: 37
val_accuracy: 0.996300
val_loss: 0.011204
train_loss: 0.008913
training_seconds: 45.000000
peak_vram_mb: 812.375000
checkpoint_path: /abs/path/to/checkpoint.pt
run_mode: single_model
model_family: cnn
config_sha: abc123def456
```

Final evaluation may additionally include `test_errors` and `test_accuracy`.

## Publishing shape

Use a metrics dict like:

```python
metrics = {
    "val_errors": ...,
    "val_accuracy": ...,
    "val_loss": ...,
    "train_loss": ...,
    "training_seconds": ...,
    "peak_vram_mb": ...,
    "checkpoint_path": "...",
    "config_path": "...",
    "val_logits_path": "...",
    "metadata_path": "...",
    "config_sha": "...",
    "run_mode": "...",
    "model_family": "...",
}
```

Include `test_errors` and `test_accuracy` only for final evaluations.
