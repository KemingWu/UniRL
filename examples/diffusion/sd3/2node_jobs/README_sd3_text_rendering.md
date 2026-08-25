# SD3.5-Medium GRPO — Text-Rendering Reward via Remote Qwen3-VL-MoE-235B Judge

Reference copy of the 2-node taiji job for running SD3.5-Medium GRPO with
the trained **Qwen3-VL-MoE-235B error-judge** as the reward. The judge
reads each generated image, emits a JSON list of text-rendering errors,
and the scorer converts `n_errors → exp(-α · n_errors) ∈ (0, 1]` into a
scalar GRPO reward.

Kept in lockstep with:
- Recipe: `../sd3_text_rendering_grpo.yaml`
- Reward service config: `../../../../unirl-reward-service/configs/text_rendering_judge_service.yaml`
- Reward scorer: `../../../../unirl-reward-service/reward_service/scorers/text_rendering_judge.py`
- Prompt dataset (generated locally via `scripts/filter_infographic_prompts.py`):
  `datasets/infographic_medium/{train,test}.jsonl`

## Topology

| Node | GPUs | Role |
|---|---|---|
| 0 (`INDEX=0`) | 8 (all) | Qwen3-VL-MoE-235B judge in vLLM, TP=8, Ray head on 6379 |
| 1 (`INDEX=1`) | 8 (all) | SD3.5-Medium GRPO training, FSDP+LoRA, Ray on 6380 |

The trainer on Node 1 reaches Node 0 over HTTP via `REWARD_SERVICE_URL`.
Because the whole 235B judge takes all 8 GPUs on Node 0 (TP=8, ~55 GB per
rank in bf16), this deployment is not GPU-wasteful — both nodes are fully
utilized.

## Placeholders to fill

| Placeholder | Meaning |
|---|---|
| `<CEPHFS_USER_ROOT>` | Personal CephFS mount (e.g. `/apdcephfs_<region>/private_<user>`) |
| `<CEPHFS_SHARED_ROOT>` | Shared CephFS root hosting SD3.5-Medium (e.g. `/apdcephfs_<region>/share_<team>`) |
| `<TAIJI_TOKEN>` | Main taiji cluster token |
| `<TAIJI_MLLM_TOKEN>` | CephFS BJ region mount token |
| `<ZWFY8_TOKEN>` | CephFS zw region mount token (for `/apdcephfs_zwfy8` — judge weights live here) |
| `<HF_TOKEN>` | HuggingFace token (base env fallback) |
| `<WANDB_API_KEY>` | wandb login |

## Prerequisite: judge checkpoint access

The judge lives at
`/apdcephfs_zwfy8/share_305110755/zehanwang/verl/checkpoints/.../actor/huggingface/`
(96 safetensors shards, ~439 GB). The launcher mounts `apdcephfs_zwfy8`
automatically via
`taiji_client mount -tk "${ZWFY8_TOKEN}" -bf TaiJi_HYAide_800H20 -l zw`
(zw = 中卫 region). Fill in the real `ZWFY8_TOKEN` before submitting; the
launcher fails fast on Node 0 startup with a clear message if
`${JUDGE_CKPT_PATH}/config.json` is unreadable.

## What the launcher does (Node 0)

1. Mounts CephFS regions (personal + shared + BJ + zwfy8)
2. Activates the image's venv
3. Installs the reward service editable with `--ignore-requires-python`
4. Verifies vLLM can load the judge's `config.json` (`Qwen3VLMoeForConditionalGeneration`) — bumps vLLM if needed
5. `ray start --head --port=6379 --num-gpus=8`
6. `exec python -m reward_service --config configs/text_rendering_judge_service.yaml`

Cold-start budget: 15–25 min (Ray + vLLM engine spin-up + loading 96 shards).

## What the launcher does (Node 1)

1. `pip install --no-deps -e .` (editable UniRL)
2. Polls `${REWARD_SERVICE_URL}/health` up to 20 min
3. If `PRETRAINED_MODEL` is a local path (default), skips HF pre-download
4. Recasts pod as a single-node trainer (Ray on port 6380 to avoid Node 0's 6379)
5. `exec bash examples/run_experiment_multinode_taiji.sh` with `EXPERIMENT=diffusion/sd3/sd3_text_rendering_grpo`

## Submit

```bash
taiji_client start -scfg train_sd3_text_rendering_2node.json
```

## Wandb signals to watch

- `rollout/reward_text_render_reward_mean` — GRPO training signal, should climb
- `rollout/reward_text_render_reward_std` — advantage variance, should be > 0
- `rollout/reward_n_errors_mean` — average error count, should drop over time
- `rollout/reward_parse_ok_mean` — judge JSON health, should stay > 0.95
