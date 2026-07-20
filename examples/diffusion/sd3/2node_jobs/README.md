# SD3.5-M DiffusionOPD — 2-Node Deployment Reference

This directory holds a **redacted reference copy** of the taiji job JSON and
launcher script used to run this recipe on a 2-node cluster where the
classical GenEval reward scorer (mmdet Mask2Former + CLIP color) is served
remotely from Node 0 and the trainer runs on Node 1.

The actual deployment lives under
`<CEPHFS_USER_ROOT>/workspace/jobs/reproduce_scripts/jobs/`
on the shared CephFS mount; these copies are for code-review and are kept in
lockstep with the recipe (`../sd3_opd_pickscore.yaml`).

## Topology

| Node | GPUs used | Role |
|---|---|---|
| 0 (`INDEX=0`) | 1 (of 8) | GenEval reward service (Ray head on 6379 + Mask2Former + CLIP ViT-L/14) |
| 1 (`INDEX=1`) | 8 | SD3.5-M DiffusionOPD training (FSDP trainside, Ray on 6380) |

Node 0 spins up a fresh **Python 3.10 miniconda env** just for the mmdet 2.x /
mmcv-full 1.7.2 stack (paper-aligned versions incompatible with the base
image's py3.12); the trainer on Node 1 stays in the image's py3.12 venv and
reaches the reward service over HTTP via `REWARD_SERVICE_URL`.

## Placeholders to fill

Both files carry placeholders where site-specific / personal / secret values
belong. Substitute before submitting:

| Placeholder | Meaning |
|---|---|
| `<CEPHFS_USER_ROOT>` | Your personal CephFS mount root (e.g. `/apdcephfs_<region>/private_<user>`) |
| `<CEPHFS_SHARED_ROOT>` | Shared CephFS root that hosts the SD3.5-M weights (e.g. `/apdcephfs_<region>/share_<team>`) |
| `<TAIJI_TOKEN>` | Main taiji cluster token |
| `<TAIJI_MLLM_TOKEN>` | CephFS BJ region mount token |
| `<HF_TOKEN>` | HuggingFace token (base env fallback in the launcher) |
| `<WANDB_API_KEY>` | wandb login |

## Submit

```bash
taiji_client start -scfg train_sd3_opd_mopd_roundrobin_2node.json
```

## What the launcher does (Node 0)

1. Install miniconda to `/dev/shm/miniconda3` (per-pod, ephemeral)
2. `conda create -n geneval310 python=3.10`
3. Pin `setuptools>=60,<70` (keeps `pkg_resources` — needed by torch 2.1 + mmcv setup.py)
4. `pip install numpy==1.26.4 torch==2.1.0 (cu121)`
5. `pip install mmengine`
6. **Build `mmcv-full==1.7.2` from source** with `TORCH_CUDA_ARCH_LIST` including `9.0+PTX` — the openmmlab prebuilt wheels only cover sm_60~sm_86, but H20 is Hopper (sm_90)
7. `pip install mmdet==2.28.2 open_clip_torch clip_benchmark`
8. `pip install "transformers>=4.40,<4.45"` (transformers ≥4.46 wants torch ≥2.4 → self-disables here)
9. `pip install "ray[default]==2.32.0" fastapi uvicorn ...`
10. `pip install "click>=8.1,<8.2"` (avoid Ray CLI `Sentinel` deepcopy bug in click 8.2+)
11. `pip install --ignore-requires-python --no-deps -e ${REWARD_SERVICE_ROOT}` (bypasses `requires-python=">=3.12"`)
12. Auto-download `mask2former_swin-s-p4-w7-224...743b7d99.pth` to `/dev/shm/geneval/`
13. Write resolved `configs/geneval_service.yaml` with absolute mmdet config + ckpt paths
14. `ray start --head --port=6379 --num-gpus=1`
15. `exec python -m reward_service --config /tmp/geneval_service_resolved.yaml`

## What the launcher does (Node 1)

1. `pip install --no-deps -e .` (editable UniRL)
2. Poll `${REWARD_SERVICE_URL}/health` until 200
3. **Skip** HF pre-download (`PRETRAINED_MODEL` is a local CephFS path, not an HF repo id)
4. `exec bash examples/run_experiment_multinode_taiji.sh` with `RAY_PORT=6380`, `INDEX=0`, `NUM_NODES=1` (treats this pod as a single-node trainer with its own Ray cluster on 6380 — Node 0's Ray on 6379 is separate)
