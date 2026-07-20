#!/bin/bash
# =============================================================================
# SD3.5-M DiffusionOPD + Remote GenEval Reward — 2-Node Launcher
#
# 2-node SPMD job on taiji:
#   Node 0 (INDEX=0): GenEval reward service (Mask2Former + CLIP, 1 GPU)
#   Node 1 (INDEX=1): SD3.5-M DiffusionOPD training (8 GPUs, FSDP trainside)
#
# The trainer's PerDomainRewardScorer holds LOCAL PickScore + OCR scorers and
# a REMOTE geneval scorer pointing at node 0's HTTP service. Each rollout
# batch is single-domain (round-robin from MultiTeacherRLDataSource), so only
# GenEval batches hit the network.
#
# Environment (set in the taiji job JSON):
#   EXPERIMENT=diffusion/sd3/sd3_opd_pickscore
#   PRETRAINED_MODEL=stabilityai/stable-diffusion-3.5-medium
#   HF_TOKEN=...
#   WANDB_API_KEY=...
#   WANDB_RUN_NAME=...
# =============================================================================

set -uo pipefail

echo "[BOOT] Script started at $(date). PID=$$"
echo "[BOOT] INDEX=${INDEX:-unset} CHIEF_IP=${CHIEF_IP:-unset} LOCAL_IP=${LOCAL_IP:-unset}"
echo "[BOOT] HOST_NUM=${HOST_NUM:-unset} HOST_GPU_NUM=${HOST_GPU_NUM:-unset}"
echo "[BOOT] EXPERIMENT=${EXPERIMENT:-diffusion/sd3/sd3_opd_pickscore}"

# ========== Mount storage ==========
export http_proxy=http://9.21.0.122:11113
export https_proxy=http://9.21.0.122:11113
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

echo "[MOUNT] Downloading jizhi_client & taiji_client ..."
wget -q http://jizhi.oa.com/jizhi_client_golang/jizhi_client -O /usr/bin/jizhi_client && chmod +x /usr/bin/jizhi_client
wget -q http://jizhi.oa.com/taiji_client_golang/taiji_client -O /usr/bin/taiji_client && chmod +x /usr/bin/taiji_client
echo "[MOUNT] Installing ceph-fuse ..."
curl -s -o /etc/yum.repos.d/ceph_el7.repo http://gaia.repo.oa.com/ceph_el7.repo
yum clean metadata -q
yum -y -q remove ceph-fuse
yum -y -q install ceph-fuse3
ln -sf /usr/bin/ceph-fuse3 /usr/bin/ceph-fuse
mkdir -p /etc/ceph
cat << EOF > /etc/ceph/ceph.conf
[client]
  client_not_support_security = true
  client_setuid_optimize = true
  fuse_fake_tmp_agent = true
  client_trash_enabled = false
  client_reconnect_stale = true
  fuse_attr_timeout = 0
  client_cache_size = 30000
  client_die_on_failed_dentry_invalidate = false
  fuse_set_user_groups = false
  fuse_clone_fd = false
  objecter_timeout_check_v2 = true
  objecter_max_osd_sessions = 80
EOF

echo "[MOUNT] Mounting CephFS (1/3) ..."
taiji_client mount -bf TaiJi_HYAide_800H20 -tk <TAIJI_TOKEN>
echo "[MOUNT] Mounting CephFS (2/3) — private ..."
taiji_client mount -l qy4 <CEPHFS_USER_ROOT>
echo "[MOUNT] Mounting CephFS (3/3) — shared ..."
taiji_client mount -tk <TAIJI_MLLM_TOKEN> -bf TaiJi_HYAide_MLLM_TJ_TJ_H20 -l BJ
echo "[MOUNT] All mounts done."

# ========== HuggingFace login ==========
export HF_TOKEN="${HF_TOKEN:-<HF_TOKEN>}"
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true
echo "[SETUP] HF login done."

# ========== Repo paths ==========
# ``CEPHFS_USER_ROOT`` is your personal CephFS mount point (e.g.
# ``/apdcephfs_hldy/private_<user>``). Fill in by exporting the env var, or
# substitute the placeholder directly before submitting the job.
DIFFRL_ROOT="<CEPHFS_USER_ROOT>/workspace/My_Code/UniRL"
REWARD_SERVICE_ROOT="${DIFFRL_ROOT}/unirl-reward-service"

if [ ! -d "${DIFFRL_ROOT}" ]; then
    echo "FATAL: Repo not found at ${DIFFRL_ROOT} after mount."
    exit 1
fi

cd "${DIFFRL_ROOT}"

# ========== Python environment ==========
export PYTHONDONTWRITEBYTECODE=1

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    new_conf=$(echo "${PYTORCH_CUDA_ALLOC_CONF}" | tr ',' '\n' \
        | grep -v 'expandable_segments:True' | paste -sd, -)
    if [ -n "${new_conf}" ]; then
        export PYTORCH_CUDA_ALLOC_CONF="${new_conf}"
    else
        unset PYTORCH_CUDA_ALLOC_CONF
    fi
fi

find "${DIFFRL_ROOT}/unirl" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

if [ -f /root/diffusionrl/.venv/bin/activate ]; then
    source /root/diffusionrl/.venv/bin/activate
    echo "[SETUP] Activated image venv: ${VIRTUAL_ENV}"
fi

export PYTHONPATH="${DIFFRL_ROOT}:${PYTHONPATH:-}"

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --quiet uv || true
fi

# Ensure common runtime deps (both nodes need these)
for pkg in "omegaconf>=2.3" "hydra-core>=1.3"; do
    name="${pkg%%[<>=!~]*}"
    case "${name}" in
        hydra-core) mod=hydra ;;
        *)          mod="${name//-/_}" ;;
    esac
    if ! python3 -c "import ${mod}" 2>/dev/null; then
        uv pip install --quiet "${pkg}" || pip install --quiet "${pkg}" || true
    fi
done

echo "[SETUP] Python: $(which python3)"
echo "[SETUP] Ray:    $(which ray 2>/dev/null || echo NOT_FOUND)"

# ========== Wandb ==========
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-unirl-sd3-opd}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sd3_opd_2node}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
wandb login "${WANDB_API_KEY}" --relogin --host=https://api.bandw.top 2>/dev/null || true

# ========== Determine node role ==========
NODE_RANK="${INDEX:-0}"

# Resolve this node's IP
if [ -n "${LOCAL_IP:-}" ]; then
    MY_IP="${LOCAL_IP}"
elif [ -n "${CHIEF_IP:-}" ]; then
    MY_IP="${CHIEF_IP}"
else
    MY_IP="$(hostname -I | awk '{print $1}')"
fi

# CHIEF_IP = Node 0 IP; on node 0 itself may be unset.
if [ -n "${CHIEF_IP:-}" ]; then
    REWARD_NODE_IP="${CHIEF_IP}"
else
    REWARD_NODE_IP="${MY_IP}"
fi

REWARD_SERVICE_PORT="${REWARD_SERVICE_PORT:-8080}"
export REWARD_SERVICE_URL="http://${REWARD_NODE_IP}:${REWARD_SERVICE_PORT}"

echo "[ROLE] NODE_RANK=${NODE_RANK}, MY_IP=${MY_IP}, REWARD_SERVICE_URL=${REWARD_SERVICE_URL}"

if [ "${NODE_RANK}" = "0" ]; then
    # ================================================================
    # NODE 0: GenEval Reward Service (Mask2Former + CLIP, py3.10 conda env)
    # ================================================================
    #
    # The image is py3.12; mmdet 2.x/mmcv-full 1.7.2 (the versions the paper
    # uses, and the ones UniRL RewardService's ``envs/geneval.txt`` pins to)
    # only support py3.8-3.10. We stand up a py3.10 conda env just for this
    # node's reward service. The trainer on Node 1 stays in the base py3.12
    # venv and reaches this service over HTTP.
    #
    # Reference: unirl-reward-service/envs/geneval.txt header comment.
    echo "[NODE0] Starting GenEval reward service (py3.10 conda + mmdet 2.28.2)..."

    # --- Install miniconda into /dev/shm (fast, ephemeral, per-pod) ---
    CONDA_ROOT="/dev/shm/miniconda3"
    if [ ! -x "${CONDA_ROOT}/bin/conda" ]; then
        echo "[NODE0] Installing miniconda to ${CONDA_ROOT} ..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "${CONDA_ROOT}" || {
            echo "[NODE0] FATAL: miniconda install failed"; exit 1; }
    fi
    # Deactivate any parent venv so conda activate cleanly takes over.
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        deactivate 2>/dev/null || true
        unset VIRTUAL_ENV
    fi
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"

    # --- Create py3.10 env for the mmdet 2.x stack ---
    GENEVAL_ENV="geneval310"
    if ! conda env list | awk '{print $1}' | grep -qx "${GENEVAL_ENV}"; then
        echo "[NODE0] Creating conda env ${GENEVAL_ENV} (python=3.10) ..."
        conda create -y -n "${GENEVAL_ENV}" python=3.10 || {
            echo "[NODE0] FATAL: conda env create failed"; exit 1; }
    fi
    conda activate "${GENEVAL_ENV}"
    echo "[NODE0] Active python: $(which python) — $(python --version)"

    # --- Install torch 2.1 + mmcv-full 1.7.2 + mmdet 2.28.2 + accessories ---
    if ! python -c "import mmdet, mmcv, ray" 2>/dev/null; then
        echo "[NODE0] Installing mmdet 2.x stack (this takes ~5-8 min) ..."
        # setuptools >= 80 removed ``pkg_resources``, but torch 2.1.0 and
        # open_clip still ``from pkg_resources import packaging``. conda's
        # bundled setuptools may already be 80+, so DOWNGRADE it explicitly
        # before installing anything else. Range 60–69 is old enough to keep
        # pkg_resources but new enough to build modern wheels.
        pip install --quiet "setuptools>=60,<70" wheel || {
            echo "[NODE0] FATAL: setuptools pin failed"; exit 1; }
        echo "[NODE0] setuptools=$(python -c 'import setuptools; print(setuptools.__version__)')"
        python -c "import pkg_resources; print('[NODE0] pkg_resources OK')" || {
            echo "[NODE0] FATAL: pkg_resources still missing after setuptools pin"; exit 1; }

        # Pin numpy<2 BEFORE torch: torch 2.1.0 was built against numpy 1.x,
        # its C ABI won't load a numpy 2 array (``_ARRAY_API not found``).
        pip install --quiet "numpy==1.26.4" || {
            echo "[NODE0] FATAL: numpy 1.26 install failed"; exit 1; }
        # System nvcc on this image is CUDA 12.9. torch's build extension
        # checks torch.version.cuda vs the nvcc's major version — cu118 vs
        # 12.9 hard-fails. Use cu121 (torch.version.cuda == '12.1', same
        # major as 12.9), which passes the check with only a minor warning.
        pip install --quiet torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 || {
            echo "[NODE0] FATAL: torch 2.1 cu121 install failed"; exit 1; }
        pip install --quiet mmengine
        # H20 = Hopper (sm_90). mmcv-full 1.7.2 prebuilt wheels only ship
        # sm_60~sm_86 SASS, so ``ms_deformable_im2col_cuda`` (used by
        # Mask2Former) fails with "no kernel image is available for execution
        # on the device" on H20. Build from source with sm_90 in the arch list.
        # This needs nvcc from a CUDA toolkit matching the image's torch build.
        if ! command -v nvcc >/dev/null 2>&1; then
            echo "[NODE0] FATAL: nvcc not on PATH — cannot source-build mmcv-full for H20 sm_90"
            exit 1
        fi
        echo "[NODE0] nvcc: $(nvcc --version | tail -1)"
        echo "[NODE0] Building mmcv-full 1.7.2 from source with TORCH_CUDA_ARCH_LIST including 9.0 (H20/Hopper) ..."
        # ``--no-build-isolation`` is CRITICAL: pip's default isolated build env
        # bootstraps a fresh setuptools (80+) which removed ``pkg_resources``,
        # and mmcv-full 1.7.2's setup.py imports pkg_resources at build time.
        # We need setup.py to see the conda env's setuptools (pinned to <70).
        # This also requires torch/numpy to already be installed in the base env
        # so setup.py can ``import torch`` for CUDA arch detection.
        pip install --quiet ninja
        TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0+PTX" \
        MMCV_WITH_OPS=1 \
        FORCE_CUDA=1 \
        MAX_JOBS=8 \
            pip install --no-build-isolation --no-binary mmcv-full mmcv-full==1.7.2 || {
            echo "[NODE0] FATAL: mmcv-full 1.7.2 source build failed"; exit 1; }
        python -c "import mmcv; print(f'[NODE0] mmcv built: {mmcv.__version__} at {mmcv.__file__}')"
        pip install --quiet mmdet==2.28.2 || {
            echo "[NODE0] FATAL: mmdet 2.28.2 install failed"; exit 1; }
        pip install --quiet "open_clip_torch>=2.20" "clip_benchmark>=1.4"
        # clip_benchmark pulls in transformers; latest transformers (>=4.46)
        # requires torch >= 2.4 and self-disables on our torch 2.1 with the
        # "PyTorch was not found" warning. Pin an older transformers that
        # still supports torch 2.1 (only the tokenizer/config path is used by
        # CLIP zero-shot color classification here).
        pip install --quiet "transformers>=4.40,<4.45"
        pip install --quiet "ray[default]==2.32.0" fastapi uvicorn pillow pydantic PyYAML omegaconf requests || {
            echo "[NODE0] FATAL: ray/fastapi install failed"; exit 1; }
        # click >= 8.2 introduced a ``Sentinel`` type that ray 2.32's CLI
        # can't deepcopy in add_command_alias — throws ValueError at every
        # ``ray`` CLI invocation. Downgrade click so ``ray start``/``ray stop``
        # don't spew tracebacks (functionally ray start still succeeded, but
        # anything that calls the CLI subprocess would fail).
        pip install --quiet "click>=8.1,<8.2"
        # Re-pin numpy and setuptools in case any downstream install upgraded them.
        pip install --quiet "numpy==1.26.4" "setuptools>=60,<70"
        python -c "import pkg_resources; import numpy; print(f'[NODE0] pkg_resources OK, numpy={numpy.__version__}')" || {
            echo "[NODE0] FATAL: pkg_resources/numpy final check failed"; exit 1; }
    fi

    # --- Install reward_service editable (bypass py>=3.12 requirement) ---
    # RewardService's pyproject.toml pins ``requires-python = ">=3.12"``, but
    # its geneval scorer needs py3.10 for the mmdet 2.x stack. Skip the
    # version check — the runtime code paths we use are py3.10-compatible.
    pip install --ignore-requires-python --no-deps -e "${REWARD_SERVICE_ROOT}" 2>&1 | tail -3 || {
        echo "[NODE0] FATAL: reward_service editable install failed"; exit 1; }

    # --- Verify imports ---
    python -c "
import mmdet, mmcv, ray, open_clip
from clip_benchmark.metrics import zeroshot_classification
from reward_service.scorers.geneval import GenEvalScorer
print(f'[NODE0] mmdet={mmdet.__version__}, mmcv={mmcv.__version__}, ray={ray.__version__}')
" || { echo "[NODE0] FATAL: post-install imports failed"; exit 1; }

    # --- Resolve mmdet-shipped config path for Mask2Former ---
    MMDET_INSTALL="$(python -c 'import mmdet, os; print(os.path.dirname(mmdet.__file__))')"
    MMDET_CFG=""
    for _cand in \
        "${MMDET_INSTALL}/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
        "$(dirname ${MMDET_INSTALL})/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"; do
        if [ -f "${_cand}" ]; then MMDET_CFG="${_cand}"; break; fi
    done
    if [ -z "${MMDET_CFG}" ]; then
        echo "[NODE0] FATAL: cannot find mask2former config under mmdet install ${MMDET_INSTALL}"; exit 1
    fi
    echo "[NODE0] mask2former config: ${MMDET_CFG}"

    # --- Download mask2former ckpt (~155MB, first-time only) ---
    CKPT_DIR="/dev/shm/geneval"
    mkdir -p "${CKPT_DIR}"
    CKPT="${CKPT_DIR}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth"
    if [ ! -f "${CKPT}" ]; then
        echo "[NODE0] Downloading mask2former ckpt ..."
        wget -q -c "https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth" -O "${CKPT}.partial" \
            && mv "${CKPT}.partial" "${CKPT}" || {
            echo "[NODE0] FATAL: mask2former ckpt download failed"; exit 1; }
    fi
    ls -lh "${CKPT}"

    # --- Write resolved service config (mmdet paths must be absolute) ---
    RESOLVED_CFG="/tmp/geneval_service_resolved.yaml"
    cat > "${RESOLVED_CFG}" <<EOF
server:
  host: 0.0.0.0
  port: ${REWARD_SERVICE_PORT}
  score_timeout_s: 300.0

rewards:
  - name: geneval
    scorer: geneval
    runtime_env: envs/geneval.txt
    num_replicas: 1
    num_gpus: 1
    num_cpus: 4
    max_concurrency: 1
    params:
      mmdet_config: "${MMDET_CFG}"
      mmdet_checkpoint: "${CKPT}"
      clip_arch: ViT-L-14
      clip_pretrained: openai
      score_type: score
      threshold: 0.3
      counting_threshold: 0.9
      max_objects: 16
      nms_threshold: 1.0
      position_threshold: 0.1
      device: cuda
EOF
    echo "[NODE0] Wrote resolved service config to ${RESOLVED_CFG}"

    # --- Start Ray head (single-node reward cluster) ---
    ray stop >/dev/null 2>&1 || true
    ray start --head --node-ip-address="${MY_IP}" --port=6379 --num-gpus=1
    echo "[NODE0] Ray started on ${MY_IP}:6379"

    # --- Launch reward service ---
    cd "${REWARD_SERVICE_ROOT}"
    echo "[NODE0] Launching reward_service ..."
    exec python -m reward_service --config "${RESOLVED_CFG}"

else
    # ================================================================
    # NODE 1: SD3.5-M DiffusionOPD Training
    # ================================================================
    echo "[NODE1] Starting SD3.5-M DiffusionOPD training..."
    echo "[NODE1] Will connect to reward service at ${REWARD_SERVICE_URL}"

    cd "${DIFFRL_ROOT}"

    # Editable install of UniRL (fast, no-deps)
    if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
        pip install --no-deps -e . 2>/dev/null || uv pip install --no-deps -e . 2>/dev/null || true
    fi

    # ---- Wait for reward service ----
    echo "[NODE1] Waiting for reward service /health ..."
    for i in $(seq 1 120); do
        if curl -s --max-time 5 "${REWARD_SERVICE_URL}/health" >/dev/null 2>&1; then
            echo "[NODE1] Reward service is ready! (attempt ${i})"
            break
        fi
        if [ "${i}" = "120" ]; then
            echo "[NODE1] WARNING: reward service not ready after 10min; proceeding anyway"
        fi
        sleep 5
    done

    # ---- Pre-download SD3.5-M base model (per-node) ----
    if [[ "${PRETRAINED_MODEL:-}" && "${PRETRAINED_MODEL}" != /* ]]; then
        : "${HF_HOME:=/dev/shm/hf_cache}"
        export HF_HOME
        HF_HUB_CACHE="${HF_HOME}/hub"
        export HF_HUB_CACHE
        mkdir -p "${HF_HUB_CACHE}"
        echo "[NODE1] Pre-downloading ${PRETRAINED_MODEL} -> ${HF_HUB_CACHE} ..."
        python3 - <<EOF || echo "[NODE1] WARNING: pre-download failed; per-actor retry may follow"
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

repo_id = "${PRETRAINED_MODEL}"
cache_dir = "${HF_HUB_CACHE}"
last_err = None
for attempt in range(1, 7):
    try:
        path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, max_workers=4)
        print(f"  OK: {path}")
        break
    except HfHubHTTPError as e:
        last_err = e
        backoff = min(60, 5 * attempt)
        print(f"  attempt {attempt}: {e}; retry in {backoff}s")
        time.sleep(backoff)
else:
    raise last_err if last_err else RuntimeError("snapshot_download failed silently")
EOF
        echo "[NODE1] Pre-download done."
    fi

    # ---- Recast this pod as a single-node trainer (its own Ray on port 6380) ----
    # Node 0 owns Ray port 6379 (for its reward service); training gets 6380.
    export INDEX=0
    export NODE_RANK=0
    export NUM_NODES=1
    export GPUS_PER_NODE="${HOST_GPU_NUM:-8}"
    export HEAD_IP="${MY_IP}"
    export CHIEF_IP="${MY_IP}"
    export EXPERIMENT="${EXPERIMENT:-diffusion/sd3/sd3_opd_pickscore}"
    export MASTER_PORT=$((29500 + RANDOM % 1000))
    export RAY_PORT=6380

    exec bash examples/run_experiment_multinode_taiji.sh \
        "logging.run_name=${WANDB_RUN_NAME}" \
        "logging.report_to_wandb=${REPORT_TO_WANDB}" \
        ${EXTRA_HYDRA_OVERRIDES:-}
fi
