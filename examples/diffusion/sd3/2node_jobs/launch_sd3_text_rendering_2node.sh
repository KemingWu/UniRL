#!/bin/bash
# =============================================================================
# SD3.5-Medium GRPO + Remote Text-Rendering Judge (Qwen3-VL-MoE-235B) — 2-Node
#
#   Node 0 (INDEX=0): TextRenderingJudge reward service — 8 GPUs, vLLM TP=8.
#                     The 235B judge reads (prompt, generated_image) and emits
#                     a JSON list of text-rendering errors; the scorer converts
#                     n_errors → scalar reward = exp(-α·n_errors).
#   Node 1 (INDEX=1): SD3.5-Medium GRPO training — 8 GPUs, FSDP + LoRA.
#                     Reaches the judge over HTTP via REWARD_SERVICE_URL.
#
# Environment (set in taiji job JSON):
#   EXPERIMENT=diffusion/sd3/sd3_text_rendering_grpo
#   JUDGE_CKPT_PATH=<abs path to Qwen3-VL-MoE-235B judge HF dir>
#   PRETRAINED_MODEL=<abs path or HF id for SD3.5-Medium>
#   WANDB_API_KEY / WANDB_RUN_NAME / HF_TOKEN (optional for gated models)
# =============================================================================

set -uo pipefail

echo "[BOOT] Script started at $(date). PID=$$"
echo "[BOOT] INDEX=${INDEX:-unset} CHIEF_IP=${CHIEF_IP:-unset} LOCAL_IP=${LOCAL_IP:-unset}"
echo "[BOOT] HOST_NUM=${HOST_NUM:-unset} HOST_GPU_NUM=${HOST_GPU_NUM:-unset}"

# ========== Proxy ==========
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# ========== Mount CephFS ==========
echo "[MOUNT] Downloading jizhi_client & taiji_client ..."
wget -q http://jizhi.oa.com/jizhi_client_golang/jizhi_client -O /usr/bin/jizhi_client && chmod +x /usr/bin/jizhi_client
wget -q http://jizhi.oa.com/taiji_client_golang/taiji_client -O /usr/bin/taiji_client && chmod +x /usr/bin/taiji_client
echo "[MOUNT] Installing ceph-fuse ..."
curl -s -o /etc/yum.repos.d/ceph_el7.repo http://gaia.repo.oa.com/ceph_el7.repo
yum clean metadata -q && yum -y -q remove ceph-fuse && yum -y -q install ceph-fuse3
ln -sf /usr/bin/ceph-fuse3 /usr/bin/ceph-fuse
mkdir -p /etc/ceph
cat << 'EOF' > /etc/ceph/ceph.conf
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

echo "[MOUNT] Mounting CephFS regions ..."
taiji_client mount -bf TaiJi_HYAide_800H20 -tk "${TAIJI_TOKEN:-<TAIJI_TOKEN>}"
taiji_client mount -l qy4 "${CEPHFS_USER_ROOT:-<CEPHFS_USER_ROOT>}"
taiji_client mount -tk "${TAIJI_MLLM_TOKEN:-<TAIJI_MLLM_TOKEN>}" -bf TaiJi_HYAide_MLLM_TJ_TJ_H20 -l BJ
# zw = 中卫 region → apdcephfs_zwfy8 (Qwen3-VL-MoE-235B judge weights live here)
taiji_client mount -tk "${ZWFY8_TOKEN:-<ZWFY8_TOKEN>}" -bf TaiJi_HYAide_800H20 -l zw
echo "[MOUNT] All mounts done."

# ========== HuggingFace login ==========
export HF_TOKEN="${HF_TOKEN:-<FILL_IN_HF_TOKEN>}"
huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>/dev/null || true

# ========== Repo paths ==========
DIFFRL_ROOT="${CEPHFS_USER_ROOT:-<CEPHFS_USER_ROOT>}/workspace/My_Code/UniRL"
REWARD_SERVICE_ROOT="${DIFFRL_ROOT}/unirl-reward-service"

if [ ! -d "${DIFFRL_ROOT}" ]; then
    echo "FATAL: Repo not found at ${DIFFRL_ROOT} after mount."; exit 1
fi
cd "${DIFFRL_ROOT}"

# ========== Python env ==========
export PYTHONDONTWRITEBYTECODE=1
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    new_conf=$(echo "${PYTORCH_CUDA_ALLOC_CONF}" | tr ',' '\n' | grep -v 'expandable_segments:True' | paste -sd, -)
    if [ -n "${new_conf}" ]; then export PYTORCH_CUDA_ALLOC_CONF="${new_conf}"; else unset PYTORCH_CUDA_ALLOC_CONF; fi
fi
find "${DIFFRL_ROOT}/unirl" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

if [ -f /root/diffusionrl/.venv/bin/activate ]; then
    source /root/diffusionrl/.venv/bin/activate
    echo "[SETUP] Activated image venv: ${VIRTUAL_ENV}"
fi

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --quiet uv || true
fi

# Common runtime deps
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

# ========== Wandb ==========
export WANDB_API_KEY="${WANDB_API_KEY:-<FILL_IN_WANDB_API_KEY>}"
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-true}"
export WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-unirl-sd3-text-rendering}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-sd3_text_rendering_grpo}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
wandb login "${WANDB_API_KEY}" --relogin --host=https://api.bandw.top 2>/dev/null || true

# ========== Node role ==========
NODE_RANK="${INDEX:-0}"
if [ -n "${LOCAL_IP:-}" ]; then MY_IP="${LOCAL_IP}"
elif [ -n "${CHIEF_IP:-}" ]; then MY_IP="${CHIEF_IP}"
else MY_IP="$(hostname -I | awk '{print $1}')"
fi
if [ -n "${CHIEF_IP:-}" ]; then REWARD_NODE_IP="${CHIEF_IP}"; else REWARD_NODE_IP="${MY_IP}"; fi
REWARD_SERVICE_PORT="${REWARD_SERVICE_PORT:-8080}"
export REWARD_SERVICE_URL="http://${REWARD_NODE_IP}:${REWARD_SERVICE_PORT}"
echo "[ROLE] NODE_RANK=${NODE_RANK}, MY_IP=${MY_IP}, REWARD_SERVICE_URL=${REWARD_SERVICE_URL}"

if [ "${NODE_RANK}" = "0" ]; then
    # ================================================================
    # NODE 0: Text-Rendering Judge reward service (Qwen3-VL-MoE-235B, vLLM TP=8)
    # ================================================================
    echo "[NODE0] Starting text-rendering judge service (Qwen3-VL-MoE-235B, TP=8)..."

    # Verify judge ckpt exists (fail fast if the CephFS region isn't mounted).
    JUDGE_CKPT_PATH="${JUDGE_CKPT_PATH:-/apdcephfs_zwfy8/share_305110755/zehanwang/verl/checkpoints/sftv4_qwen3vl_235b_train50k_f1_exp_error_judge_30B_woKL_12roll_64n_20260623_0023/global_step_100/actor/huggingface}"
    export JUDGE_CKPT_PATH
    if [ ! -f "${JUDGE_CKPT_PATH}/config.json" ]; then
        echo "[NODE0] FATAL: judge config.json not found at ${JUDGE_CKPT_PATH}/config.json"
        echo "[NODE0]        (mount apdcephfs_zwfy8, or point JUDGE_CKPT_PATH at a copy in a mounted region)"
        exit 1
    fi
    echo "[NODE0] Judge ckpt OK at ${JUDGE_CKPT_PATH}"

    cd "${REWARD_SERVICE_ROOT}"

    # Install reward service (editable). --ignore-requires-python because the
    # pyproject pins py>=3.12 but the vLLM stack has broad compatibility;
    # the runtime code we hit works on the base venv's Python either way.
    uv pip install --no-deps --ignore-requires-python -e . 2>&1 | tail -3 \
        || pip install --no-deps --ignore-requires-python -e . 2>&1 | tail -3

    # vLLM must support Qwen3VLMoeForConditionalGeneration. Upgrade if the
    # base image is too old (Qwen3-VL-MoE support landed in vLLM 0.6.x+).
    python3 -c "import vllm; print('[NODE0] vllm', vllm.__version__)" || {
        uv pip install --quiet vllm 2>&1 | tail -3 || pip install --quiet vllm 2>&1 | tail -3
    }
    python3 -c "
import vllm
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('${JUDGE_CKPT_PATH}', trust_remote_code=True)
print(f'[NODE0] vllm={vllm.__version__}, judge arch={cfg.architectures}')
" || { echo "[NODE0] FATAL: vllm / judge config not loadable"; exit 1; }

    ray stop >/dev/null 2>&1 || true
    ray start --head --node-ip-address="${MY_IP}" --port=6379 --num-gpus=8
    echo "[NODE0] Ray head started on ${MY_IP}:6379 with 8 GPUs"

    # ── Patch judge tokenizer_config.json (compat fix) ────────────────
    # The judge ckpt was trained with an older transformers where
    # ``extra_special_tokens`` accepted a list; current transformers expects
    # a dict. The 14 special tokens are ALREADY registered in tokenizer.json
    # (tokenizers-lib native format), so dropping the duplicate field is safe.
    # Build a lightweight mirror of the ckpt dir under /dev/shm with all
    # files symlinked except tokenizer_config.json (which we rewrite).
    PATCHED_CKPT="/dev/shm/judge_patched"
    rm -rf "${PATCHED_CKPT}"
    mkdir -p "${PATCHED_CKPT}"
    for f in "${JUDGE_CKPT_PATH}"/*; do
        b="$(basename "$f")"
        [ "$b" = "tokenizer_config.json" ] && continue
        ln -s "$f" "${PATCHED_CKPT}/$b"
    done
    python3 - <<PYEOF
import json
src = "${JUDGE_CKPT_PATH}/tokenizer_config.json"
dst = "${PATCHED_CKPT}/tokenizer_config.json"
with open(src) as f:
    d = json.load(f)
# The 14 special tokens are already in tokenizer.json's added_tokens; the
# ``extra_special_tokens`` list here is a legacy duplicate. Current transformers
# expects this field to be a dict; drop it entirely.
if "extra_special_tokens" in d:
    print(f"[NODE0] dropping extra_special_tokens field ({len(d['extra_special_tokens'])} entries) from tokenizer_config.json")
    del d["extra_special_tokens"]
with open(dst, "w") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)
print(f"[NODE0] wrote patched tokenizer_config.json -> {dst}")
PYEOF
    export JUDGE_CKPT_PATH="${PATCHED_CKPT}"
    echo "[NODE0] Using patched ckpt mirror: ${JUDGE_CKPT_PATH}"

    # Substitute the checked-in yaml's __JUDGE_CKPT_PATH__ placeholder — the
    # reward_service config loader uses plain yaml.safe_load (no OmegaConf
    # ${oc.env:...} resolution).
    RESOLVED_CFG="/tmp/text_rendering_judge_service_resolved.yaml"
    sed "s|__JUDGE_CKPT_PATH__|${JUDGE_CKPT_PATH}|g" configs/text_rendering_judge_service.yaml > "${RESOLVED_CFG}"
    echo "[NODE0] Wrote resolved service config to ${RESOLVED_CFG}"

    echo "[NODE0] Launching reward_service ..."
    exec python -m reward_service --config "${RESOLVED_CFG}"

else
    # ================================================================
    # NODE 1: SD3.5-Medium GRPO training (FSDP + LoRA, 8 GPUs)
    # ================================================================
    echo "[NODE1] Starting SD3.5-Medium GRPO training..."
    echo "[NODE1] Will connect to reward service at ${REWARD_SERVICE_URL}"

    cd "${DIFFRL_ROOT}"

    if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
        pip install --no-deps -e . 2>/dev/null || uv pip install --no-deps -e . 2>/dev/null || true
    fi

    echo "[NODE1] Waiting for reward service /health ..."
    for i in $(seq 1 240); do   # up to 20 min: judge takes long to load 96 shards
        if curl -s --max-time 5 "${REWARD_SERVICE_URL}/health" >/dev/null 2>&1; then
            echo "[NODE1] Reward service is ready! (attempt ${i})"
            break
        fi
        if [ "${i}" = "240" ]; then
            echo "[NODE1] WARNING: reward service not ready after 20min; proceeding anyway"
        fi
        sleep 5
    done

    # SD3.5-Medium is already at a local CephFS path by default in the recipe;
    # nothing to pre-download.
    if [[ "${PRETRAINED_MODEL:-}" && "${PRETRAINED_MODEL}" != /* ]]; then
        : "${HF_HOME:=/dev/shm/hf_cache}"
        export HF_HOME HF_HUB_CACHE="${HF_HOME}/hub"
        mkdir -p "${HF_HUB_CACHE}"
        echo "[NODE1] PRETRAINED_MODEL is an HF id (${PRETRAINED_MODEL}); pre-downloading..."
        python3 - <<EOF || echo "[NODE1] WARNING: pre-download failed"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${PRETRAINED_MODEL}", cache_dir="${HF_HUB_CACHE}", max_workers=4)
EOF
    fi

    # Recast this pod as a single-node trainer (its own Ray on port 6380).
    export INDEX=0 NODE_RANK=0 NUM_NODES=1
    export GPUS_PER_NODE="${HOST_GPU_NUM:-8}"
    export HEAD_IP="${MY_IP}" CHIEF_IP="${MY_IP}"
    export EXPERIMENT="${EXPERIMENT:-diffusion/sd3/sd3_text_rendering_grpo}"
    export MASTER_PORT=$((29500 + RANDOM % 1000))
    export RAY_PORT=6380

    exec bash examples/run_experiment_multinode_taiji.sh \
        "logging.run_name=${WANDB_RUN_NAME}" \
        "logging.report_to_wandb=${REPORT_TO_WANDB}" \
        ${EXTRA_HYDRA_OVERRIDES:-}
fi
