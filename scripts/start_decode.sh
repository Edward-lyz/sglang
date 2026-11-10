#!/bin/bash

set -euo pipefail

MODEL_PATH="/Kimi-K3"
export MODEL_PATH

DRAFT_MODEL_PATH="/Kimi-K3-DSpark"
export DRAFT_MODEL_PATH

MASTER_IP=$(cut -d ',' -f 1 "$FED_MEMBER_FILE")
export MASTER_IP

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NUM_NODES=2
export NUM_NODES
GPUS_PER_NODE=8
export TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

DP_SIZE=$NUM_NODES
export DP_SIZE

TP_SIZE=$TOTAL_GPUS
export TP_SIZE

EP_SIZE=$TOTAL_GPUS
export EP_SIZE

export MC_TE_METRIC=true
export MC_IB_PCI_RELAXED_ORDERING=1
export MC_NUM_QP_PER_EP=4
export MC_HANDSHAKE_MAX_LENGTH=10485760
export MC_MAX_EP_PER_CTX=3200

export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_DEBUG=INFO
export NCCL_CTA_POLICY=2
export NCCL_CUMEM_ENABLE=1

export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
export TRITON_CACHE_DIR=/root/.triton/cache

export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_HCA_LIST="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1"
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=130
export NVSHMEM_BOOTSTRAP=UID

export SGLANG_TBO_DEBUG=0
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export SGLANG_TOOL_STRICT_LEVEL=2
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=50
export SGLANG_DISAGGREGATION_QUEUE_SIZE=50
export SGLANG_K3_MLA_CELL_SIZE=13312
export SGLANG_K3_DSPARK_MHA_CELL_SIZE=10240
export SGLANG_K3_ONLY_LANGUAGE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_K3_DEL_MLA_KVB_DCP=1
export SGLANG_ENABLE_K3_MOE_WEIGHT_SHARD=1
export SGLANG_K3_MOE_WEIGHT_SHARD_LAYERS=1-92
export SGLANG_HEALTH_CHECK_TIMEOUT=20

export HTTP_PORT=8000
export PD_SEPARATE_STAGE=decode

export EMBEDDING_IB_DEVICE=mlx5_13
IB_DEVICE=$(nvidia-smi topo -m | grep -v bond_0 | grep -oP '^\s*NIC\d+:\s+\K\S+' | paste -sd,)
export IB_DEVICE

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/sgl-workspace/sglang/python"

LOG_FILE="${LOG_FILE:-/sgl-workspace/decode.log}"

PARALLEL_ARGS=(
    --tp-size "$TP_SIZE"
    --dp-size "$DP_SIZE"
    --dcp-size "$GPUS_PER_NODE"
    --ep-size "$EP_SIZE"
    --enable-dp-attention
    --enable-dp-lm-head
)

MTP_OPT=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-dspark-block-size 3
    --enable-linear-replayssm-spec
)

sglang serve \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --tokenizer-worker-num 8 \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --allow-auto-truncate \
    "${PARALLEL_ARGS[@]}" \
    --nnodes "$NUM_NODES" \
    --node-rank "$FED_POD_INDEX" \
    --dist-init-addr "$MASTER_IP:6676" \
    --dist-timeout 3600 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device mlx5_bond_0 \
    --disaggregation-zmq-ports 7000-7007 \
    --disaggregation-zmq-max-sockets 3000 \
    --disaggregation-decode-extra-slots 0 \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --moe-runner-backend deep_gemm \
    --attention-backend tokenspeed_mla \
    --mamba-full-memory-ratio 0.8 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --disable-flashinfer-autotune \
    --chunked-prefill-size -1 \
    --page-size 64 \
    --mem-fraction-static 0.85 \
    --max-running-requests $((224 * DP_SIZE)) \
    --cuda-graph-max-bs 224 \
    --cuda-graph-bs 1 2 4 8 16 24 32 40 48 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 \
    "${MTP_OPT[@]}" \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --linear-replayssm-cache-len 8 \
    --decode-log-interval 10 \
    --watchdog-timeout 3600 \
    --enable-metrics \
    --enable-cache-report \
    --collect-tokens-histogram
