#!/usr/bin/env bash

set -euo pipefail

export MODEL_ID="$MODEL_ID"

MASTER_IP=$(cut -d ',' -f 1 "$FED_MEMBER_FILE")
export MASTER_IP

export MODEL_PATH="/Kimi-K3-ATTN-MOE-DENSE-FP8"
export DRAFT_MODEL_PATH="/Kimi-K3-DSpark"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NUM_NODES=2
export GPUS_PER_NODE=8
export TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

export PP_SIZE=$((NUM_NODES * 8))
export TP_SIZE=1

export MC_HANDSHAKE_MAX_LENGTH=10485760
export MC_IB_PCI_RELAXED_ORDERING=1
export MC_MAX_EP_PER_CTX=3200
export MC_NUM_QP_PER_EP=4
export MC_TE_METRIC=true

export NCCL_DEBUG=INFO
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_IFNAME="bond0"

export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export TRITON_CACHE_DIR=/root/.triton/cache

export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_HCA_LIST="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1"
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=130

export SGLANG_DISAGGREGATION_QUEUE_SIZE=50
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=50
export SGLANG_DYNAMIC_CHUNKING_COST_MODEL=1
export SGLANG_ENABLE_PP_WEIGHT_FILTER=1
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_FLASHKDA_MAX_SEQ_LEN=24576
export SGLANG_FLASHKDA_MIN_SEQ_LEN=64
export SGLANG_OPT_FLASH_KDA_CUTE=1
export SGLANG_HEALTH_CHECK_TIMEOUT=20
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_K3_ATTN_MOE_FP8=1
export SGLANG_K3_KDA_SHORT_CONV=1
export SGLANG_MAX_KV_CHUNK_CAPACITY=65536
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export SGLANG_OPT_USE_FLATROW_CONCAT_MHA_K=1
export SGLANG_OPT_USE_K3_KDA_FUSED_BETA_SIGMOID=1
export SGLANG_OPT_USE_K3_KDA_FUSED_TRACK_CONV_STATE=1
export SGLANG_OPT_USE_KDA_STRIDED_ONORM=1
export SGLANG_OPT_USE_TRITON_MLA_QKV_FP8_CAST=1
export SGLANG_PARALLEL_SCHEDULER_LAUNCH=1
export SGLANG_PP_EAGER_BANK_FORWARD=1
export SGLANG_PP_EAGER_BANK_RANKS="7"
export SGLANG_PP_LAYER_PARTITION="7,5,7,5,7,5,7,5,7,5,7,5,7,5,6,3"
export SGLANG_PP_PD_KV_TRANSFER_TARGET_PARALLEL=1
export SGLANG_PP_PD_PREFILL_EARLY_LAST_CHUNK=1
export SGLANG_PP_PD_PREFILL_SIDEBAND_BOOTSTRAP_CONSENSUS=1
export SGLANG_TBO_DEBUG=0
export SGLANG_TOOL_STRICT_LEVEL=2
export SGLANG_WARMUP_BATCH_SIZE=1
export SGLANG_WARMUP_PREFILL_LENS="512,2048,8192,32768,65536"
export SGLANG_WARMUP_TIMEOUT=600
export SGLANG_PP_PD_PREFILL_EARLY_STREAM_OUTPUT=1
export SGLANG_IDLE_METRIC_HEARTBEAT=1

export HTTP_PORT=8000
export PD_SEPARATE_STAGE=prefill

export EMBEDDING_IB_DEVICE=mlx5_13
export SGLANG_MC_POOL_GB=24
IB_DEVICE=$(nvidia-smi topo -m | grep -v bond_0 | grep -oP '^\s*NIC\d+:\s+\K\S+' | paste -sd,)
export IB_DEVICE

export DEEPEP_CONFIG_JSON='{"normal_dispatch": {"num_sms": 32, "num_max_nvl_chunked_send_tokens": 36, "num_max_nvl_chunked_recv_tokens": 288, "num_max_rdma_chunked_send_tokens": 20, "num_max_rdma_chunked_recv_tokens": 128}, "normal_combine": {"num_sms": 32, "num_max_nvl_chunked_send_tokens": 4, "num_max_nvl_chunked_recv_tokens": 288, "num_max_rdma_chunked_send_tokens": 12, "num_max_rdma_chunked_recv_tokens": 128}}'

export AIAK_HOST_CACHE_PORT=7070
export AS_AGENT_ADDRESS="${HOST_IP}:7070"
export AS_KDA_CHECKPOINT_GPU_COUNT=4
export AS_READ_MAX_BLOCKS=0
export AS_SHARED_MEMORY_SIZE_BYTES=429496729600 # 400GiB
export AS_WRITE_MAX_BLOCKS=16

export PYTHONPATH="/sgl-workspace/sglang/python:${PYTHONPATH:-}"
LOG_TO_FILE="${LOG_TO_FILE:-0}"
LOG_FILE="${LOG_FILE:-/sgl-workspace/prefill.log}"

PARALLEL_ARGS=(
    --tp-size "$TP_SIZE"
    --pp-size "$PP_SIZE"
    --disable-overlap-schedule
)

MTP=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-dspark-block-size 4
)

MM_ARG=(
    --encoder-transfer-backend zmq_to_scheduler
    --enable-task-results-pull
    --enable-mooncake-in-zmq-mode
    --embedding-ib-device "$EMBEDDING_IB_DEVICE"
    --language-only
)

CACHE_OPT=(
    --per-node-gpu-num 8
    --enable-asradix-cache
)

if [[ "${PROFILE_NSYS:-0}" == "1" ]]; then
    export SGLANG_ENABLE_NVTX_SCHEDULER=1
    nsys start --session-new=sglprof \
        -c cudaProfilerApi \
        --capture-range-end=repeat:3:sync \
        -o "/sgl-workspace/ascache_pp${FED_POD_INDEX}"
    NSYS_PREFIX="nsys launch --session=sglprof -t cuda,nvtx,osrt --cuda-graph-trace=node --trace-fork-before-exec=true --cuda-memory-usage=false"
else
    NSYS_PREFIX=""
fi

(
    if [[ "$LOG_TO_FILE" == "1" ]]; then
        exec >"$LOG_FILE" 2>&1
    fi

    $NSYS_PREFIX sglang serve \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        --model-path "$MODEL_PATH" \
        --served-model-name "$MODEL_ID" \
        --tokenizer-worker-num 8 \
        --reasoning-parser kimi_k3 \
        --tool-call-parser kimi_k3 \
        --grammar-backend llguidance \
        --allow-auto-truncate \
        "${PARALLEL_ARGS[@]}" \
        --nnodes "$NUM_NODES" \
        --node-rank "$FED_POD_INDEX" \
        --dist-init-addr "$MASTER_IP:6676" \
        --dist-timeout 3600 \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend mooncake \
        --disaggregation-ib-device mlx5_bond_0 \
        --disaggregation-bootstrap-port 8200 \
        --disaggregation-zmq-ports 7000-7007 \
        --disaggregation-zmq-max-sockets 3000 \
        "${MM_ARG[@]}" \
        --attention-backend trtllm_mla \
        --moe-runner-backend flashinfer_mxfp4 \
        --linear-attn-prefill-backend flashkda \
        --mamba-full-memory-ratio 0.25 \
        --max-mamba-cache-size 512 \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
        --disable-flashinfer-autotune \
        --chunked-prefill-size $((16 * 1024)) \
        --max-prefill-tokens $((512 * 1024)) \
        --max-total-tokens $((4 * 1024 * 1024)) \
        --max-running-requests 128 \
        --pp-async-batch-depth 2 \
        --page-size 64 \
        --mem-fraction-static 0.8 \
        "${MTP[@]}" \
        "${CACHE_OPT[@]}" \
        --kv-cache-dtype fp8_e4m3 \
        --watchdog-timeout 3600 \
        --enable-metrics \
        --enable-cache-report \
        --collect-tokens-histogram
    # --ep-dispatch-algorithm fake
) &
