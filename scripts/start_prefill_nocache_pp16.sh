#!/bin/bash
# Kimi-K3 prefill, cache-off, *full* pipeline parallelism (PP16 / TP1).
#
# Sibling of start_prefill_nocache.sh, which is PP2 / TP8. Full PP removes the
# intra-stage TP all-reduce entirely; the trade is that the 93 layers must be
# split across 16 stages, so stage balance and the single cross-node PP hop
# (rank7 -> rank8) dominate throughput.
#
# Measured on 2x8 (bzz2, 400G NDR), variable-length prefill from a 256K trace
# (mean input 76K, 200 requests, aiakperf --fake-bootstrap), two runs each:
#   this file as-is                     61232 / 62014  -> 61623 tok/s
#   + SGLANG_PP_EAGER_BANK_FORWARD=1    63575 / 64190  -> 63882 tok/s  (+3.7%)
# The eager-bank switch is off by default and deliberately not exported here;
# turn it on explicitly once it has been validated on a target deployment.
#
# SGLANG_DYNAMIC_CHUNKING_COST_MODEL below needs luno-4962. Without it the
# export is a dead variable, dynamic chunking falls back to the smooth-factor
# heuristic, and the numbers above will not reproduce.
#
# Two constraints are load-bearing, do not change one without the other:
#
# 1. SGLANG_FLASHKDA_MAX_SEQ_LEN must track --chunked-prefill-size.
#    kda_flashkda.py compares each request's extend length against it and
#    silently drops all 69 KDA layers to the Triton chunk_kda path when
#    exceeded, which costs ~5%. Raising the chunk size alone measures as a
#    regression for this reason.
#
# 2. SGLANG_PP_LAYER_PARTITION prefix sums must avoid 8/24/52/68/84.
#    A DSPARK aux-hidden capture layer (0-based 7/23/51/67/83) landing on the
#    last layer of a stage hits output_attn_res_proj, which kimi_k3.py only
#    creates on the last PP rank -> AttributeError at startup. The partition
#    below also cuts the cross-node hop at layer 48, the largest value that
#    still keeps the attn-res bank at 4 rows (rows are appended at layers
#    where layer_idx % 12 == 0); moving it to 49 adds a 5th row to every
#    cross-node message and measures -3.8%.
set -uo pipefail

MASTER_IP=$(cut -d ',' -f 1 "$FED_MEMBER_FILE")
export MASTER_IP
MODEL_PATH="/Kimi-K3"
export MODEL_PATH
DRAFT_MODEL_PATH="/Kimi-K3-DSpark"
export DRAFT_MODEL_PATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NUM_NODES=2
export NUM_NODES
GPUS_PER_NODE=8
export TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Full PP: one stage per GPU, no tensor parallelism.
PP_SIZE=$((NUM_NODES * GPUS_PER_NODE))
export PP_SIZE

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

export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
export TRITON_CACHE_DIR=/root/.triton/cache

export NVSHMEM_HCA_LIST="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1"
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=130
export NVSHMEM_BOOTSTRAP=UID

export SGLANG_TBO_DEBUG=0
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export SGLANG_TOOL_STRICT_LEVEL=2
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=50
export SGLANG_DISAGGREGATION_QUEUE_SIZE=50
export SGLANG_FLASHKDA_MIN_SEQ_LEN=64
# keep in sync with --chunked-prefill-size (see header, constraint 1)
export SGLANG_FLASHKDA_MAX_SEQ_LEN=24576
# 93 layers over 16 stages; prefix sums avoid 8/24/52/68/84 and the cross-node
# cut sits at 48 (see header, constraint 2)
export SGLANG_PP_LAYER_PARTITION="7,6,6,6,6,6,6,5,6,6,6,6,5,6,5,5"
export SGLANG_WARMUP_PREFILL_LENS="512,2048,8192,32768,65536"
export SGLANG_WARMUP_BATCH_SIZE=1
export SGLANG_WARMUP_TIMEOUT=600
export SGLANG_HEALTH_CHECK_TIMEOUT=20

# Size each chunk by an analytic compute quota instead of a fixed token count,
# so a long-prefix request gets a smaller chunk and every chunk costs about the
# same as one --chunked-prefill-size chunk at zero prefix. Needed for
# variable-length traffic; the MFU constant cancels in the admission decision.
export SGLANG_DYNAMIC_CHUNKING_COST_MODEL=1

export HTTP_PORT=8000
export PD_SEPARATE_STAGE=prefill

export EMBEDDING_IB_DEVICE=mlx5_13
export SGLANG_MC_POOL_GB=24
IB_DEVICE=$(nvidia-smi topo -m | grep -v bond_0 | grep -oP '^\s*NIC\d+:\s+\K\S+' | paste -sd,)
export IB_DEVICE

export DEEPEP_CONFIG_JSON='{"normal_dispatch": {"num_sms": 32, "num_max_nvl_chunked_send_tokens": 36, "num_max_nvl_chunked_recv_tokens": 288, "num_max_rdma_chunked_send_tokens": 20, "num_max_rdma_chunked_recv_tokens": 128}, "normal_combine": {"num_sms": 32, "num_max_nvl_chunked_send_tokens": 4, "num_max_nvl_chunked_recv_tokens": 288, "num_max_rdma_chunked_send_tokens": 12, "num_max_rdma_chunked_recv_tokens": 128}}'

export AS_KDA_CHECKPOINT_GPU_COUNT=4
export AS_AGENT_ADDRESS="${HOST_IP}:7070"
export AS_SHARED_MEMORY_SIZE_BYTES=429496729600
export AIAK_HOST_CACHE_PORT=7070

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}/sgl-workspace/sglang/python"

PARALLEL_ARGS=(
    --pp-size "$PP_SIZE"
    --disable-overlap-schedule
    --enable-dynamic-chunking
)

MTP=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-dspark-block-size 3
)

MM_ARG=(
    --encoder-transfer-backend zmq_to_scheduler
    --enable-task-results-pull
    --enable-mooncake-in-zmq-mode
    --embedding-ib-device "$EMBEDDING_IB_DEVICE"
    --language-only
)

CACHE_OPT=(
    --disable-radix-cache
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
    --enable-strict-thinking \
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
    "${MM_ARG[@]}" \
    --attention-backend trtllm_mla \
    --linear-attn-prefill-backend flashkda \
    --mamba-full-memory-ratio 0.25 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --disable-flashinfer-autotune \
    --chunked-prefill-size $((24 * 1024)) \
    --max-prefill-tokens $((512 * 1024)) \
    --max-total-tokens 1050000 \
    --max-running-requests 64 \
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


