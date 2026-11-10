#!/bin/bash
# ============================================================================
# GLM W4 Prefill 节点启动脚本
#
# 用法:
#   bash start_glm_w4_prefill.sh [选项]
#
# 选项:
#   --no-cache    关闭 prefix cache。默认使用 --enable-asradix-cache，
#                 传入该参数后替换为 --disable-radix-cache。
#   --bench-mode  压测/基准测试模式，额外传入 --ep-dispatch-algorithm fake。
#   -h, --help    显示本帮助信息并退出。
#
# 依赖的环境变量:
#   MODEL_PATH        模型权重路径 (必填)
#   MODEL_ID          served-model-name (必填)
#   HTTP_PORT         服务监听端口 (必填)
#   FED_MEMBER_FILE   集群成员列表文件，取第一项作为 MASTER_IP
#   FED_POD_INDEX     当前节点 rank (传给 --node-rank)
#
# 示例:
#   bash start_glm_w4_prefill.sh
#   bash start_glm_w4_prefill.sh --no-cache
#   bash start_glm_w4_prefill.sh --bench-mode
#   bash start_glm_w4_prefill.sh --no-cache --bench-mode
# ============================================================================

usage() {
    cat <<'EOF'
GLM W4 Prefill 节点启动脚本

用法:
  bash start_glm_w4_prefill.sh [选项]

选项:
  --no-cache    关闭 prefix cache。默认使用 --enable-asradix-cache，
                传入该参数后替换为 --disable-radix-cache。
  --bench-mode  压测/基准测试模式，额外传入 --ep-dispatch-algorithm fake。
  -h, --help    显示本帮助信息并退出。

依赖的环境变量:
  MODEL_PATH        模型权重路径 (必填)
  MODEL_ID          served-model-name (必填)
  HTTP_PORT         服务监听端口 (必填)
  FED_MEMBER_FILE   集群成员列表文件，取第一项作为 MASTER_IP
  FED_POD_INDEX     当前节点 rank (传给 --node-rank)

示例:
  bash start_glm_w4_prefill.sh
  bash start_glm_w4_prefill.sh --no-cache
  bash start_glm_w4_prefill.sh --bench-mode
  bash start_glm_w4_prefill.sh --no-cache --bench-mode
EOF
}

CACHE_ARGS=(--enable-asradix-cache)
BENCH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-cache)
            CACHE_ARGS=(--disable-radix-cache)
            ;;
        --bench-mode)
            BENCH_ARGS=(--ep-dispatch-algorithm fake)
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "警告: 未知参数 '$arg'，已忽略。使用 --help 查看用法。" >&2
            ;;
    esac
done

echo "Prefix cache: ${CACHE_ARGS[*]}"
echo "Bench mode: ${BENCH_ARGS[*]:-disabled}"

. ~/.bashrc

export AIAK_ENABLE_EXPERT_TRACE="0"
export ASYNC_SEND_KV_CACHE="TRUE"
export AllocExtendKernel_IMPL=cuda
export DEEP_DP_COMBINE_CONFIG=(24, 36, 256, 8, 128)
export DISABLE_TP_WITH_SP="1"
export ENABLE_DEEP_DP="1"
export ENABLE_DEEP_EP_COMBINE_CONVERT_CUDA="true"
export ENABLE_DEEP_EP_DISPATCH_CONVERT_CUDA="true"
export ENABLE_FLASHINFER_WARMUP="0"
export ENABLE_MICRO_BATCH_OVERLAP="0"
export ENABLE_ONLY_TP0_SEND_KVCACH="0"
export ENABLE_PREFILL_OVERLAP="0"
export ENABLE_SELECT_EXPERTS="0"
export ENABLE_SILU_AND_MUL_ONE_EXPERT="1"
export ENABLE_SORT_CHUNKS_BY_IDXS_CUDA="1"
export ENABLE_TP_WITH_CP_NSA="0"
export ENABLE_TRITON_WARMUP="1"
export ENABLE_W8A8_BLOCK_FP8_GROUPED_GEMM_TMA="FALSE"
export ENABLE_W8A8_BLOCK_FP8_GROUPED_GEMM_TUNED="1"
export EP_STATS_DUMP_THRESH="0.95"
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/x86_64-linux-gnu:/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}
export NCCL_DEBUG="INFO"
export NCCL_IB_ADAPTIVE_ROUTING="1"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIMEOUT="22"
export NCCL_NVLS_ENABLE="0"
export NCCL_SOCKET_IFNAME="lo"
export NVSHMEM_BOOTSTRAP="UID"
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="bond0"
export NVSHMEM_DISABLE_P2P="true"
export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NVSHMEM_IB_GID_INDEX="3"
export NVSHMEM_IB_TRAFFIC_CLASS="130"
export PD_SEPARATE_STAGE=prefill
export PER_TOKEN_GROUP_QUANT_FP8_CUDA="true"
export PYTHONPATH=/workspace/aiak_sglang/python:$PYTHONPATH
export REDUNDANT_EXPERT_META_DIR=/workspace/ep_stats
export SELECT_EXPERTS_FREP="1"
export SET_DEEP_DP_NUM_EXPERTS="256"
export TRITON_CACHE_DIR=/root/triton_cache
export USE_SPEEDGATE_COMPUTE_POSITION="1"
export USE_SPEEDGATE_REQ_TO_TOKEN_POOL="1"
export USE_TORCH_NSA_TOPK="1"

export MASTER_IP=$(cut -d ',' -f 1 $FED_MEMBER_FILE)
export NUM_NODES=1
export DP_SIZE=1
export ATTN_TP_SIZE=1

export MC_TE_METRIC=true
export MC_IB_PCI_RELAXED_ORDERING=1
export MC_NUM_QP_PER_EP=4
export MC_HANDSHAKE_MAX_LENGTH=10485760

export SGLANG_TBO_DEBUG=0
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=50
export SGLANG_DISAGGREGATION_QUEUE_SIZE=50
export SGLANG_MASKED_GEMM_FAST_ACT=1
export SGLANG_TOOL_STRICT_LEVEL="1"
export SGLANG_MOONCAKE_SEND_AUX_TCP=1
export SGLANG_LOG_MS=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
export SGLANG_NSA_THRESHOLD_MIN_KVLEN=8192
export SGLANG_NSA_TOPK_IMPL=cuda_stable

export DEEPEP_CONFIG_JSON='{"normal_dispatch": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 36, "num_max_nvl_chunked_recv_tokens": 256, "num_max_rdma_chunked_send_tokens": 8, "num_max_rdma_chunked_recv_tokens": 128}, "normal_combine": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 36, "num_max_nvl_chunked_recv_tokens": 256, "num_max_rdma_chunked_send_tokens": 8, "num_max_rdma_chunked_recv_tokens": 128}}'
export IB_DEVICE=$(nvidia-smi topo -m| grep -v bond_0 | grep -oP '^\s*NIC\d+:\s+\K\S+' | paste -sd,)
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_GLM51_MXFP4_MODE=1
export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=10240
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

export SGLANG_OPT_USE_RMSNORM_QUANT_FUSED=1
export SGLANG_OPT_USE_SHARED_EXPERT_SILU_QUANT_FUSED=1
export SGLANG_OPT_MEGA_MOE_FUSED_TAIL=1
export SGLANG_OPT_USE_FUSED_QKV_NORM_QUANT=1
export SGLANG_OPT_USE_FUSED_RAGGED_INDEX_TRANSFORM=1
export SGLANG_OPT_USE_FUSED_KV_LATENT_PACK=1
export SGLANG_INDEXER_CACHE_OPT=1

export AS_LOAD_ALLGATHER_BATCH=10
export AS_READ_MAX_BLOCKS=96
export AS_WRITE_MAX_BLOCKS=16

echo "Model Path: $MODEL_PATH"
echo "Master IP: $MASTER_IP"

[[ -v MODEL_PATH ]] || { echo "MODEL_PATH not exist" >&2; exit 1; }
[[ -v MODEL_ID ]] || { echo "MODEL_ID not exist" >&2; exit 1; }
[[ -v HTTP_PORT ]] || { echo "HTTP_PORT not exist" >&2; exit 1; }

export TORCHINDUCTOR_COMPILE_THREADS=1
python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --log-level info \
    --model-path $MODEL_PATH \
    --port $HTTP_PORT \
    --served-model-name $MODEL_ID \
    --trust-remote-code \
    --disaggregation-bootstrap-port 8200 \
    --disaggregation-ib-device $IB_DEVICE \
    --disaggregation-mode prefill \
    --disaggregation-zmq-ports 7000-7007 \
    --dist-init-addr $MASTER_IP:6676 \
    --dp-size 1 \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --nnodes $NUM_NODES \
    --node-rank $FED_POD_INDEX \
    --per-node-gpu-num 8 \
    --tp-size 8 \
    --disable-cuda-graph \
    "${CACHE_ARGS[@]}" \
    ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"} \
    --max-prefill-tokens 196608 \
    --max-running-requests 256 \
    --max-total-tokens 1056768 \
    --mem-fraction-static 0.835 \
    --page-size 64 \
    --moe-runner-backend flashinfer_mxfp4 \
    --moe-a2a-backend megamoe \
    --moe-dense-tp-size 1 \
    --attention-backend nsa \
    --enable-nsa-prefill-context-parallel \
    --nsa-prefill-backend trtllm \
    --nsa-prefill-cp-mode "round-robin-split" \
    --speculative-algorithm EAGLE \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-num-steps 3 \
    --chunked-prefill-size 65536 \
    --decode-log-interval 1 \
    --disable-overlap-schedule \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --tokenizer-worker-num 32 \
    --allow-auto-truncate \
    --collect-tokens-histogram \
    --enable-cache-report \
    --enable-metrics \
    --tool-call-parser glm47 \
    --reasoning-parser glm45