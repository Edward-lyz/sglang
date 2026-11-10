. ~/.bashrc

export AIAK_DEBUG_SUBMODULE="0"
export AIAK_ENABLE_EXPERT_TRACE="0"
export AICK_BCCL_ALLTOALLV_ALGO=NVSHMEM
export AICK_NVSHMEM_BLOCKS="1"
export AICK_NVSHMEM_BUFFSIZE_MB="8192"
export AICK_NVSHMEM_MAX_TOKEN_NUM="16384"
export ASYNC_SEND_KV_CACHE="1"
export AllocExtendKernel_IMPL=cuda
export DISABLE_TP_WITH_SP="0"
export ENABLE_COPY_AND_SET_NATIVE="0"
export ENABLE_DEEP_DP="1"
export ENABLE_FLASHINFER_WARMUP="0"
export ENABLE_FP8_TRANS="0"
export ENABLE_FUSED_MOE_CPP_RUNTIME="1"
export ENABLE_MICRO_BATCH_OVERLAP="0"
export ENABLE_MTP_DECODER_OPT="1"
export ENABLE_ONLY_TP0_SEND_KVCACH="0"
export ENABLE_POST_PROC_CUDA="1"
export ENABLE_POST_REORDER_FUSION="1"
export ENABLE_SELECT_EXPERTS="0"
export ENABLE_SILU_AND_MUL_ONE_EXPERT="1"
export ENABLE_SORT_CHUNKS_BY_IDXS_CUDA="1"
export ENABLE_TP_WITH_CP_NSA="0"
export EP_STATS_DUMP_THRESH="0.95"
export GC_COLLECT_GEN_STEP="10000"
export GROUPED_GEMM_EXPECTED_M="4096"
export LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/x86_64-linux-gnu:/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}
export MOONCAKE_ASYNC_SEND="FALSE"
export MOONCAKE_META_SERVER_PASSWORD=test-deepseek
export MOONCAKE_META_SERVER_URL=redis://10.220.21.235:6379
export MOONCAKE_REMOTE_TP_SIZE="1"
export MOONCAKE_SEND_RECV_TIMEOUT="10"
export MOONCAKE_USE_RDMA="TRUE"
export NCCL_BUFFSIZE="524288"
export NCCL_DEBUG="INFO"
export NCCL_IB_ADAPTIVE_ROUTING="1"
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIMEOUT="22"
export NCCL_LL128_BUFFSIZE="524288"
export NCCL_MAX_NCHANNELS="1"
export NCCL_MIN_NCHANNELS="1"
export NCCL_NVLS_ENABLE="0"
export NCCL_SOCKET_IFNAME="lo"
export NVSHMEM_BOOTSTRAP="UID"
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="bond0"
export NVSHMEM_DISABLE_P2P="true"
export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
export NVSHMEM_IB_ENABLE_IBGDA="true"
export NVSHMEM_IB_GID_INDEX="3"
export NVSHMEM_IB_TRAFFIC_CLASS="130"
export NVSHMEM_IBGDA_NUM_RC_PER_PE="1"
export NVSHMEM_SYMMETRIC_HEAP_SIZE=32G
export PYTHONPATH=/workspace/aiak_sglang/python:$PYTHONPATH
export REDUNDANT_EXPERT_META_DIR=/workspace/ep_stats
export SELECT_EXPERTS_FREP="1"
export SET_DEEP_DP_MAX_TOKENS="384"
export SET_DEEP_DP_NUM_EXPERTS="256"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="256"
export SYNC_COMM="0"
export TRITON_CACHE_DIR=/root/triton_cache
export USE_CUDA_ROPE="1"
export USE_CUTLASS_FP8_BLOCK_GEMM="1"
export USE_FAST_A2A="1"
export USE_FUSED_INPUT_TO_FP8="1"
export USE_TORCH_NSA_TOPK="1"
export MC_MAX_EP_PER_CTX="3200"

export MASTER_IP=$(cut -d ',' -f 1 $FED_MEMBER_FILE)
export NUM_NODES=1
export DP_SIZE=$((NUM_NODES * 8))
export TP_SIZE=$((NUM_NODES * 8))

export MC_TE_METRIC=true
export MC_IB_PCI_RELAXED_ORDERING=1
export MC_NUM_QP_PER_EP=4
export MC_HANDSHAKE_MAX_LENGTH=10485760

export SGLANG_TBO_DEBUG=0
export SGLANG_MASKED_GEMM_FAST_ACT=1
export SGLANG_TOOL_STRICT_LEVEL="1"
export SGLANG_ENABLE_OPT_PREBUILT_BATCH=1
export SGLANG_LOG_MS=1
export SGLANG_NSA_TOPK_IMPL=cuda
export SGLANG_MTP_OPT_LOAD=1
export SGLANG_DEL_MLA_DEAD_WEIGHT=1
export SGLANG_ENABLE_MOE_UE8M0_CHUNKED_REQUANT=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_NSA_PAGED_MQA_LOGITS_MAX_CHUNK_MB=128

export DEEPEP_CONFIG_JSON='{"normal_dispatch": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 36, "num_max_nvl_chunked_recv_tokens": 256, "num_max_rdma_chunked_send_tokens": 8, "num_max_rdma_chunked_recv_tokens": 128}, "normal_combine": {"num_sms": 24, "num_max_nvl_chunked_send_tokens": 6, "num_max_nvl_chunked_recv_tokens": 256, "num_max_rdma_chunked_send_tokens": 8, "num_max_rdma_chunked_recv_tokens": 128}}'
export IB_DEVICE=$(nvidia-smi topo -m| grep -v bond_0 | grep -oP '^\s*NIC\d+:\s+\K\S+' | paste -sd,)
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_GLM51_MXFP4_MODE=1
export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=256
export SGLANG_OPT_MEGA_MOE_KEEP_NEW_SF_ONLY=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0

export SGLANG_OPT_USE_RMSNORM_QUANT_FUSED=1
export SGLANG_OPT_USE_SHARED_EXPERT_SILU_QUANT_FUSED=1
export SGLANG_OPT_MEGA_MOE_FUSED_TAIL=1
export SGLANG_OPT_USE_FUSED_QKV_NORM_QUANT=1
export SGLANG_OPT_USE_FUSED_RAGGED_INDEX_TRANSFORM=1
export SGLANG_OPT_USE_FUSED_KV_LATENT_PACK=1
export SGLANG_INDEXER_CACHE_OPT=1

[[ -v MODEL_PATH ]] || { echo "MODEL_PATH not exist" >&2; exit 1; }
[[ -v MODEL_ID ]] || { echo "MODEL_ID not exist" >&2; exit 1; }
[[ -v HTTP_PORT ]] || { echo "HTTP_PORT not exist" >&2; exit 1; }

echo "Model Path: $MODEL_PATH"
echo "Master IP: $MASTER_IP"

python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --log-level info \
    --model-path ${MODEL_PATH} \
    --port 8000 \
    --served-model-name $MODEL_ID \
    --trust-remote-code \
    --disaggregation-ib-device $IB_DEVICE \
    --disaggregation-mode decode \
    --disaggregation-zmq-max-sockets 3000 \
    --disaggregation-zmq-ports 7000-7007 \
    --prefill-round-robin-balance \
    --dist-init-addr ${MASTER_IP}:6676 \
    --dp-size $DP_SIZE \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --nnodes $NUM_NODES \
    --node-rank $FED_POD_INDEX \
    --tp-size $TP_SIZE \
    --disable-radix-cache \
    --kv-cache-dtype fp8_e4m3 \
    --max-running-requests $((64 * DP_SIZE)) \
    --mem-fraction-static 0.89 \
    --page-size 64 \
    --moe-runner-backend flashinfer_mxfp4 \
    --moe-a2a-backend megamoe \
    --moe-dense-tp-size 1 \
    --chunked-prefill-size -1 \
    --split-graph-list 2 14 78 \
    --attention-backend nsa \
    --nsa-decode-backend trtllm \
    --speculative-algorithm EAGLE \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-num-steps 3 \
    --cuda-graph-max-bs 64 \
    --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 \
    --decode-log-interval 10 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --num-reserved-decode-tokens 102 \
    --tokenizer-worker-num 32 \
    --watchdog-timeout 1000000 \
    --allow-auto-truncate \
    --collect-tokens-histogram \
    --enable-cache-report \
    --enable-metrics \
    --tool-call-parser glm47 \
    --reasoning-parser glm45