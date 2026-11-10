FROM iregistry.baidu-int.com/lyz/lmsysorg/sglang:dev-dsv41

## ===========================================================================================
# 这个 base 是社区 lmsysorg/sglang:dev-dsv41 镜像搬到 iregistry 上的副本，里面已经带了
# DeepSeek V4.1 需要的 kernel（FlashMLA v4.1 的 fp8/fp4 paged 布局、DSA top-k v2）与
# 对应的 torch / sgl-kernel / flashinfer，所以这一层只做两件事：把 aiak_sglang 换进去，
# 再补厂内环境需要的依赖。
#
# 和 kimi-k3 交付镜像相比，这里刻意少了两步：
# 1. tokenspeed_mla 的精确除法 patch。那个包只在 aiak-inference-sglang 基础镜像里，
#    社区镜像里没有，patch 会直接失败。
# 2. kimi_k3 的 deep_gemm 预编译缓存。缓存是按 K3 编的，对 DSv4.1 无效。要给 DSv4.1
#    做一份就用
#      python3 -m sglang.compile_deep_gemm --model $path_to_dsv41 --tp 8 --trust-remote-code
#    编出 ~/.cache/deep_gemm，打包上传后在这里加一段 wget + tar。
## ===========================================================================================


RUN rm -rf /sgl-workspace/sglang
COPY aiak_sglang /sgl-workspace/sglang

RUN cat >/etc/apt/apt.conf.d/proxy.conf <<EOF
Acquire::http::Proxy "http://agent.baidu.com:8891/";
Acquire::https::Proxy "http://agent.baidu.com:8891/";
EOF

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Keep every shell helper under scripts/ directly executable in the image.
# New scripts are picked up automatically without another Dockerfile edit.
RUN find /sgl-workspace/sglang/scripts -type f -name '*.sh' -exec chmod +x {} +

RUN pip config set global.index-url http://mirrors.baidubce.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.baidubce.com

# PaaS prestop 钩子：容器退出前抓一次 py-spy 栈。
RUN cd /root && wget http://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/sglang-prestop-pyspy-dump.sh

# Install dependencies from configuration file
COPY aiak_sglang/dockerfile/gpu_requirements.env /tmp/gpu_requirements.env
RUN pip install --no-cache-dir --force-reinstall -r /tmp/gpu_requirements.env && \
    rm -rf /tmp/gpu_requirements.env

# PD 分离的传输后端，cuda13 build 对齐基础镜像。
RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/mooncake_transfer_engine_cuda13-0.3.12.post1-cp312-cp312-manylinux_2_28_x86_64.whl

# NOTE: Move these into the basic image to minimize difference with the community.
RUN pip install jsonschema_rs
RUN pip install distro
RUN pip install decord
RUN pip install PyNvVideoCodec

# Warmup flashinfer trtllm-gen fused MoE module to pre-create cubin symlinks.
# Avoids a FileExistsError race in ensure_symlink() when multiple TP workers
# trigger the JIT module load concurrently on the first container start.
RUN python3 -c "from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module; get_trtllm_moe_sm100_module()" || \
    echo "[warn] flashinfer trtllm-gen MoE warmup failed (non-SM100 build env?); will fall back to runtime init"

# ci测试部分命令使用sglang命令
# 版本号跟基础镜像装的那个保持一致，避免这一层把 sglang 的版本号改掉。
RUN set -eux; \
    base_version="$(python3 -c 'import importlib.metadata as m; print(m.version("sglang"))')"; \
    SETUPTOOLS_SCM_PRETEND_VERSION="$base_version" \
        pip install -e /sgl-workspace/sglang/python --no-deps --no-build-isolation

RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/msit_llm-8.3.2-py3-none-any.whl onnx --no-deps

WORKDIR /sgl-workspace/sglang
