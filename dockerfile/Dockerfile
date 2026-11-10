FROM iregistry.baidu-int.com/hac-aiacc/aiak-inference-sglang:kimi-k3

## ===========================================================================================
# 这个是base是在社区这个commit v0.5.8 基础上打的镜像，作了如下修改：
# # 1. 增加了Bzz2卡上的deep_gemm预编译缓存。 编译命令如下
# # # 1.1. 下载cubins
# export {http,https}_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
# # 从 https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/
# # 下载指定版本hash的文件（会校验文件内容hash），
# # 到默认的位置 /root/.cache/flashinfer/cubins
# python3 -m flashinfer.__main__ --download-cubin
# # 1.2. 使用deepseek模型触发预编译. 编译的结果在 ~/.cache/deep_gemm/
# # 注意这里要取消代理
# unset http_proxy
# unset https_proxy
# python3 -m sglang.compile_deep_gemm --model $path_to_deepseek --tp 8 --trust-remote-code
# # 2. 更新了 xgrammar 版本 0.1.27 -> 0.1.30。命令如下：
# # # pip install xgrammar==0.1.30 --no-deps
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

# RUN pip install --upgrade tokenspeed-mla --break-system-packages --proxy http://agent.baidu.com:8891

# 改精确除法
RUN set -eux; \
    target=/usr/local/lib/python3.12/dist-packages/tokenspeed_mla/mla_decode_fp8.py; \
    python3 -c 'import sys; from pathlib import Path; p = Path("/usr/local/lib/python3.12/dist-packages/tokenspeed_mla/mla_decode_fp8.py"); old = "epi_scale = epilogue_params.output_scale * cute.arch.rcp_approx(row_sum)"; new = "epi_scale = epilogue_params.output_scale / row_sum"; text = p.read_text(); sys.exit("target line not found") if old not in text else p.write_text(text.replace(old, new, 1))'; \
    python3 -m py_compile "$target"

RUN pip config set global.index-url http://mirrors.baidubce.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.baidubce.com

# Install deepgemm cache
RUN cd /root && wget http://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/cache_k3_0728.tar.gz && tar -xzvf cache_k3_0728.tar.gz && rm cache_k3_0728.tar.gz

RUN cd /root && wget http://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/sglang-prestop-pyspy-dump.sh

# Install dependencies from configuration file
COPY aiak_sglang/dockerfile/gpu_requirements.env /tmp/gpu_requirements.env
RUN pip install --no-cache-dir --force-reinstall -r /tmp/gpu_requirements.env && \
    rm -rf /tmp/gpu_requirements.env

# NOTE: Move these into the basic image to minimize difference with the community.
RUN pip install jsonschema_rs
RUN pip install distro
RUN pip install decord
RUN pip install PyNvVideoCodec

# Python NVTX bindings for Nsight Systems application ranges.
RUN python3 -m pip install --no-cache-dir nvtx && \
    python3 -c 'import nvtx; nvtx.annotate("image-build-nvtx-check")(lambda: None)()'

# Warmup flashinfer trtllm-gen fused MoE module to pre-create cubin symlinks.
# Avoids a FileExistsError race in ensure_symlink() when multiple TP workers
# trigger the JIT module load concurrently on the first container start.
RUN python3 -c "from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module; get_trtllm_moe_sm100_module()" || \
    echo "[warn] flashinfer trtllm-gen MoE warmup failed (non-SM100 build env?); will fall back to runtime init"

# ci测试部分命令使用sglang命令
# SETUPTOOLS_SCM_PRETEND_VERSION keeps the version aligned with the base image.
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.5.15.post1 \
    pip install -e /sgl-workspace/sglang/python --no-deps --no-build-isolation

RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/msit_llm-8.3.2-py3-none-any.whl onnx --no-deps

WORKDIR /sgl-workspace/sglang
