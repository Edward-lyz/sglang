FROM iregistry.baidu-int.com/hac-aiacc/sglang:v0.5.12-cu130-runtime

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


COPY aiak_sglang /tmp/aiak_sglang

RUN cat >/etc/apt/apt.conf.d/proxy.conf <<EOF
Acquire::http::Proxy "http://agent.baidu.com:8891/";
Acquire::https::Proxy "http://agent.baidu.com:8891/";
EOF

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN cd /tmp/aiak_sglang && find . -name "*.py" -exec cp --parents {} /sgl-workspace/sglang \; && \
    cp -R python/sglang/jit_kernel /sgl-workspace/sglang/python/sglang && \
    git log -1 --format="%H %s %an %ad" > /sgl-workspace/sglang/aiak-last-commit && \
    rm -rf /tmp/aiak_sglang


RUN pip install --upgrade tokenspeed-mla --break-system-packages --proxy http://agent.baidu.com:8891

RUN pip config set global.index-url http://mirrors.baidubce.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.baidubce.com

# Install deepgemm cache
RUN cd /root && wget http://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/cache_k26_0526.tar.gz && tar -xzvf cache_k26_0526.tar.gz && rm cache_k26_0526.tar.gz

# Install dependencies from configuration file
COPY aiak_sglang/dockerfile/gpu_requirements.env /tmp/gpu_requirements.env
RUN pip install --no-cache-dir --force-reinstall -r /tmp/gpu_requirements.env && \
    rm -rf /tmp/gpu_requirements.env

# NOTE: Move these into the basic image to minimize difference with the community.
RUN pip install jsonschema_rs
RUN pip install distro
RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/msit_llm-8.3.2-py3-none-any.whl

# Warmup flashinfer trtllm-gen fused MoE module to pre-create cubin symlinks.
# Avoids a FileExistsError race in ensure_symlink() when multiple TP workers
# trigger the JIT module load concurrently on the first container start.
RUN python3 -c "from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module; get_trtllm_moe_sm100_module()" || \
    echo "[warn] flashinfer trtllm-gen MoE warmup failed (non-SM100 build env?); will fall back to runtime init"

WORKDIR /sgl-workspace/sglang
