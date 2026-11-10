FROM iregistry.baidu-int.com/zhangyanzhe/sglang-test:v0.5.17-cu130 as base

## ===========================================================================================
# 这个是base是在社区这个commit v0.5.8 基础上打的镜像。
# 1. 增加了Bzz2卡上的deep_gemm预编译缓存。 编译命令如下
# # 1. 下载cubins
# export {http,https}_proxy=http://gzbh-aip-paddlecloud140.gzbh:8128
# # 从 https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/
# # 下载指定版本hash的文件（会校验文件内容hash），
# # 到默认的位置 /root/.cache/flashinfer/cubins
# python3 -m flashinfer.__main__ --download-cubin
# # 2. 使用deepseek模型触发预编译. 编译的结果在 ~/.cache/deep_gemm/
# # 注意这里要取消代理
# unset http_proxy
# unset https_proxy
# python3 -m sglang.compile_deep_gemm --model $path_to_deepseek --tp 8 --trust-remote-code
# 2. 基础镜像内置 flash_mla 0b475cc；当前镜像覆盖安装 5b200a9(cu13)，支持 DSV4 head_num=64 Decode 及 400B MXFP4 KV cache(c4)。
## ===========================================================================================


COPY aiak_sglang /tmp/aiak_sglang

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN cp -Rf /tmp/aiak_sglang/scripts /sgl-workspace/sglang/
RUN cp -Rf /tmp/aiak_sglang/benchmark /sgl-workspace/sglang/

RUN cd /tmp/aiak_sglang && find . -name "*.py" -exec cp --parents {} /sgl-workspace/sglang \; && \
    cp -R python/sglang/kernels/jit /sgl-workspace/sglang/python/sglang/kernels && \
    cp -R python/sglang/jit_kernel /sgl-workspace/sglang/python/sglang && \
    git log -1 --format="%H %s %an %ad" > /sgl-workspace/sglang/aiak-last-commit && \
    git log -1 --format="%h" > /sgl-workspace/sglang/.aiak-commit-short && \
    rm -rf /tmp/aiak_sglang

RUN pip config set global.index-url http://mirrors.baidubce.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.baidubce.com

# Re-install sglang as editable so the dist-info (and sglang.__version__) records
# our internal version instead of the base image's 0.0.0.dev0 fallback.
RUN AIAK_VERSION="0.5.8+aiak.g$(cat /sgl-workspace/sglang/.aiak-commit-short)" && \
    echo "Installing sglang as ${AIAK_VERSION}" && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${AIAK_VERSION}" \
        pip install --no-deps --force-reinstall -e /sgl-workspace/sglang/python && \
    pip show sglang | head -3 && \
    rm -f /sgl-workspace/sglang/.aiak-commit-short

# Install dependencies from configuration file
COPY aiak_sglang/dockerfile/gpu_requirements.env /tmp/gpu_requirements.env
RUN pip install --no-cache-dir --force-reinstall -r /tmp/gpu_requirements.env && \
    rm -rf /tmp/gpu_requirements.env


RUN pip install --force-reinstall py-spy
RUN pip install xgrammar==0.2.1 --no-deps
RUN pip install jsonschema_rs --no-deps
RUN pip uninstall deep_ep -y
RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/deep_ep/develop/bak/deep_ep-1.2.1+8f2a57d-cp312-cp312-linux_x86_64.whl
RUN pip uninstall deep_gemm -y
RUN rm -r /usr/local/lib/python3.12/dist-packages/deep_gemm
RUN pip install --no-cache-dir --force-reinstall https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/DeepGemm/develop/torch2.11.0/deep_gemm-2.5.0+4400f34-py3-none-any.whl
# c4(cache-4bit): cu13 FlashMLA (DSV4 400B MXFP4 sparse decode) 从 BOS 安装。
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps 'https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/FlashMLA/develop/torch2.11.0/flash_mla-1.0.0+5b200a9-cp312-cp312-linux_x86_64.whl'

# c4(cache-4bit): overlay the FlashMLA DSV4 400B MXFP4 sparse-decode bridge onto
# the installed sgl_kernel wrapper (the repo copy under kernels/aot is NOT the
# module imported at runtime as sgl_kernel.flash_mla). Resolve the install dir via
# sysconfig purelib (no `import sgl_kernel`, which fails/empties during a GPU-less
# build and would silently copy the bridge to /flash_mla.py). Assert it landed.
RUN SGLK="$(python3 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_paths()["purelib"],"sgl_kernel"))')" && \
    cp /sgl-workspace/sglang/python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py "$SGLK/flash_mla.py" && \
    grep -q is_mxfp4_kvcache "$SGLK/flash_mla.py"


# Install operator JIT cache (deep_gemm / tvm-ffi / flashinfer / triton / torchinductor)
# glm53_qbo_nvfp4_bzz2_cache_1031_v2: 以 cache_glm5.1_0620_merged 为基础（旧包文件是其子集、同路径内容一致），
# 叠加 GLM-5.3 qbo-NVFP4 权重 + v15.4.8.5 1P1D(MTP7, P: CP8 asradix, D: DP8 cuda graph) 经 AS 链路
# glm-53-cache-1026 数据集压测后 P/D 两侧运行时编译的产物。包内路径为 root/.cache、root/triton_cache、
# tmp/torchinductor_root，必须 -C / 解压。线上 YAML 设置 TRITON_CACHE_DIR=/root/triton_cache，这里同步为镜像默认值。
RUN wget -nv -O /tmp/cache.tar.gz http://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/glm53_qbo_nvfp4_bzz2_cache_1031_v2.tar.gz && \
    rm -rf /root/.cache /root/triton_cache /tmp/torchinductor_root && \
    tar -xzf /tmp/cache.tar.gz -C / && \
    rm -f /tmp/cache.tar.gz
ENV TRITON_CACHE_DIR=/root/triton_cache

RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/msit_llm-8.3.2-py3-none-any.whl

# msit_llm pins NumPy below 2, while the runtime SciPy/CuPy stack requires
# NumPy 2.x. Re-pin after installing msit_llm to keep the engine importable.
RUN pip install --no-cache-dir --force-reinstall --no-deps numpy==2.3.5

# Cutlass DSL 4.6 calls the TVM-FFI dataclass mapping API introduced in 0.1.11.
# Keep this pin after all accelerator wheel installs, including deep_gemm, which
# declares the older 0.1.9 requirement.
RUN pip install --no-cache-dir --force-reinstall --no-deps apache-tvm-ffi==0.1.11

WORKDIR /sgl-workspace/sglang
