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
# 2. 基础镜像内置 flash_mla 0b475cc，当前镜像会覆盖安装 9d77d8a，以支持 DSV4 head_num=64 Decode。
## ===========================================================================================


COPY aiak_sglang /tmp/aiak_sglang

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN cd /tmp/aiak_sglang && find . -name "*.py" -exec cp --parents {} /sgl-workspace/sglang \; && \
    cp -R python/sglang/kernels/jit /sgl-workspace/sglang/python/sglang/kernels && \
    git log -1 --format="%H %s %an %ad" > /sgl-workspace/sglang/aiak-last-commit && \
    rm -rf /tmp/aiak_sglang

RUN pip config set global.index-url http://mirrors.baidubce.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.baidubce.com

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
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    'https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/FlashMLA/develop/torch2.9.1/flash_mla-1.0.0+9d77d8a-cp312-cp312-linux_x86_64.whl'

# Install deepgemm cache
RUN cd /root && wget https://ai-data.bj.bcebos.com/mae/cache_glm5.1_0620_merged.tar.gz  && tar -xzvf cache_glm5.1_0620_merged.tar.gz && rm /root/.cache -rf && mv cache_glm5.1_0620_merged/.cache /root/.cache && rm cache_glm5.1_0620_merged.tar.gz

RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/msit_llm-8.3.2-py3-none-any.whl

# msit_llm pins NumPy below 2, while the runtime SciPy/CuPy stack requires
# NumPy 2.x. Re-pin after installing msit_llm to keep the engine importable.
RUN pip install --no-cache-dir --force-reinstall --no-deps numpy==2.3.5

# Cutlass DSL 4.6 calls the TVM-FFI dataclass mapping API introduced in 0.1.11.
# Keep this pin after all accelerator wheel installs, including deep_gemm, which
# declares the older 0.1.9 requirement.
RUN pip install --no-cache-dir --force-reinstall --no-deps apache-tvm-ffi==0.1.11

WORKDIR /sgl-workspace/sglang
