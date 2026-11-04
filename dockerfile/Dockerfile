FROM iregistry.baidu-int.com/hac-aiacc/lmsysorg/sglang:glm5-blackwell-deep_gemm_cache-flashmla1.0.0_2B0b475cc as base

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

RUN cp -Rf /tmp/aiak_sglang/scripts /sgl-workspace/sglang/
RUN cp -Rf /tmp/aiak_sglang/benchmark /sgl-workspace/sglang/

RUN cd /tmp/aiak_sglang && find . -name "*.py" -exec cp --parents {} /sgl-workspace/sglang \; && \
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
RUN pip install xgrammar==0.1.30 --no-deps
RUN pip install jsonschema_rs --no-deps
RUN pip uninstall deep_ep -y
RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/deep_ep/develop/bak/deep_ep-1.2.1+8f2a57d-cp312-cp312-linux_x86_64.whl
RUN pip uninstall deep_gemm -y
RUN rm -r /usr/local/lib/python3.12/dist-packages/deep_gemm
RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/DeepGemm/develop/torch2.9.1/deep_gemm-2.4.2+91bc9ee-cp312-cp312-linux_x86_64.whl
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    'https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/FlashMLA/develop/torch2.9.1/flash_mla-1.0.0+9d77d8a-cp312-cp312-linux_x86_64.whl'

# Install deepgemm cache
RUN cd /root && wget https://ai-data.bj.bcebos.com/mae/cache_glm5.1_0620_merged.tar.gz  && tar -xzvf cache_glm5.1_0620_merged.tar.gz && rm /root/.cache -rf && mv cache_glm5.1_0620_merged/.cache /root/.cache && rm cache_glm5.1_0620_merged.tar.gz

RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/msit_llm-8.3.2-py3-none-any.whl

WORKDIR /sgl-workspace/sglang
