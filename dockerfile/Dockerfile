FROM iregistry.baidu-int.com/lyz/lmsysorg/sglang:dev-dsv41

## ===========================================================================================
# 这个 base 是社区 lmsysorg/sglang:dev-dsv41 镜像搬到 iregistry 上的副本，里面已经带了
# DeepSeek V4.1 需要的 kernel（FlashMLA v4.1 的 fp8/fp4 paged 布局、DSA top-k v2）与
# 对应的 torch / sgl-kernel / flashinfer，所以这一层只做两件事：把 aiak_sglang 换进去，
# 再补厂内环境需要的依赖。
#
# 和 kimi-k3 交付镜像相比，这里刻意少了一步：tokenspeed_mla 的精确除法 patch。
# 那个包只在 aiak-inference-sglang 基础镜像里，社区镜像里没有，patch 会直接失败。
#
# deep_gemm 处理见文件末尾：megamoe 要求固定 dev@1ff3932f 的 wheel，并配套一份用同一
# 版本烤出来的 DSv4.1-Flash JIT 预编译缓存（kimi-k3 那份按 K3 编的，对 DSv4.1 无效）。
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

# Rust 扩展打进源码树：base 镜像设了 SGLANG_RUST_BUILD_MODE=never，而上面的 editable
# install 没有 setuptools-rust，不会编 Rust 扩展；不打进来的话 as_native TreeCore 启动即报
# "not bundled or cached"。这里用 base 自带的 cargo/stable（无外网，拉不到 rust-toolchain.toml
# 固定的 1.92）+ 已缓存的 crates 编好，放到 never 模式会直接 import 的包路径下。
# never 模式不校验 .so 与源码是否一致：改了 rust/ 或换 torch 必须重新出镜像。
RUN cd /sgl-workspace/sglang && \
    RUSTUP_TOOLCHAIN=stable SGLANG_RUST_BUILD_MODE=auto SGLANG_CACHE_DIR=/tmp/rust-ext-build \
    python3 - <<'PY'
import shutil
import sysconfig
from pathlib import Path

import torch  # noqa: F401  libtorch must be resident before the TreeCore dlopen

from sglang.srt.mem_cache.rust_tree_core import extension
from sglang.srt.rust_extensions import load_rust_extension

suffix = sysconfig.get_config_var("EXT_SUFFIX")
for module in (
    extension.bindings,
    load_rust_extension("sglang.srt.rust_extensions._multimodal"),
):
    dst = Path("python") / (module.__name__.replace(".", "/") + suffix)
    shutil.copy2(module.__file__, dst)
    print("[rust-ext] bundled", module.__name__, "->", dst)
PY
RUN rm -rf /tmp/rust-ext-build && \
    cd /sgl-workspace/sglang && \
    SGLANG_RUST_BUILD_MODE=never SGLANG_CACHE_DIR=/tmp/rust-ext-empty python3 -c \
        'import torch; from sglang.srt.mem_cache.rust_tree_core.extension import bindings; from sglang.srt.rust_extensions import load_rust_extension; m = load_rust_extension("sglang.srt.rust_extensions._multimodal"); print("[rust-ext] never-mode OK:", bindings.__file__, m.__file__)' && \
    rm -rf /tmp/rust-ext-empty

RUN pip install https://cce-ai-datasets.bj.bcebos.com/hac-aiacc/deepseek/msit_llm-8.3.2-py3-none-any.whl onnx --no-deps

# deep_gemm 库固定成经 DSv4.1-Flash 验证过的构建（dev@1ff3932f，含 SF-alignment 修复，
# 治 CUDA graph capture 期 mega_moe.cuh:379 num_bytes%16 崩溃）。base 镜像自带的 deep_gemm
# 不保证是这个版本，而下面那份 JIT 缓存正是用这个版本在 bzz2 pod 上烤出来的，二者必须对齐，
# 所以这里显式卸掉旧的（含无 pip 元数据的裸目录）再装 BOS 上 pin 好的 wheel
# （torch 2.13.0+cu130 → torch2.13.0/ 路径；社区那条是 torch2.9.1，ABI 对不上不能复用）。
# 刷新方式：重编 deep_gemm 后打 wheel 传 BOS，替换下面 URL，并同步刷新那份 JIT 缓存包。
RUN set -eux; \
    pip uninstall -y deep_gemm 2>/dev/null || true; \
    for sp in $(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib']); print(sysconfig.get_paths()['platlib'])"); do \
        rm -rf "$sp/deep_gemm"; \
    done; \
    pip install --no-deps --no-cache-dir \
        https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/DeepGemm/develop/torch2.13.0/deep_gemm-0.1.8+1ff3932f-cp312-cp312-linux_x86_64.whl; \
    python3 -c "import importlib.metadata as m; v=m.version('deep_gemm'); print('[deep_gemm] installed', v); assert v == '0.1.8+1ff3932f', v"

# FlashMLA 07a1089：带 fused_norm_rope_attn_rope_cast（mega-attn fused-o），读引擎的 V4 584B / V4.1 池。
# base 镜像的 sgl_kernel.flash_mla 没有这个算子、也没有顶层 flash_mla 包，缺它时
# SGLANG_OPT_FLASHMLA_FUSED_ATTN_O 会每层静默失效（日志刷 "stays unfused"）。
# 顶层 flash_mla 只被 fused-o 路径(fused_o_attn.py)使用，不改普通 attention(走 sgl_kernel.flash_mla)。
# 仅适配 py3.12 + torch2.13 + cu130 + sm100/sm103；换 torch/CUDA 必须重编（文件名不体现这点）。
# 校验只读 dist-info 版本，不 import CUDA 扩展：构建环境无 GPU 驱动(libcuda.so.1)，import 会失败。
RUN pip install --no-deps --no-cache-dir \
        https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/FlashMLA/develop/flash_mla-1.0.0+07a1089.sm100a.sm103a-cp312-cp312-linux_x86_64.whl && \
    python3 -c "import importlib.metadata as m; v=m.version('flash_mla'); print('[flash_mla] installed', v); assert v.startswith('1.0.0+07a1089'), v"

# DSv4.1-Flash 的 JIT 预编译缓存：DeepGEMM cubin、Triton kernel、sgl-kernel JIT、tilelang。
# 内容取自跑最终 megamoe+DSpark 配置的 dsv41-flash PD pod(deep_gemm dev@1ff3932f,与上面 pin 的 wheel 同源):
# prefill(tp4 cp4 fake-dp2) + decode(dp8) 两侧 /root/.cache + /root/.tilelang 的并集,1028 在 bzz2 PD pod 上重烤。
# 换了 deep_gemm 就必须同步重烤这份(源码变→JIT cache key 变),两者要成对更新。
# 包内顶层就是 root/,所以解到 / 即落回原路径,省掉容器启动时的 kernel 编译;打包时已排除 pip HTTP 缓存。
#
# 刷新方式:在装了新 wheel 的 PD pod 上跑最终配置(prefill+decode),各自打包 /root/.cache + /root/.tilelang
# 取并集上传 BOS,再替换下面的 URL。包内路径约定不能变,末尾 test 会挡住路径不对的包。
RUN wget -q -O /tmp/dsv41-flash-cache.tar.gz \
        https://cce-ai-datasets.bj.bcebos.com/hac_test/vllm/DeepGemm/develop/torch2.13.0/deep_gemm_deepseekv41_flash_pd_cache_1028.tar.gz && \
    tar -xzf /tmp/dsv41-flash-cache.tar.gz -C / --exclude='root/.cache/pip' && \
    rm -f /tmp/dsv41-flash-cache.tar.gz && \
    test -d /root/.cache/sglang/deep_gemm/cache && test -d /root/.tilelang/cache

WORKDIR /sgl-workspace/sglang
