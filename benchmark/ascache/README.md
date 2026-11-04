# AttentionStore 主机链路与原生组件

## 合入范围

基线：`e10a5b3c0e`，分支 `ds-dev-ds4.1f-1026-cache-analyze`。
按功能合入历史 108 实现，不用旧快照覆盖整个仓库。

- 保留基线 xgrammar、模型和 CP 路径、小 chunk metadata 批处理、pinned H2D 改动。
- 合入缓存前缀比较、后缀拷贝、无分配长度计算和 SWA staging 顺序优化。
- 合入异步 load-back、分组 diffuse、有限 refill 和可选成功日志。
- 合入 CP 感知 Rust Full/SWA 树、整数 token 数组 OOB 广播。
- 合入配套 SDK 的原生 RPC/copy/release、页 metadata 和 FIFO writer 接口。
- 延后 allocation vote，但保留队列顺序、readiness 和结果计数的全 rank 发布门槛。
- Python 仍负责 admission、树和 slot 所有权边界；不是全部非 forward 原生化。

代码、模块、C ABI、日志、测试和配置使用功能名，不使用实验轮次后缀。
旧实验配置不提供别名，启动时必须使用下表中的新名称。
SDK 必须使用本次配套合入版本；不能混用旧的原生扩展。

## 构建

先按 SDK README 构建 helper、原生 worker 和 metadata，并安装对应 wheel。
然后在引擎仓库运行：

```bash
python benchmark/ascache/build_native.py
```

使用基线已有的 Rust 源码指纹构建/加载机制，保留 inspection 模式。
该命令允许源码构建并预热 OOB packer，不加载模型、不初始化 CUDA。
需要匹配的 PyTorch、C++ 和 Rust 工具链。仓库锁定 Rust 1.92；本机没有该
工具链且无法下载，构建/测试使用了 `RUSTUP_TOOLCHAIN=stable`（本机 1.98.1）。
不修改仓库工具链配置；使用其他工具链需显式设置并重新验证。

## 可选配置

在**现有业务启动命令**之前设置；模型、GPU 拓扑和服务端参数保持原部署配置。
这些选项默认关闭，成功日志默认开启。

```bash
export ATTENTIONSTORE_USE_PEERMEM=1
export ATTENTIONSTORE_PEERMEM_DEVICE=mlx5_bond_0  # 按节点实际拓扑修改
export ATTENTIONSTORE_PEERMEM_GID_INDEX=3
export ATTENTIONSTORE_NATIVE_CONTROL=1
export ATTENTIONSTORE_NATIVE_METADATA=1
export ATTENTIONSTORE_NATIVE_WRITE=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=as_native
export SGLANG_ENABLE_REQUEST_OOB_BROADCAST=1
export SGLANG_AS_DEFER_ALLOCATION_GATE=1
export SGLANG_ENABLE_AS_LOAD_BACK_ASYNC=1
export SGLANG_AS_LOAD_BACK_READ_WORKERS=4
export SGLANG_AS_LOAD_BACK_MAX_PENDING=32
export SGLANG_AS_LOAD_BACK_COALESCE_READY=1
export SGLANG_AS_LOAD_BACK_REFILL_PASSES=2
export SGLANG_AS_READ_DIFFUSE_BATCHED=1
export SGLANG_AS_READ_DIFFUSE_GROUP_COMMIT=1
export SGLANG_AS_SPAG_BATCH_WRITE_FENCE=1
export SGLANG_AS_SPAG_REUSE_FINISHED_WRITE=1
export SGLANG_LOG_AS_CACHE_OPERATIONS=0
```

需按现有部署启用统一 radix cache、AS cache、CP diffuse 等配套功能。
`as_native` 只选择已支持的 Full/SWA 契约；session、Mamba、PP、自定义契约
保留 Python 树。OOB 不支持的请求回退旧广播。所有 rank 必须使用一致配置。
forward 线程和 ahead 开关保持业务部署选择，本次不强制切换。

## CPU 验证

源码模式可用 `PYTHONPATH=/path/to/sdk:python` 选择配套 SDK，不改变全局安装。

```bash
CUDA_VISIBLE_DEVICES=99 python -m pytest -q \
  benchmark/ascache/test_deferred_allocation.py \
  benchmark/ascache/test_native_serving.py \
  benchmark/ascache/test_native_write_serving.py \
  benchmark/ascache/test_native_shutdown.py
CUDA_VISIBLE_DEVICES=99 python benchmark/ascache/test_gloo_load_back.py /tmp/as-gloo-grouped --grouped 1
CUDA_VISIBLE_DEVICES=99 python benchmark/ascache/test_gloo_load_back.py /tmp/as-gloo-plain --grouped 0
python -m unittest discover -s benchmark/ascache -p test_cpu_helpers.py
python benchmark/ascache/validate_request_oob.py /tmp/as-oob.json
```

Gloo 测试使用真实 4 个 CPU rank，注入 allocation、submit、read 错误，验证失败请求
不发布、资源只释放一次。测试目录必须是新目录。`CUDA_VISIBLE_DEVICES=99` 用于
隐藏 GPU，兼容既有测试工具对设备字符串的读取；测试不初始化 CUDA。

本次结果见 `validation.json`。新基线并非历史压测快照，历史吞吐不能当作本次
合入性能。未启动 GPU 推理/压测，未做业务精度、真实 RDMA、长稳或生产验收。
