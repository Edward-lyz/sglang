"""Keep restore resources available only during a persistent MegaMoE launch."""

import logging
import os
from contextlib import contextmanager

_SMS = int(os.environ.get("AS_MEGA_MOE_NUM_SMS", "0"))
_validated = False
logger = logging.getLogger(__name__)


@contextmanager
def as_moe_sm_scope():
    """仅在已验证的 persistent MoE 启动期间设置 SM 预算，退出时恢复。"""
    global _validated
    if _SMS == 0:
        yield
        return
    if _SMS != 112:
        raise RuntimeError(
            "AS MegaMoE overlap currently supports only the audited 112-SM budget"
        )
    import deep_gemm
    import torch

    if not _validated:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if props.major != 10 or props.multi_processor_count != 148:
            raise RuntimeError("AS MegaMoE overlap expects 148-SM SM100 hardware")
        logger.info(
            "AS MegaMoE scoped budget enabled: persistent MoE=112, ordinary GEMM=148"
        )
        _validated = True
    original = deep_gemm.get_num_sms()
    if original != 148:
        raise RuntimeError(
            f"AS MegaMoE scope entered with unexpected SM budget {original}"
        )
    try:
        # 只限制 persistent MoE 的这次启动，给 AS 通信留出资源；不能把预算设为全局常驻值。
        deep_gemm.set_num_sms(_SMS)
        yield
    finally:
        # 异常也必须恢复，普通 GEMM 和下一次启动继续使用原有完整 SM 预算。
        deep_gemm.set_num_sms(original)
