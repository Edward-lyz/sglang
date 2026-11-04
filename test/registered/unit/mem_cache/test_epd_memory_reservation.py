from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache import kv_cache_configurator


@pytest.mark.parametrize(
    "language_only,backend,pool_mb,transport,expected_mb",
    [
        (True, "mooncake", None, "cpu", 4096 + 100),
        (True, "mooncake", "4096", "cpu", 4096 + 100),
        (True, "mooncake", "4096", "cuda_ipc", 4096 + 100 + 256),
        (True, "mooncake", "0", "cpu", 100),
        (True, "zmq_to_scheduler", None, "cpu", 100),
        (True, "zmq_to_scheduler", "2048", "cpu", 2048 + 100),
        (False, "mooncake", "4096", "cpu", 100),
        (True, "http", "4096", "cpu", 100),
    ],
)
def test_epd_pool_is_charged_before_kv_sizing(
    monkeypatch, language_only, backend, pool_mb, transport, expected_mb
):
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_disagg",
        lambda: SimpleNamespace(
            language_only=language_only, encoder_transfer_backend=backend
        ),
    )
    monkeypatch.setenv("SGLANG_VLM_CACHE_SIZE_MB", "100")
    monkeypatch.setenv("SGLANG_MM_FEATURE_CACHE_MB", "256")
    monkeypatch.delenv("SGLANG_EMBEDDING_POOL_SIZE_MB", raising=False)
    if pool_mb is not None:
        monkeypatch.setenv("SGLANG_EMBEDDING_POOL_SIZE_MB", pool_mb)
    assert kv_cache_configurator.mm_runtime_reservation_gb(
        is_multimodal=True, mm_feature_transport=transport
    ) == expected_mb / 1024


def test_text_model_has_no_multimodal_reservation(monkeypatch):
    monkeypatch.setenv("SGLANG_EMBEDDING_POOL_SIZE_MB", "4096")
    assert kv_cache_configurator.mm_runtime_reservation_gb(
        is_multimodal=False, mm_feature_transport="cpu"
    ) == 0


def test_minimum_static_fraction_includes_epd_reservation(monkeypatch):
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_world_group",
        lambda: SimpleNamespace(world_size=1, cpu_group=None),
    )
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_schedule",
        lambda: SimpleNamespace(mem_fraction_static=0.7),
    )
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_mm",
        lambda: SimpleNamespace(mm_feature_transport="cpu"),
    )
    monkeypatch.setattr(
        kv_cache_configurator,
        "get_available_gpu_memory",
        lambda *args, **kwargs: 42,
    )
    monkeypatch.setattr(
        kv_cache_configurator,
        "mm_runtime_reservation_gb",
        lambda **kwargs: 4.1,
    )
    configurator = SimpleNamespace(
        device="cuda",
        gpu_id=0,
        mambaish_config=None,
        model_config=SimpleNamespace(is_multimodal=True),
    )
    with pytest.raises(ValueError, match="above 0.785"):
        kv_cache_configurator.KVCacheConfigurator._profile_available_bytes(
            configurator, 176
        )
