"""GPU processor registry.

Maps model architecture to GPU processor class.
Only the matched module is imported at runtime.
"""

# arch -> (module_path, class_name)
_REGISTRY = {
    "KimiK25ForConditionalGeneration": (
        "sglang.srt.multimodal.processors.kimi_k25",
        "KimiK25VLImageProcessor",
    ),
    "KimiK3ForConditionalGeneration": (
        "sglang.srt.multimodal.processors.kimi_k3",
        "KimiK3ImageProcessor",
    ),
    "KimiK31ForConditionalGeneration": (
        "sglang.srt.multimodal.processors.kimi_k3",
        "KimiK3ImageProcessor",
    ),
}


def get_gpu_processor_cls(model_path: str):
    """Get GPU processor class by reading model's config.json architecture.

    Only imports the matched processor module.
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    architectures = getattr(config, "architectures", [])

    for arch in architectures:
        entry = _REGISTRY.get(arch)
        if entry is not None:
            import importlib
            mod = importlib.import_module(entry[0])
            return getattr(mod, entry[1])

    raise ValueError(
        f"No GPU processor for architectures: {architectures}. "
        f"Available: {list(_REGISTRY.keys())}"
    )
