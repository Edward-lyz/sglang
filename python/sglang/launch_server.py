"""Launch the inference server."""

import asyncio
import logging
import os
import sys
import warnings

# Configure logging before the heavy imports below so the pre-launch
# BOOTSTRAP_ANALYZING points are visible (root logger defaults to WARNING and
# would otherwise drop INFO). configure_logger(force=True) resets this later.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)
logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] main_imports start")

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] main_imports done")

suppress_noisy_warnings()


def run_server(server_args):
    """Run the server based on the gRPC flags and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.smg_grpc_mode or server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server_aiak import launch_server

            launch_server(server_args)
    elif server_args.smg_grpc_mode:
        # Legacy SMG gRPC server (--smg-grpc-mode, or the deprecated --grpc-mode
        # which __post_init__ folds into smg_grpc_mode). The native Rust gRPC
        # server is a separate path, enabled by --grpc-port, that starts
        # alongside the default HTTP server below.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] import_http_server start")
        from sglang.srt.entrypoints.http_server import launch_server

        logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] import_http_server done")
        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    from sglang.srt.plugins import load_plugins

    logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] load_plugins start")
    load_plugins()
    logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] load_plugins done")

    logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] prepare_server_args start")
    server_args = prepare_server_args(sys.argv[1:])
    logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] prepare_server_args done")

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
