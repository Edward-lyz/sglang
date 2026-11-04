"""Launch the inference server."""

import asyncio
import logging
import os
import sys

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
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.encoder_only:
        from sglang.srt.disaggregation.encode_server import launch_server

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] import_http_server start")
        from sglang.srt.entrypoints.http_server import launch_server

        logger.info(f"[BOOTSTRAP_ANALYZING][{os.getpid()}] import_http_server done")
        launch_server(server_args)


if __name__ == "__main__":
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
