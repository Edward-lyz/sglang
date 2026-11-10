"""Independent HTTP worker listeners sharing a SO_REUSEPORT service port."""

import logging
import socket

import uvicorn
from uvicorn.config import STARTUP_FAILURE
from uvicorn.supervisors.multiprocess import Multiprocess, Process

logger = logging.getLogger("uvicorn.error")


def bind_reuseport_socket(config):
    """Bind an independent listener and close it if socket setup fails.

    Args:
        config: Uvicorn configuration specifying the listener address and
            worker settings.

    Returns:
        socket.socket: The bound listener owned by the caller.

    Raises:
        OSError: Socket creation, option setup or binding failed.
    """
    family = socket.AF_INET6 if ":" in config.host else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        listener.bind((config.host, config.port))
        return listener
    except BaseException:
        listener.close()
        raise


class ReusePortMultiprocess(Multiprocess):
    """Run HTTP workers with independent listeners on the same service port."""
    def _start_process(self):
        listener = bind_reuseport_socket(self.config)
        try:
            process = Process(self.config, [listener])
            process.start()
            return process
        finally:
            listener.close()

    def init_processes(self):
        """Start all configured workers and clean up a partial startup failure.

        Returns:
            None.

        Raises:
            Exception: Worker startup failed after child cleanup.
        """
        try:
            for _ in range(self.processes_num):
                self.processes.append(self._start_process())
        except BaseException:
            self.terminate_all()
            self.join_all()
            raise

    def keep_subprocess_alive(self):
        """Replace dead children unless shutdown or startup failure stops the
        parent.

        Returns:
            None.

        Raises:
            Exception: Starting a replacement worker failed.
        """
        if self.should_exit.is_set():
            return
        for index, process in enumerate(self.processes):
            if process.is_alive(timeout=self.config.timeout_worker_healthcheck):
                continue
            process.kill()
            process.join()
            if process.exitcode == STARTUP_FAILURE:
                logger.error("HTTP worker failed during startup; stopping parent.")
                self.should_exit.set()
                return
            if self.should_exit.is_set():
                return
            logger.info("Child process [%s] died", process.pid)
            self.processes[index] = self._start_process()

    def restart_all(self):
        """Replace workers one by one, retaining old workers if readiness
        fails.

        Returns:
            None.

        Raises:
            Exception: Starting a replacement worker failed.
        """
        for index, old_process in enumerate(self.processes):
            if self.should_exit.is_set():
                return
            new_process = self._start_process()
            if not new_process.wait_until_ready(
                self.config.timeout_worker_healthcheck, self.should_exit
            ):
                new_process.kill()
                new_process.join()
                logger.error(
                    "Replacement worker [%s] did not become ready; "
                    "keeping worker [%s].",
                    new_process.pid,
                    old_process.pid,
                )
                return
            old_process.terminate()
            old_process.join()
            self.processes[index] = new_process

    def handle_ttin(self):
        """Start one additional HTTP worker in response to the scale-up signal.

        Returns:
            None.

        Raises:
            Exception: Starting the additional worker failed.
        """
        process = self._start_process()
        self.processes.append(process)
        self.processes_num += 1


def run_uvicorn_reuseport(app, **kwargs):
    """Run and clean up the independent-listener HTTP supervisor.

    Args:
        app: ASGI application or import path accepted by Uvicorn.
        kwargs: Keyword options forwarded unchanged to Uvicorn configuration.

    Returns:
        None.

    Raises:
        RuntimeError: The platform does not support SO_REUSEPORT.
        ValueError: Worker count or service port does not support independent
            listeners.
        OSError: The service listener cannot be bound.
    """
    if not hasattr(socket, "SO_REUSEPORT"):
        raise RuntimeError("SGLANG_HTTP_REUSE_PORT requires SO_REUSEPORT support")
    config = uvicorn.Config(app, **kwargs)
    if config.workers < 2 or config.port == 0:
        raise ValueError("Independent HTTP listeners require workers > 1 and port > 0")
    logger.info(
        "Starting %s HTTP workers with independent SO_REUSEPORT listeners on %s:%s",
        config.workers,
        config.host,
        config.port,
    )
    supervisor = ReusePortMultiprocess(config, sockets=[])
    with bind_reuseport_socket(config):
        try:
            supervisor.run()
        finally:
            for process in supervisor.processes:
                if process.process.is_alive():
                    process.terminate()
            for process in supervisor.processes:
                process.process.join()
