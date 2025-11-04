"""Worker process management for multi-process testing.

Provides WorkerProcess class for managing validator/miner worker processes
with command/response queue communication.
"""

import multiprocessing as mp
import queue
from contextlib import suppress
from typing import Any


class WorkerProcess:
    """Manages a validator or miner worker process with command/response queues."""

    def __init__(self, worker_id: int, target, config: dict[str, Any]) -> None:
        self.worker_id = worker_id
        self.command_queue: mp.Queue = mp.Queue()
        self.response_queue: mp.Queue = mp.Queue()
        self.process = mp.Process(
            target=target,
            args=(worker_id, self.command_queue, self.response_queue, config),
            daemon=True,
        )

    def start(self, timeout: float = 20.0) -> None:
        self.process.start()
        try:
            ready = self.response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError("worker failed to signal readiness") from exc
        if ready.get("type") != "READY" or not ready.get("success", False):
            raise RuntimeError(ready.get("error", "worker failed to start"))

    def stop(self, timeout: float = 10.0) -> None:
        with suppress(RuntimeError):
            self.send_command("STOP", timeout=timeout)
        self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def send_command(self, command_type: str, params: dict[str, Any] | None = None, timeout: float = 600.0) -> Any:
        payload: dict[str, Any] = {
            "type": command_type,
            "worker_id": self.worker_id,
        }
        if params:
            payload["params"] = params
        self.command_queue.put(payload)
        try:
            response = self.response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(f"timeout waiting for {command_type}") from exc
        if response.get("type") != "RESPONSE":
            raise RuntimeError(f"unexpected message: {response}")
        if not response.get("success", False):
            raise RuntimeError(f"command {command_type} failed: {response.get('error')}")
        return response.get("data")
