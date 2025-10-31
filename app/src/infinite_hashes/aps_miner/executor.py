"""
Hard timeout executor for APScheduler.

Runs each job in a separate multiprocessing.Process with enforced timeouts.
"""

import threading
import time
import traceback
from multiprocessing import Process, Queue

from apscheduler.executors.base import BaseExecutor
from apscheduler.job import Job


def _child_wrapper(func, args, kwargs, q: Queue):
    """
    Runs inside the spawned process.
    Executes the job function and puts (status, result, error) into the queue.
    """
    try:
        rv = func(*args, **kwargs)
        q.put(("ok", rv, None))
    except Exception as exc:
        tb = traceback.format_exc()
        q.put(("err", None, (exc, tb)))


class HardTimeoutExecutor(BaseExecutor):
    """
    APScheduler executor that launches each job in a separate multiprocessing.Process,
    with a hard timeout (terminate()) and full result/traceback reporting.

    Per-job options:
        job.kwargs["_exec"] = {
            "timeout": 30,   # seconds (optional)
        }
    """

    def start(self, scheduler, alias):
        super().start(scheduler, alias)
        self._logger = getattr(scheduler, "_logger", None)
        self._procs = {}  # job_id -> (Process, Queue)

    def shutdown(self, wait=True):
        """
        Terminates all running processes when the scheduler stops.
        """
        for job_id, (p, q) in list(self._procs.items()):
            if p.is_alive():
                p.terminate()
            p.join(timeout=1)
        self._procs.clear()

    def _do_submit_job(self, job: Job, run_times):
        """
        Called by APScheduler to execute a job.
        Spawns a process, starts a watcher thread that enforces timeout
        and reports success/failure back to the scheduler.
        """
        # Extract executor-specific options
        job_kwargs = dict(job.kwargs)
        exec_opts = dict(job_kwargs.pop("_exec", {}) or {})
        timeout = exec_opts.get("timeout", None)  # seconds or None

        # Create IPC queue for result passing
        q = Queue()
        p = Process(target=_child_wrapper, args=(job.func, job.args, job_kwargs, q), daemon=True)

        try:
            p.start()
            self._procs[job.id] = (p, q)
        except Exception as exc:
            # Log error ourselves and report to scheduler
            if self._logger:
                self._logger.error("Failed to start job %s process: %s\n%s", job.id, exc, traceback.format_exc())
            self._run_job_error(job.id, exc, None)
            return

        def watch():
            """
            Runs in a background thread.
            Waits for the process result, handles timeouts, kills if needed,
            and calls APScheduler callbacks.
            """
            try:
                if timeout is not None:
                    end = time.time() + timeout
                else:
                    end = None

                while True:
                    # If the process sent a result, consume and report it
                    if not q.empty():
                        status, rv, err = q.get()
                        if status == "ok":
                            self._run_job_success(job.id, run_times)
                        else:
                            exc, tb_str = err
                            # Log the traceback string ourselves
                            if self._logger:
                                self._logger.error("Job %s raised an exception:\n%s", job.id, tb_str)
                            # Report error to scheduler (without traceback to avoid logging issues)
                            self._run_job_error(job.id, exc, None)
                        break

                    # If timeout expired, terminate the process
                    if end is not None and time.time() > end:
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=1)
                        exc = TimeoutError(f"Job hard-timeout after {timeout}s (pid={p.pid})")
                        if self._logger:
                            self._logger.error(
                                "Job %s timed out: Process %s terminated after %ss", job.id, p.pid, timeout
                            )
                        self._run_job_error(job.id, exc, None)
                        break

                    # If process died without sending result — report error
                    if not p.is_alive() and q.empty():
                        exc = RuntimeError(f"Job process exited without result (pid={p.pid})")
                        if self._logger:
                            self._logger.error("Job %s process died: No result received (pid=%s)", job.id, p.pid)
                        self._run_job_error(job.id, exc, None)
                        break

                    time.sleep(0.05)
            finally:
                try:
                    p.close()
                except Exception as exc:
                    if self._logger:
                        self._logger.warning(
                            "Failed to close process for job %s: %s",
                            job.id,
                            exc,
                        )
                self._procs.pop(job.id, None)

        # Launch watcher thread
        threading.Thread(target=watch, daemon=True).start()
