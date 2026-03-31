# trainer/stats/simple_system_stats.py
import time
import json
import os
from typing import Optional, Dict, Any, List

import psutil
import threading   # NEW

# optional libs (guarded)
import torch
import pynvml

from src.trainer.stats.base import TrainerStats as BaseTrainerStats

trainer_stats_name="sys"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return SimpleSystemStats()


def _init_nvml_once():
    """Return True if NVML initialized (or already init failed)."""
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def _safe_gpu_util_and_mem():
    """Return dict with small GPU info for gpu 0 (or None values)."""
    info = {
        "gpu_present": False,
        "gpu_name": None,
        "gpu_util_percent": None,
        "gpu_mem_used_bytes": None,
        "gpu_mem_total_bytes": None,
        "torch_allocated_bytes": None,
        "torch_reserved_bytes": None,
    }

    if torch is not None and torch.cuda.is_available():
        info["gpu_present"] = True
        try:
            info["torch_allocated_bytes"] = int(torch.cuda.memory_allocated(0))
            info["torch_reserved_bytes"] = int(torch.cuda.memory_reserved(0))
        except Exception:
            info["torch_allocated_bytes"] = None
            info["torch_reserved_bytes"] = None

    # pynvml for gpu util and total/memory used (global device stats)
    if pynvml is not None:
        try:
            # attempt to init (no-op if already)
            if _init_nvml_once():
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="ignore")
                    info["gpu_name"] = name
                except Exception:
                    info["gpu_name"] = None
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    info["gpu_util_percent"] = int(getattr(util, "gpu", None) or 0)
                except Exception:
                    info["gpu_util_percent"] = None
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    info["gpu_mem_used_bytes"] = int(getattr(mem, "used", None) or 0)
                    info["gpu_mem_total_bytes"] = int(getattr(mem, "total", None) or 0)
                except Exception:
                    info["gpu_mem_used_bytes"] = None
                    info["gpu_mem_total_bytes"] = None
                info["gpu_present"] = True if info["gpu_name"] or info["gpu_mem_total_bytes"] else info["gpu_present"]
        except Exception:
            # NVML not available or no GPU 0
            pass

    return info


def _sample_host_process():
    """Return small dict of host/process stats."""
    d = {}
    try:
        d["host_cpu_percent"] = psutil.cpu_percent(interval=None)
    except Exception:
        d["host_cpu_percent"] = None
    try:
        vm = psutil.virtual_memory()
        d["host_mem_percent"] = vm.percent
        d["host_mem_used_bytes"] = int(vm.used)
        d["host_mem_total_bytes"] = int(vm.total)
    except Exception:
        d["host_mem_percent"] = None
        d["host_mem_used_bytes"] = None
        d["host_mem_total_bytes"] = None
    try:
        p = psutil.Process(os.getpid())
        mi = p.memory_info()
        d["process_rss_bytes"] = int(mi.rss)
        d["process_vms_bytes"] = int(mi.vms)
    except Exception:
        d["process_rss_bytes"] = None
        d["process_vms_bytes"] = None
    return d


class SimpleSystemStats(BaseTrainerStats):
    """
    Lightweight TrainerStats that samples host + single-GPU metrics on demand,
    stores rows in memory and writes to a file at stop_train().

    Per-step snapshots captured:
      - before_step (sample at start_step)
      - after_forward (sample at stop_forward)
      - after_backward (sample at stop_backward)
      - after_optimizer (sample at stop_optimizer_step)
      - after_checkpoint (sample at stop_save_checkpoint)
    """

    def __init__(self, checkpoint_sampling_interval: float = 0.1):
        """
        Args:
            checkpoint_sampling_interval: interval (seconds) between checkpoint samples.
        """
        self.out_path = "training_stats.json"
        self.rows: List[Dict[str, Any]] = []
        # per-step transient holders
        self._before_sample = None
        self._after_forward = None
        self._after_backward = None
        self._after_optimizer = None
        self._after_checkpoint = None

        # timestamp markers
        self._ts_step_start = None
        self._ts_forward_start = None
        self._ts_backward_start = None
        self._ts_optimizer_start = None
        self._ts_checkpoint_start = None

        self._last_loss = None
        self._step_idx = 0
        self._epoch_idx = 0
        self._running = False

        # --- checkpoint sampler thread state (NEW) ---
        self._checkpoint_sampling_interval = checkpoint_sampling_interval
        self._checkpoint_sampling = False
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._checkpoint_samples: List[Dict[str, Any]] = []
        self._checkpoint_lock = threading.Lock()

    # ---------- life-cycle ----------
    def start_train(self) -> None:
        self._running = True
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass

    def stop_train(self) -> None:
        self._running = False
        # write all rows to disk now
        try:
            if self.out_path.lower().endswith((".jsonl", ".ndjson")):
                with open(self.out_path, "w") as f:
                    for r in self.rows:
                        f.write(json.dumps(r) + "\n")
            else:
                # single JSON array
                with open(self.out_path, "w") as f:
                    json.dump(self.rows, f, indent=2)
        except Exception as e:
            print(f"[SimpleSystemStats] failed to write stats to {self.out_path}: {e}")

        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ---------- step / phase methods ----------
    def start_step(self) -> None:
        self._step_idx += 1
        self._ts_step_start = time.time()
        self._before_sample = {
            "time": self._ts_step_start,
            "host_process": _sample_host_process(),
            "gpu": _safe_gpu_util_and_mem(),
        }
        # Clear other phase samples
        self._after_forward = None
        self._after_backward = None
        self._after_optimizer = None
        self._after_checkpoint = None

        # reset phase timestamps
        self._ts_forward_start = None
        self._ts_backward_start = None
        self._ts_optimizer_start = None
        self._ts_checkpoint_start = None

    def stop_step(self) -> None:
        now = time.time()
        step_duration = now - self._ts_step_start if self._ts_step_start else None

        # assemble final row
        row = {
            "step_idx": self._step_idx,
            "epoch_idx": self._epoch_idx,
            "timestamp": now,
            "step_duration_s": step_duration,
            "loss": self._last_loss,
            "snapshots": {
                "before_step": self._before_sample,
                "after_forward": self._after_forward,
                "after_backward": self._after_backward,
                "after_optimizer": self._after_optimizer,
                "after_checkpoint": self._after_checkpoint,
            }
        }
        self.rows.append(row)

        self._before_sample = None
        self._after_forward = None
        self._after_backward = None
        self._after_optimizer = None
        self._after_checkpoint = None
        self._ts_step_start = None

        self.log_step()

    def start_forward(self) -> None:
        self._ts_forward_start = time.time()

    def stop_forward(self) -> None:
        torch.cuda.synchronize()
        now = time.time()
        dur = now - self._ts_forward_start if self._ts_forward_start else None
        self._after_forward = {
            "time": now,
            "duration_s": dur,
            "host_process": _sample_host_process(),
            "gpu": _safe_gpu_util_and_mem(),
        }

    def log_loss(self, loss) -> None:
        try:
            if hasattr(loss, "item"):
                self._last_loss = float(loss.item())
            else:
                self._last_loss = float(loss)
        except Exception:
            self._last_loss = None

    def start_backward(self) -> None:
        self._ts_backward_start = time.time()

    def stop_backward(self) -> None:
        torch.cuda.synchronize()
        now = time.time()
        dur = now - self._ts_backward_start if self._ts_backward_start else None
        self._after_backward = {
            "time": now,
            "duration_s": dur,
            "host_process": _sample_host_process(),
            "gpu": _safe_gpu_util_and_mem(),
        }

    def start_optimizer_step(self) -> None:
        self._ts_optimizer_start = time.time()

    def stop_optimizer_step(self) -> None:
        torch.cuda.synchronize()
        now = time.time()
        dur = now - self._ts_optimizer_start if self._ts_optimizer_start else None
        self._after_optimizer = {
            "time": now,
            "duration_s": dur,
            "host_process": _sample_host_process(),
            "gpu": _safe_gpu_util_and_mem(),
        }

    # ---------- checkpoint sampler (NEW) ----------
    def _checkpoint_sampler_loop(self):
        """Background loop run during checkpointing to collect host/gpu samples."""
        try:
            while self._checkpoint_sampling:
                now = time.time()
                sample = {
                    "time": now,
                    "host_process": _sample_host_process(),
                    "gpu": _safe_gpu_util_and_mem(),
                }
                # append under lock
                with self._checkpoint_lock:
                    self._checkpoint_samples.append(sample)
                time.sleep(self._checkpoint_sampling_interval)
        except Exception:
            # keep best-effort semantics: don't raise from the thread
            return

    def start_save_checkpoint(self) -> None:
        """Start checkpointing and begin background sampling thread for the checkpoint."""
        # record start timestamp as before
        self._ts_checkpoint_start = time.time()
        # clear any previous checkpoint samples
        with self._checkpoint_lock:
            self._checkpoint_samples = []
        # start background sampler thread
        self._checkpoint_sampling = True
        t = threading.Thread(target=self._checkpoint_sampler_loop, daemon=True)
        self._checkpoint_thread = t
        t.start()

    def stop_save_checkpoint(self) -> None:
        """Stop checkpointing and stop/join the sampling thread; store the collected samples."""
        # synchronize GPU work first (as before)
        torch.cuda.synchronize()

        # stop the background sampler
        self._checkpoint_sampling = False

        # assemble duration and include the collected samples
        now = time.time()
        dur = now - self._ts_checkpoint_start if self._ts_checkpoint_start else None

        if self._checkpoint_thread is not None:
            # join with a timeout to avoid blocking forever
            self._checkpoint_thread.join(timeout=max(2.0, self._checkpoint_sampling_interval * 5.0))
            self._checkpoint_thread = None

        # copy samples out under lock
        with self._checkpoint_lock:
            samples_copy = list(self._checkpoint_samples)
            # optionally clear the stored list
            # self._checkpoint_samples = []

        # last host/gpu snapshot (best-effort: take last sample if available, otherwise do one final sample)
        if samples_copy:
            last_sample = samples_copy[-1]
            host_snapshot = last_sample.get("host_process")
            gpu_snapshot = last_sample.get("gpu")
        else:
            host_snapshot = _sample_host_process()
            gpu_snapshot = _safe_gpu_util_and_mem()

        self._after_checkpoint = {
            "time": now,
            "duration_s": dur,
            "host_process": host_snapshot,
            "gpu": gpu_snapshot,
            "checkpoint_samples": samples_copy,
        }

    # ---------- logging hooks ----------
    def log_step(self) -> None:
        pass

    def log_stats(self) -> None:
        # short summary printed at end (non-crashing)
        try:
            n = len(self.rows)
            last_loss = self._last_loss
            print(f"[SimpleSystemStats] collected {n} steps. last_loss={last_loss}")
        except Exception:
            pass
