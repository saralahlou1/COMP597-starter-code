# trainer/stats/simple_system_stats.py
import time
import json
import os
from typing import Optional, Dict, Any, List

import psutil

# optional libs (guarded)
try:
    import torch
except Exception:
    torch = None

try:
    import pynvml
except Exception:
    pynvml = None

from src.trainer.stats.base import TrainerStats as BaseTrainerStats

trainer_stats_name="sys"

def construct_trainer_stats(conf : config.Config, **kwargs) -> base.TrainerStats:
    return SimpleSystemStats()


def _init_nvml_once():
    """Return True if NVML initialized (or already init failed)."""
    if pynvml is None:
        return False
    try:
        # safe to call multiple times in practice; guard with try/except
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

    # torch.cuda gives per-process memory (if torch present and cuda available)
    if torch is not None and torch.cuda.is_available():
        info["gpu_present"] = True
        try:
            # device 0 (assumption: training uses single GPU)
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

    def __init__(self):
        """
        Args:
            out_path: path to write stats at the end. If suffix is .jsonl or .ndjson,
                      will write one JSON object per line. Otherwise writes a JSON array.
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

    # ---------- life-cycle ----------
    def start_train(self) -> None:
        self._running = True
        # init NVML once if installed (harmless if fails)
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
            # best-effort: print error but do not crash training
            print(f"[SimpleSystemStats] failed to write stats to {self.out_path}: {e}")

        # try to shutdown NVML if we initialized it
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
        # append to memory
        self.rows.append(row)

        # reset transient markers for next step
        self._before_sample = None
        self._after_forward = None
        self._after_backward = None
        self._after_optimizer = None
        self._after_checkpoint = None
        self._ts_step_start = None

        # call log_step hook (no-op by default)
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

    def start_save_checkpoint(self) -> None:
        self._ts_checkpoint_start = time.time()

    def stop_save_checkpoint(self) -> None:
        torch.cuda.synchronize()
        now = time.time()
        dur = now - self._ts_checkpoint_start if self._ts_checkpoint_start else None
        self._after_checkpoint = {
            "time": now,
            "duration_s": dur,
            "host_process": _sample_host_process(),
            "gpu": _safe_gpu_util_and_mem(),
        }

    # ---------- logging hooks ----------
    def log_step(self) -> None:
        # no-op by default. Subclass to push to TB/W&B if desired.
        pass

    def log_stats(self) -> None:
        # short summary printed at end (non-crashing)
        try:
            n = len(self.rows)
            last_loss = self._last_loss
            print(f"[SimpleSystemStats] collected {n} steps. last_loss={last_loss}")
        except Exception:
            pass

