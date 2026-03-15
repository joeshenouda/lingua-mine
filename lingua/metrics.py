# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timezone

import torch
import torch.nn as nn

from lingua.distributed import get_is_master
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()


@dataclass
class TensorboardArgs:
    log_dir: Optional[str] = None
    comment: Optional[str] = None
    purge_step: Optional[int] = None
    max_queue: Optional[int] = None
    flush_secs: Optional[int] = None
    filename_suffix: Optional[str] = None


@dataclass
class LoggingArgs:
    freq: int = 10  # Log every freq optimizer steps
    acc_freq: Optional[int] = None  # Log every acc_freq gradient accumulation steps

    tensorboard: Optional[TensorboardArgs] = None


class MetricLogger:
    def __init__(self, outdir: Path, args: Optional[Any] = None):
        self.outdir = outdir
        self.jsonl_writer = None
        self.args = args
        self.tb_writer = None

    def open(self):
        if self.jsonl_writer is None:
            self.jsonl_writer = open(self.outdir, "a")
        if (
            self.args is not None
            and self.args.logging.tensorboard is not None
            and get_is_master()
        ):
            tb_args = asdict(self.args.logging.tensorboard)
            # Filter out None values
            tb_args = {k: v for k, v in tb_args.items() if v is not None}
            if 'log_dir' not in tb_args:
                name = getattr(self.args, 'name', 'experiment')
                tb_args['log_dir'] = str(self.outdir.parent / f'tb_{name}')
            self.tb_writer = SummaryWriter(**tb_args)

    def log(self, metrics: Dict[str, Any]):
        if self.tb_writer is not None:
            step = metrics.get("global_step", 0)
            for key, value in metrics.items():
                if key != "global_step" and isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        metrics.update({"created_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
        "power_draw",
    ],
)


class GPUMemoryMonitor:
    """
    Class to monitor GPU memory usage
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        # reset stats, clear cache
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]
        power_draw = torch.cuda.power_draw()

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
            power_draw,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def __str__(self):
        mem_stats = self.get_peak_stats()
        display_str = f"{self.device_name} ({self.device_index}): {self.device_capacity_gib} GiB capacity, "
        display_str += (
            f"{mem_stats.max_reserved_gib} GiB peak, {mem_stats.max_reserved_pct}% peak"
        )
        return f"{display_str}"


def upload_train_to_tensorboard(
    ckpt_dir, log_dir=None, train=True, eval=True
):
    from torch.utils.tensorboard import SummaryWriter
    from omegaconf import OmegaConf
    import json
    from pathlib import Path

    cfg = OmegaConf.load(Path(ckpt_dir) / "config.yaml")
    cfg = OmegaConf.to_container(cfg)

    if log_dir is None:
        log_dir = Path(ckpt_dir) / "tensorboard"

    if train:
        writer = SummaryWriter(log_dir=str(log_dir / "train"))
        with open(Path(ckpt_dir) / "metrics.jsonl") as f:
            for l in f:
                m = json.loads(l)
                step = m.get("global_step", 0)
                for key, value in m.items():
                    if key != "global_step" and isinstance(value, (int, float)):
                        writer.add_scalar(key, value, step)
        writer.close()

    if eval:
        writer = SummaryWriter(log_dir=str(log_dir / "eval"))
        with open(Path(ckpt_dir) / "metrics.eval.jsonl") as f:
            for l in f:
                m = json.loads(l)
                step = m.get("global_step", 0)
                for name, value in m.items():
                    if "/" in name and isinstance(value, (int, float)):
                        writer.add_scalar(f"evals/{name.replace('/','.')}", value, step)
        writer.close()


def get_num_params(model: nn.Module) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    numel = {n: p.numel() for n, p in model.named_parameters()}
    return sum(numel.values())
