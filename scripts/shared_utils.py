# -*- coding: utf-8 -*-
"""
Shared utilities for AWQ benchmarking scripts.
Common functions used across isolated benchmark processes.
"""
import csv
import gc
import inspect
import io
import json
import logging
import os
import shutil
import tarfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

# ---------------------------
# Logging
# ---------------------------
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------------------
# Utilities: humanize bytes
# ---------------------------
def human_bytes(n: int | None) -> str:
    if n is None:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} EB"

# ---------------------------
# Dataset helpers
# ---------------------------
def get_dataset_text(dataset_name: str, split: str = "train") -> str:
    """Download and return text for small test corpora."""
    from datasets import load_dataset
    
    logging.info("Loading dataset '%s' split='%s'...", dataset_name, split)
    if dataset_name == "wikitext2":
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return "\n\n".join(d["text"] for d in raw_dataset)
    if dataset_name == "ptb":
        url = "https://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            file_path = f"./simple-examples/data/ptb.{split}.txt"
            text_bytes = tar.extractfile(file_path).read()
        return text_bytes.decode("utf-8")
    raise ValueError(f"Unsupported dataset name for testing: {dataset_name!r}")

def build_token_dataloader(
    all_tokens: np.ndarray, seq_len: int, max_batches: int = 50
) -> np.ndarray:
    """Slice a long token stream into [B, T] windows for perplexity evaluation."""
    samples = []
    for i in range(max_batches):
        start = i * seq_len
        end = start + seq_len
        if end > len(all_tokens):
            break
        samples.append(np.reshape(all_tokens[start:end], (1, seq_len)))
    if not samples:
        raise ValueError(
            f"Not enough tokens to build evaluation batches. "
            f"Need >= {seq_len}, got {len(all_tokens)}."
        )
    return np.array(samples, dtype=np.int32)

# ---------------------------
# Instrumentation: CPU/GPU memory + time
# ---------------------------
try:
    import psutil
    _PSUTIL_OK = True
except Exception:
    _PSUTIL_OK = False
    psutil = None

def get_backend():
    """Get current Keras backend."""
    return os.environ.get("KERAS_BACKEND", "tensorflow")

def _gpu_devices():
    backend = get_backend()
    if backend == "tensorflow":
        import tensorflow as tf
        try:
            return tf.config.list_physical_devices("GPU")
        except Exception:
            return []
    return []

def _gpu_mem_supported() -> bool:
    backend = get_backend()
    if backend == "tensorflow":
        import tensorflow as tf
        return hasattr(tf.config.experimental, "get_memory_info") and hasattr(
            tf.config.experimental, "reset_memory_stats"
        )
    return False

def gpu_reset_peaks():
    if not _gpu_mem_supported():
        return
    import tensorflow as tf
    for i, _ in enumerate(_gpu_devices()):
        tf.config.experimental.reset_memory_stats(f"GPU:{i}")

def gpu_peaks() -> dict[int, dict[str, int]]:
    """Returns {gpu_index: {'current': bytes, 'peak': bytes}} since last reset."""
    out: dict[int, dict[str, int]] = {}
    if not _gpu_mem_supported():
        return out
    import tensorflow as tf
    for i, _ in enumerate(_gpu_devices()):
        info = tf.config.experimental.get_memory_info(f"GPU:{i}")
        out[i] = {
            "current": int(info.get("current", 0)),
            "peak": int(info.get("peak", 0)),
        }
    return out

class CPUMemSampler:
    """Poll process RSS while running to estimate per-window peak main memory."""
    def __init__(self, interval_sec: float = 0.05):
        self.interval = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0
        self._proc = psutil.Process(os.getpid()) if _PSUTIL_OK else None

    def start(self):
        if self._proc is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._proc is None:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            time.sleep(self.interval)

    @property
    def peak_bytes(self) -> int | None:
        return self._peak if self._proc is not None else None

@contextmanager
def profile_section():
    """Context manager that captures wall time, CPU RSS peak, and GPU stats."""
    cpu = CPUMemSampler(interval_sec=0.05)
    cpu.start()
    
    have_gpu = len(_gpu_devices()) > 0 and _gpu_mem_supported()
    baseline_gpu = gpu_peaks() if have_gpu else {}
    if have_gpu:
        gpu_reset_peaks()
    
    t0 = time.perf_counter()
    results = {
        "elapsed_sec": None,
        "cpu_peak_bytes": None,
        "gpu_stats": {},
        "gpu_baseline": baseline_gpu,
    }
    try:
        yield results
    finally:
        elapsed = time.perf_counter() - t0
        cpu.stop()
        results["elapsed_sec"] = elapsed
        results["cpu_peak_bytes"] = cpu.peak_bytes
        results["gpu_stats"] = gpu_peaks() if have_gpu else {}

# ---------------------------
# Model saving & on-disk size
# ---------------------------
def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def _safe_remove(path: Path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        try:
            path.unlink()
        except FileNotFoundError:
            pass

def save_model_and_size(model, filepath: str) -> int:
    """Saves model to disk and returns on-disk size in bytes."""
    out_path = Path(filepath)
    _safe_remove(out_path)
    model.save(str(out_path))
    size_bytes = _path_size_bytes(out_path)
    logging.info("Saved model to %s (%s).", out_path, human_bytes(size_bytes))
    return size_bytes

def get_model_disk_size(filepath: str) -> int:
    """Get size of saved model on disk."""
    return _path_size_bytes(Path(filepath))

# ---------------------------
# Perplexity evaluation
# ---------------------------
def calculate_perplexity(model, dataloader: np.ndarray) -> float:
    """Compute perplexity on a token dataloader: [B, T] int32."""
    import keras
    from keras import ops
    from keras import losses
    
    logging.info("Evaluating perplexity on %d batches...", len(dataloader))
    total_nll = ops.zeros((), dtype="float32")
    total_tokens = ops.zeros((), dtype="float32")
    
    for batch in tqdm(dataloader, desc="PPL", leave=False):
        batch = ops.convert_to_tensor(batch, dtype="int32")
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        
        if hasattr(model, "preprocessor") and model.preprocessor is not None:
            inputs = {
                "token_ids": input_ids,
                "padding_mask": ops.ones_like(input_ids, dtype="bool"),
            }
        else:
            inputs = input_ids
            
        logits = model(inputs, training=False)
        mask = ops.cast(targets != 1, "float32")
        
        ce = losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
        ce = ce * mask
        total_nll = total_nll + ops.sum(ce)
        total_tokens = total_tokens + ops.sum(mask)
    
    avg_nll = total_nll / ops.maximum(total_tokens, 1.0)
    return float(ops.exp(avg_nll))

# ---------------------------
# Inference benchmarks
# ---------------------------
def _supports_arg(fn, argname: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return argname in sig.parameters
    except Exception:
        return False

def _extract_texts_from_generate_output(output):
    """Try to normalize model.generate outputs into a list[str]."""
    if output is None:
        return []
    if isinstance(output, list):
        if all(isinstance(x, str) for x in output):
            return output
        if all(isinstance(x, dict) and "text" in x for x in output):
            return [x["text"] for x in output]
    if isinstance(output, dict):
        val = output.get("text")
        if isinstance(val, str):
            return [val]
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return val
    return [str(output)]

def _token_count(model, text: str) -> int:
    try:
        tok = model.preprocessor.tokenizer.tokenize(text)
        return int(len(tok))
    except Exception:
        return int(len(text.split()))

def _generate(model, prompts: list[str], *, max_new_tokens: int | None = None, max_length: int | None = None):
    """Thin wrapper that tries max_new_tokens first, then max_length."""
    if hasattr(model, "generate"):
        try:
            if max_new_tokens is not None and _supports_arg(model.generate, "max_new_tokens"):
                return model.generate(prompts, max_new_tokens=max_new_tokens)
        except TypeError:
            pass
        if max_length is not None:
            return model.generate(prompts, max_length=max_length)
        return model.generate(prompts)
    raise AttributeError("Model has no .generate(...) method")

def benchmark_first_token_latency_ms(model, prompt: str) -> float:
    """Approximates first-token latency by measuring a single-token generation call."""
    try:
        # Warmup
        try:
            _ = _generate(model, [prompt], max_new_tokens=1)
        except Exception:
            pass
        
        t0 = time.perf_counter()
        if hasattr(model, "generate") and _supports_arg(model.generate, "max_new_tokens"):
            _ = _generate(model, [prompt], max_new_tokens=1)
        else:
            base_len = _token_count(model, prompt)
            _ = _generate(model, [prompt], max_length=base_len + 1)
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3
    except Exception as e:
        logging.warning("First-token latency benchmark failed: %s", e)
        return 0.0

def benchmark_generation_throughput(model, prompts: list[str], target_new_tokens: int = 50) -> float:
    """Measures tokens/sec over a short generation run."""
    try:
        # Warmup
        try:
            _ = _generate(model, prompts[:1], max_new_tokens=1)
        except Exception:
            pass
        
        t0 = time.perf_counter()
        outputs = _generate(model, prompts, max_new_tokens=target_new_tokens)
        t1 = time.perf_counter()
        elapsed = max(t1 - t0, 1e-6)
        
        outs = _extract_texts_from_generate_output(outputs)
        while len(outs) < len(prompts):
            outs.append(outs[-1] if outs else "")
        
        new_tokens = 0
        for p, o in zip(prompts, outs):
            new_tokens += max(_token_count(model, o) - _token_count(model, p), 0)
        
        return float(new_tokens) / float(elapsed)
    except Exception as e:
        logging.warning("Throughput benchmark failed: %s", e)
        return 0.0

def benchmark_peak_gpu_memory_bytes(model, prompts: list[str], max_length: int) -> int:
    """Measures peak GPU memory usage during inference. Returns bytes."""
    backend = get_backend()
    
    def generation_func():
        try:
            _ = model.generate(prompts, max_length=max_length)
        except TypeError:
            try:
                _ = model.generate(prompts, max_new_tokens=max(1, max_length or 1))
            except Exception:
                _ = model.generate(prompts)
    
    if backend == "tensorflow":
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            logging.warning("GPU not available for TensorFlow.")
            return 0
        
        if hasattr(tf.config.experimental, "reset_memory_stats"):
            tf.config.experimental.reset_memory_stats("GPU:0")
        
        generation_func()
        
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        return mem_info.get("peak", 0)
    
    elif backend == "torch":
        import torch
        if not torch.cuda.is_available():
            logging.warning("GPU not available for PyTorch.")
            return 0
        
        device = torch.device("cuda:0")
        try:
            model.to(device)
        except Exception:
            pass
        
        torch.cuda.reset_peak_memory_stats(device)
        generation_func()
        return torch.cuda.max_memory_allocated(device)
    
    elif backend == "jax":
        try:
            from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
            generation_func()
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            nvmlShutdown()
            return mem_info.used
        except ImportError:
            logging.warning("pynvml not found for JAX backend.")
            return 0
    
    return 0

# ---------------------------
# Results I/O
# ---------------------------
def save_results(results: dict, filepath: str):
    """Save results dict to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info("Saved results to %s", filepath)

def load_results(filepath: str) -> dict:
    """Load results dict from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

# ---------------------------
# CSV logging
# ---------------------------
CSV_HEADER = [
    "timestamp",
    "model_name",
    "model_preset",
    "dataset_name",
    "seq_len",
    "eval_batches",
    "calib_samples",
    "n_grid",
    "pre_perplexity",
    "post_perplexity",
    "perplexity_delta",
    "perplexity_delta_pct",
    "quant_time_sec",
    "quant_cpu_peak_bytes",
    "quant_gpu_peak_bytes",
    "disk_size_pre_bytes",
    "disk_size_post_bytes",
    "disk_size_delta_bytes",
    "disk_size_delta_pct",
    "pre_infer_peak_gpu_mem_bytes",
    "post_infer_peak_gpu_mem_bytes",
    "infer_gpu_mem_delta_bytes",
    "infer_gpu_mem_delta_pct",
    "pre_first_token_latency_ms",
    "post_first_token_latency_ms",
    "first_token_latency_delta_ms",
    "first_token_latency_delta_pct",
    "pre_throughput_tokens_per_sec",
    "post_throughput_tokens_per_sec",
    "throughput_delta_tokens_per_sec",
    "throughput_delta_pct",
]

def append_row_to_csv(csv_path: str, row: dict):
    path = Path(csv_path)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})
    logging.info("Appended results to %s", path)

def compute_delta(pre: float, post: float) -> tuple[float, float]:
    """Compute absolute delta and percentage change."""
    delta = post - pre
    if pre != 0:
        delta_pct = (delta / abs(pre)) * 100.0
    else:
        delta_pct = 0.0 if post == 0 else float('inf')
    return delta, delta_pct
