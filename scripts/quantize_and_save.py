#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Quantize model and save to disk.
Does NOT benchmark - that happens in a fresh process.

Usage:
    python quantize_and_save.py --model-class keras_hub.models.Gemma3CausalLM \
                                --model-preset gemma3_1b \
                                --output-dir ./benchmark_outputs
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

# Add parent dir to path for shared_utils
sys.path.insert(0, str(Path(__file__).parent))

from shared_utils import (
    setup_logging,
    human_bytes,
    save_model_and_size,
    save_results,
    load_results,
    profile_section,
    gpu_reset_peaks,
    gpu_peaks,
    _gpu_mem_supported,
    _PSUTIL_OK,
)


def get_model_class(class_path: str):
    """Dynamically import and return model class from string."""
    parts = class_path.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model class path: {class_path}")
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Quantize model with AWQ and save")
    parser.add_argument("--model-class", required=True, help="Full path to model class")
    parser.add_argument("--model-preset", required=True, help="Model preset name")
    parser.add_argument("--output-dir", default="/workspace/outputs", help="Directory for outputs")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for calibration")
    parser.add_argument("--calib-samples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--num-grid-points", type=int, default=20, help="AWQ grid search points")
    parser.add_argument("--group-size", type=int, default=128, help="Weight group size")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Keras components
    from keras.quantizers import AWQConfig

    logging.info("=" * 60)
    logging.info("QUANTIZATION STEP (AWQ)")
    logging.info("=" * 60)
    logging.info("Model: %s / %s", args.model_class, args.model_preset)
    logging.info("Config: num_grid_points=%d, group_size=%d, calib_samples=%d",
                 args.num_grid_points, args.group_size, args.calib_samples)

    # Load calibration dataset from previous step
    calib_path = output_dir / "calibration_dataset.json"
    if calib_path.exists():
        calib_data = load_results(str(calib_path))
        calibration_dataset = calib_data.get("calibration_dataset", [])
        logging.info("Loaded %d calibration samples from previous step", len(calibration_dataset))
    else:
        logging.warning("No calibration dataset found, will regenerate")
        calibration_dataset = None

    # Reset GPU memory stats
    if _gpu_mem_supported():
        gpu_reset_peaks()

    # 1) Load fresh model (not from saved - to ensure clean weights)
    logging.info("Loading fresh model from preset...")
    model_class = get_model_class(args.model_class)
    model = model_class.from_preset(args.model_preset)
    model_name = model.name or args.model_preset

    # 2) Generate calibration data if needed
    if not calibration_dataset:
        from shared_utils import get_dataset_text
        logging.info("Generating calibration dataset...")
        train_text = get_dataset_text("wikitext2", split="train")
        calibration_dataset = [s.strip() + "." for s in train_text.split(".") if s.strip()][:args.calib_samples]

    # 3) Configure AWQ
    awq_config = AWQConfig(
        dataset=calibration_dataset,
        tokenizer=(
            model.preprocessor.tokenizer
            if hasattr(model, "preprocessor")
            else None
        ),
        weight_bits=4,  # AWQ only supports 4-bit
        num_samples=args.calib_samples,
        sequence_length=args.seq_len,
        group_size=args.group_size,
        num_grid_points=args.num_grid_points,
    )

    # 4) Quantize with profiling
    if _PSUTIL_OK:
        import psutil
        import os
        pre_cpu = psutil.Process(os.getpid()).memory_info().rss
    else:
        pre_cpu = 0
    
    pre_gpu = gpu_peaks() if _gpu_mem_supported() else {}

    logging.info("Starting AWQ quantization...")
    with profile_section() as prof:
        model.quantize("awq", config=awq_config)
    
    logging.info("Quantization complete in %.2f seconds", prof["elapsed_sec"])

    if _PSUTIL_OK:
        import psutil
        import os
        post_cpu = psutil.Process(os.getpid()).memory_info().rss
    else:
        post_cpu = 0
    
    post_gpu = gpu_peaks() if _gpu_mem_supported() else {}

    # 5) Save quantized model
    quantized_model_path = str(output_dir / f"{model_name}_awq.keras")
    logging.info("Saving quantized model...")
    disk_size_bytes = save_model_and_size(model, quantized_model_path)

    # Get GPU peak during quantization
    quant_gpu_peak = 0
    gpu_stats = prof.get("gpu_stats", {})
    if gpu_stats:
        quant_gpu_peak = max(d.get("peak", 0) for d in gpu_stats.values())

    # 6) Save quantization results
    results = {
        "stage": "quantization",
        "model_class": args.model_class,
        "model_preset": args.model_preset,
        "model_name": model_name,
        "seq_len": args.seq_len,
        "calib_samples": args.calib_samples,
        "num_grid_points": args.num_grid_points,
        "group_size": args.group_size,
        "quantization_time_sec": prof["elapsed_sec"],
        "cpu_peak_bytes": prof["cpu_peak_bytes"],
        "gpu_peak_bytes": quant_gpu_peak,
        "pre_cpu_bytes": pre_cpu,
        "post_cpu_bytes": post_cpu,
        "pre_gpu_stats": pre_gpu,
        "post_gpu_stats": post_gpu,
        "quantized_model_path": quantized_model_path,
        "disk_size_bytes": disk_size_bytes,
    }
    
    results_path = str(output_dir / "quantization_results.json")
    save_results(results, results_path)

    logging.info("=" * 60)
    logging.info("QUANTIZATION COMPLETE")
    logging.info("Quantization time: %.2f sec", prof["elapsed_sec"])
    logging.info("CPU peak: %s", human_bytes(prof["cpu_peak_bytes"]))
    logging.info("GPU peak: %s", human_bytes(quant_gpu_peak))
    logging.info("Disk size: %s", human_bytes(disk_size_bytes))
    logging.info("Model saved to: %s", quantized_model_path)
    logging.info("Results saved to: %s", results_path)
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
