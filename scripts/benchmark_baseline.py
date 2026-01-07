#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Benchmark baseline (pre-quantized) model.
Runs in its own process to ensure clean memory state.

Usage:
    python benchmark_baseline.py --model-class keras_hub.models.Gemma3CausalLM \
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
    get_dataset_text,
    build_token_dataloader,
    calculate_perplexity,
    save_model_and_size,
    benchmark_first_token_latency_ms,
    benchmark_generation_throughput,
    benchmark_peak_gpu_memory_bytes,
    save_results,
    gpu_reset_peaks,
    _gpu_mem_supported,
)


def get_model_class(class_path: str):
    """Dynamically import and return model class from string like 'keras_hub.models.Gemma3CausalLM'"""
    parts = class_path.rsplit('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model class path: {class_path}")
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline (pre-quantized) model")
    parser.add_argument("--model-class", required=True, help="Full path to model class, e.g., keras_hub.models.Gemma3CausalLM")
    parser.add_argument("--model-preset", required=True, help="Model preset name, e.g., gemma3_1b")
    parser.add_argument("--output-dir", default="/workspace/outputs", help="Directory for outputs")
    parser.add_argument("--dataset-name", default="wikitext2", help="Dataset for evaluation")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--eval-batches", type=int, default=50, help="Number of eval batches")
    parser.add_argument("--calib-samples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("BASELINE BENCHMARK (Pre-Quantized Model)")
    logging.info("=" * 60)
    logging.info("Model: %s / %s", args.model_class, args.model_preset)
    logging.info("Dataset: %s | seq_len=%d | eval_batches=%d", 
                 args.dataset_name, args.seq_len, args.eval_batches)

    # Reset GPU memory stats at the very start
    if _gpu_mem_supported():
        gpu_reset_peaks()

    # 1) Load model
    logging.info("Loading model from preset...")
    model_class = get_model_class(args.model_class)
    model = model_class.from_preset(args.model_preset)
    model_name = model.name or args.model_preset

    # 2) Load and prepare dataset
    logging.info("Loading evaluation dataset...")
    test_text = get_dataset_text(args.dataset_name, split="test")
    train_text = get_dataset_text(args.dataset_name, split="train")

    logging.info("Tokenizing...")
    all_tokens = model.preprocessor.tokenizer.tokenize(test_text)
    test_dataloader = build_token_dataloader(all_tokens, args.seq_len, max_batches=args.eval_batches)

    # Calibration dataset (will be saved for quantization step)
    calibration_dataset = [s.strip() + "." for s in train_text.split(".") if s.strip()][:args.calib_samples]
    gen_prompts = calibration_dataset if calibration_dataset else ["The quick brown fox"]

    # 3) Save baseline model and get disk size
    baseline_model_path = str(output_dir / f"{model_name}_baseline.keras")
    logging.info("Saving baseline model...")
    disk_size_bytes = save_model_and_size(model, baseline_model_path)

    # 4) Calculate perplexity
    logging.info("Calculating perplexity...")
    perplexity = calculate_perplexity(model, test_dataloader)
    logging.info("Baseline perplexity: %.4f", perplexity)

    # 5) Benchmark inference metrics
    logging.info("Benchmarking first-token latency...")
    # Run twice, use second (warmed up)
    _ = benchmark_first_token_latency_ms(model, gen_prompts[0])
    first_token_latency_ms = benchmark_first_token_latency_ms(model, gen_prompts[0])
    logging.info("First-token latency: %.3f ms", first_token_latency_ms)

    logging.info("Benchmarking throughput...")
    # Run twice, use second (warmed up)
    _ = benchmark_generation_throughput(model, gen_prompts[:min(4, len(gen_prompts))], target_new_tokens=50)
    throughput_tokens_per_sec = benchmark_generation_throughput(
        model, gen_prompts[:min(4, len(gen_prompts))], target_new_tokens=50
    )
    logging.info("Throughput: %.2f tokens/sec", throughput_tokens_per_sec)

    logging.info("Benchmarking peak GPU memory...")
    peak_gpu_mem_bytes = benchmark_peak_gpu_memory_bytes(model, gen_prompts[:1], max_length=args.seq_len)
    logging.info("Peak GPU memory: %s", human_bytes(peak_gpu_mem_bytes))

    # 6) Save results
    results = {
        "stage": "baseline",
        "model_class": args.model_class,
        "model_preset": args.model_preset,
        "model_name": model_name,
        "dataset_name": args.dataset_name,
        "seq_len": args.seq_len,
        "eval_batches": args.eval_batches,
        "calib_samples": args.calib_samples,
        "perplexity": perplexity,
        "disk_size_bytes": disk_size_bytes,
        "first_token_latency_ms": first_token_latency_ms,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "peak_gpu_mem_bytes": peak_gpu_mem_bytes,
        "baseline_model_path": baseline_model_path,
        # Save calibration data for quantization step
        "calibration_dataset": calibration_dataset[:100],  # Save subset to avoid huge JSON
    }
    
    results_path = str(output_dir / "baseline_results.json")
    save_results(results, results_path)

    # Also save full calibration dataset separately
    calib_path = str(output_dir / "calibration_dataset.json")
    save_results({"calibration_dataset": calibration_dataset}, calib_path)

    logging.info("=" * 60)
    logging.info("BASELINE BENCHMARK COMPLETE")
    logging.info("Results saved to: %s", results_path)
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
