#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Benchmark quantized model in a fresh process.
Ensures no memory contamination from previous steps.

Usage:
    python benchmark_quantized.py --output-dir ./benchmark_outputs
"""
import argparse
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
    get_model_disk_size,
    benchmark_first_token_latency_ms,
    benchmark_generation_throughput,
    benchmark_peak_gpu_memory_bytes,
    save_results,
    load_results,
    gpu_reset_peaks,
    _gpu_mem_supported,
)

import keras


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantized model (fresh process)")
    parser.add_argument("--output-dir", default="/workspace/outputs", help="Directory with outputs from previous steps")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    
    output_dir = Path(args.output_dir)

    logging.info("=" * 60)
    logging.info("QUANTIZED MODEL BENCHMARK (Fresh Process)")
    logging.info("=" * 60)

    # Load results from previous steps
    quant_results_path = output_dir / "quantization_results.json"
    baseline_results_path = output_dir / "baseline_results.json"
    
    if not quant_results_path.exists():
        logging.error("Quantization results not found at %s", quant_results_path)
        logging.error("Run quantize_and_save.py first!")
        return 1
    
    quant_info = load_results(str(quant_results_path))
    quantized_model_path = quant_info["quantized_model_path"]
    model_name = quant_info.get("model_name", "model")
    
    # Load baseline results for comparison
    baseline_info = {}
    if baseline_results_path.exists():
        baseline_info = load_results(str(baseline_results_path))
    
    dataset_name = baseline_info.get("dataset_name", "wikitext2")
    seq_len = baseline_info.get("seq_len", 128)
    eval_batches = baseline_info.get("eval_batches", 50)

    logging.info("Loading quantized model from: %s", quantized_model_path)
    logging.info("Dataset: %s | seq_len=%d | eval_batches=%d", dataset_name, seq_len, eval_batches)

    # Reset GPU memory stats at the very start - CRITICAL for clean measurement
    if _gpu_mem_supported():
        gpu_reset_peaks()
        logging.info("GPU memory stats reset for clean measurement")

    # 1) Load quantized model
    logging.info("Loading quantized model...")
    model = keras.saving.load_model(quantized_model_path)
    
    # Get disk size
    disk_size_bytes = get_model_disk_size(quantized_model_path)
    logging.info("Quantized model disk size: %s", human_bytes(disk_size_bytes))

    # 2) Load and prepare dataset
    logging.info("Loading evaluation dataset...")
    test_text = get_dataset_text(dataset_name, split="test")
    train_text = get_dataset_text(dataset_name, split="train")

    logging.info("Tokenizing...")
    all_tokens = model.preprocessor.tokenizer.tokenize(test_text)
    test_dataloader = build_token_dataloader(all_tokens, seq_len, max_batches=eval_batches)

    # Generation prompts
    calib_path = output_dir / "calibration_dataset.json"
    if calib_path.exists():
        calib_data = load_results(str(calib_path))
        gen_prompts = calib_data.get("calibration_dataset", ["The quick brown fox"])[:10]
    else:
        gen_prompts = [s.strip() + "." for s in train_text.split(".") if s.strip()][:10]

    # 3) Calculate perplexity
    logging.info("Calculating perplexity...")
    perplexity = calculate_perplexity(model, test_dataloader)
    logging.info("Quantized model perplexity: %.4f", perplexity)

    # 4) Benchmark inference metrics
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
    peak_gpu_mem_bytes = benchmark_peak_gpu_memory_bytes(model, gen_prompts[:1], max_length=seq_len)
    logging.info("Peak GPU memory: %s", human_bytes(peak_gpu_mem_bytes))

    # 5) Save results
    results = {
        "stage": "quantized",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "eval_batches": eval_batches,
        "perplexity": perplexity,
        "disk_size_bytes": disk_size_bytes,
        "first_token_latency_ms": first_token_latency_ms,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "peak_gpu_mem_bytes": peak_gpu_mem_bytes,
        "quantized_model_path": quantized_model_path,
    }
    
    results_path = str(output_dir / "quantized_benchmark_results.json")
    save_results(results, results_path)

    logging.info("=" * 60)
    logging.info("QUANTIZED BENCHMARK COMPLETE")
    logging.info("Results saved to: %s", results_path)
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
