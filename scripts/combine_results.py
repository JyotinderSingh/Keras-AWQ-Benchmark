#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Combine results from all steps and generate final report.

Usage:
    python combine_results.py --output-dir ./benchmark_outputs
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent dir to path for shared_utils
sys.path.insert(0, str(Path(__file__).parent))

from shared_utils import (
    setup_logging,
    human_bytes,
    load_results,
    save_results,
    compute_delta,
    append_row_to_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Combine benchmark results")
    parser.add_argument("--output-dir", default="/workspace/outputs", help="Directory with outputs")
    parser.add_argument("--csv-path", default="awq_benchmarks.csv", help="CSV file for results")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper()))
    
    output_dir = Path(args.output_dir)

    logging.info("=" * 60)
    logging.info("COMBINING BENCHMARK RESULTS")
    logging.info("=" * 60)

    # Load all results
    baseline_path = output_dir / "baseline_results.json"
    quant_path = output_dir / "quantization_results.json"
    quantized_bench_path = output_dir / "quantized_benchmark_results.json"

    missing = []
    if not baseline_path.exists():
        missing.append("baseline_results.json")
    if not quant_path.exists():
        missing.append("quantization_results.json")
    if not quantized_bench_path.exists():
        missing.append("quantized_benchmark_results.json")
    
    if missing:
        logging.error("Missing result files: %s", ", ".join(missing))
        logging.error("Run all benchmark steps first!")
        return 1

    baseline = load_results(str(baseline_path))
    quant = load_results(str(quant_path))
    quantized = load_results(str(quantized_bench_path))

    # Extract values
    pre_ppl = baseline["perplexity"]
    post_ppl = quantized["perplexity"]
    
    pre_disk = baseline["disk_size_bytes"]
    post_disk = quantized["disk_size_bytes"]
    
    pre_latency = baseline["first_token_latency_ms"]
    post_latency = quantized["first_token_latency_ms"]
    
    pre_throughput = baseline["throughput_tokens_per_sec"]
    post_throughput = quantized["throughput_tokens_per_sec"]
    
    pre_gpu_mem = baseline["peak_gpu_mem_bytes"]
    post_gpu_mem = quantized["peak_gpu_mem_bytes"]

    # Compute deltas
    ppl_delta, ppl_delta_pct = compute_delta(pre_ppl, post_ppl)
    disk_delta, disk_delta_pct = compute_delta(float(pre_disk), float(post_disk))
    latency_delta, latency_delta_pct = compute_delta(pre_latency, post_latency)
    throughput_delta, throughput_delta_pct = compute_delta(pre_throughput, post_throughput)
    gpu_delta, gpu_delta_pct = compute_delta(float(pre_gpu_mem), float(post_gpu_mem))

    # Print summary
    logging.info("")
    logging.info("=" * 70)
    logging.info("BENCHMARK COMPARISON: Baseline vs AWQ Quantized")
    logging.info("=" * 70)
    logging.info("Model: %s / %s", baseline.get("model_class", ""), baseline.get("model_preset", ""))
    logging.info("")
    
    logging.info("%-30s %15s %15s %15s %10s", "Metric", "Baseline", "Quantized", "Delta", "Delta %")
    logging.info("-" * 70)
    
    logging.info("%-30s %15.4f %15.4f %15.4f %9.2f%%", 
                 "Perplexity", pre_ppl, post_ppl, ppl_delta, ppl_delta_pct)
    
    logging.info("%-30s %15s %15s %15s %9.2f%%", 
                 "Disk Size", human_bytes(pre_disk), human_bytes(post_disk), 
                 human_bytes(int(disk_delta)), disk_delta_pct)
    
    logging.info("%-30s %14.2fms %14.2fms %14.2fms %9.2f%%", 
                 "First Token Latency", pre_latency, post_latency, latency_delta, latency_delta_pct)
    
    logging.info("%-30s %12.2f t/s %12.2f t/s %12.2f t/s %9.2f%%", 
                 "Throughput", pre_throughput, post_throughput, throughput_delta, throughput_delta_pct)
    
    logging.info("%-30s %15s %15s %15s %9.2f%%", 
                 "Peak GPU Memory", human_bytes(pre_gpu_mem), human_bytes(post_gpu_mem),
                 human_bytes(int(gpu_delta)), gpu_delta_pct)
    
    logging.info("-" * 70)
    logging.info("")
    logging.info("Quantization Time: %.2f seconds", quant["quantization_time_sec"])
    logging.info("Quantization CPU Peak: %s", human_bytes(quant.get("cpu_peak_bytes")))
    logging.info("Quantization GPU Peak: %s", human_bytes(quant.get("gpu_peak_bytes")))
    logging.info("")

    # Save combined results
    combined = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_class": baseline.get("model_class", ""),
        "model_preset": baseline.get("model_preset", ""),
        "model_name": baseline.get("model_name", ""),
        "dataset_name": baseline.get("dataset_name", ""),
        "seq_len": baseline.get("seq_len", 128),
        "eval_batches": baseline.get("eval_batches", 50),
        "calib_samples": quant.get("calib_samples", 128),
        "num_grid_points": quant.get("num_grid_points", 20),
        "group_size": quant.get("group_size", 128),
        
        # Baseline metrics
        "pre_perplexity": pre_ppl,
        "pre_disk_size_bytes": pre_disk,
        "pre_first_token_latency_ms": pre_latency,
        "pre_throughput_tokens_per_sec": pre_throughput,
        "pre_peak_gpu_mem_bytes": pre_gpu_mem,
        
        # Quantization metrics
        "quantization_time_sec": quant["quantization_time_sec"],
        "quant_cpu_peak_bytes": quant.get("cpu_peak_bytes"),
        "quant_gpu_peak_bytes": quant.get("gpu_peak_bytes"),
        
        # Quantized metrics
        "post_perplexity": post_ppl,
        "post_disk_size_bytes": post_disk,
        "post_first_token_latency_ms": post_latency,
        "post_throughput_tokens_per_sec": post_throughput,
        "post_peak_gpu_mem_bytes": post_gpu_mem,
        
        # Deltas
        "perplexity_delta": ppl_delta,
        "perplexity_delta_pct": ppl_delta_pct,
        "disk_size_delta_bytes": int(disk_delta),
        "disk_size_delta_pct": disk_delta_pct,
        "first_token_latency_delta_ms": latency_delta,
        "first_token_latency_delta_pct": latency_delta_pct,
        "throughput_delta_tokens_per_sec": throughput_delta,
        "throughput_delta_pct": throughput_delta_pct,
        "gpu_mem_delta_bytes": int(gpu_delta),
        "gpu_mem_delta_pct": gpu_delta_pct,
    }
    
    combined_path = str(output_dir / "combined_results.json")
    save_results(combined, combined_path)

    # Append to CSV
    csv_row = {
        "timestamp": combined["timestamp"],
        "model_name": combined["model_name"],
        "model_preset": combined["model_preset"],
        "dataset_name": combined["dataset_name"],
        "seq_len": combined["seq_len"],
        "eval_batches": combined["eval_batches"],
        "calib_samples": combined["calib_samples"],
        "num_grid_points": combined["num_grid_points"],
        "pre_perplexity": f"{pre_ppl:.6f}",
        "post_perplexity": f"{post_ppl:.6f}",
        "perplexity_delta": f"{ppl_delta:.6f}",
        "perplexity_delta_pct": f"{ppl_delta_pct:.2f}",
        "quant_time_sec": f"{quant['quantization_time_sec']:.6f}",
        "quant_cpu_peak_bytes": int(quant.get("cpu_peak_bytes") or 0),
        "quant_gpu_peak_bytes": int(quant.get("gpu_peak_bytes") or 0),
        "disk_size_pre_bytes": pre_disk,
        "disk_size_post_bytes": post_disk,
        "disk_size_delta_bytes": int(disk_delta),
        "disk_size_delta_pct": f"{disk_delta_pct:.2f}",
        "pre_infer_peak_gpu_mem_bytes": pre_gpu_mem,
        "post_infer_peak_gpu_mem_bytes": post_gpu_mem,
        "infer_gpu_mem_delta_bytes": int(gpu_delta),
        "infer_gpu_mem_delta_pct": f"{gpu_delta_pct:.2f}",
        "pre_first_token_latency_ms": f"{pre_latency:.3f}",
        "post_first_token_latency_ms": f"{post_latency:.3f}",
        "first_token_latency_delta_ms": f"{latency_delta:.3f}",
        "first_token_latency_delta_pct": f"{latency_delta_pct:.2f}",
        "pre_throughput_tokens_per_sec": f"{pre_throughput:.6f}",
        "post_throughput_tokens_per_sec": f"{post_throughput:.6f}",
        "throughput_delta_tokens_per_sec": f"{throughput_delta:.6f}",
        "throughput_delta_pct": f"{throughput_delta_pct:.2f}",
    }
    
    csv_path = output_dir / args.csv_path
    append_row_to_csv(str(csv_path), csv_row)

    logging.info("Combined results saved to: %s", combined_path)
    logging.info("CSV updated: %s", csv_path)
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
