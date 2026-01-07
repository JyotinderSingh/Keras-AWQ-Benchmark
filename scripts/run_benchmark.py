#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWQ Benchmark Orchestrator

Runs benchmark phases in separate subprocesses for memory isolation.
Designed for Docker but works anywhere.
"""
import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_phase(phase_num: int, total: int, message: str):
    print(f"{Colors.BLUE}[PHASE {phase_num}/{total}] {message}{Colors.NC}")

def print_success(message: str):
    print(f"{Colors.GREEN}✓ {message}{Colors.NC}")

def print_error(message: str):
    print(f"{Colors.RED}✗ {message}{Colors.NC}")

def run_phase(script_name: str, args: list, phase_name: str) -> bool:
    """Run a benchmark phase in a subprocess."""
    script_dir = Path(__file__).parent
    script_path = script_dir / script_name
    
    cmd = [sys.executable, str(script_path)] + args
    
    print(f"{Colors.CYAN}Running: {' '.join(cmd)}{Colors.NC}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{phase_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"{phase_name} failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="AWQ Benchmark Orchestrator - Runs benchmarks with process isolation"
    )
    parser.add_argument("--model-class", required=True, 
                        help="Full path to model class (e.g., keras_hub.models.Gemma3CausalLM)")
    parser.add_argument("--model-preset", required=True,
                        help="Model preset name (e.g., gemma3_1b)")
    parser.add_argument("--output-dir", default="/workspace/outputs",
                        help="Output directory (default: /workspace/outputs)")
    parser.add_argument("--dataset-name", default="wikitext2",
                        help="Dataset for evaluation (default: wikitext2)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length (default: 128)")
    parser.add_argument("--eval-batches", type=int, default=50,
                        help="Number of eval batches (default: 50)")
    parser.add_argument("--calib-samples", type=int, default=128,
                        help="Calibration samples (default: 128)")
    parser.add_argument("--n-grid", type=int, default=20,
                        help="AWQ grid search points (default: 20)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Weight group size (default: 128)")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (default: INFO)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline benchmark")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip quantization step")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print()
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"{Colors.GREEN}AWQ BENCHMARK WITH PROCESS ISOLATION{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"Model:        {Colors.YELLOW}{args.model_class} / {args.model_preset}{Colors.NC}")
    print(f"Output:       {Colors.YELLOW}{args.output_dir}{Colors.NC}")
    print(f"Dataset:      {Colors.YELLOW}{args.dataset_name}{Colors.NC}")
    print(f"Seq Length:   {Colors.YELLOW}{args.seq_len}{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    print()
    
    start_time = time.time()
    
    # Phase 1: Baseline benchmark
    if args.skip_baseline:
        print_phase(1, 4, "SKIPPING baseline benchmark (--skip-baseline)")
    else:
        print_phase(1, 4, "Benchmarking BASELINE model (isolated process)...")
        baseline_args = [
            "--model-class", args.model_class,
            "--model-preset", args.model_preset,
            "--output-dir", args.output_dir,
            "--dataset-name", args.dataset_name,
            "--seq-len", str(args.seq_len),
            "--eval-batches", str(args.eval_batches),
            "--calib-samples", str(args.calib_samples),
            "--log-level", args.log_level,
        ]
        if not run_phase("benchmark_baseline.py", baseline_args, "Baseline benchmark"):
            return 1
        print_success("Baseline benchmark complete")
    print()
    
    # Phase 2: Quantization
    if args.skip_quantize:
        print_phase(2, 4, "SKIPPING quantization (--skip-quantize)")
    else:
        print_phase(2, 4, "Quantizing model with AWQ (isolated process)...")
        quant_args = [
            "--model-class", args.model_class,
            "--model-preset", args.model_preset,
            "--output-dir", args.output_dir,
            "--seq-len", str(args.seq_len),
            "--calib-samples", str(args.calib_samples),
            "--n-grid", str(args.n_grid),
            "--group-size", str(args.group_size),
            "--log-level", args.log_level,
        ]
        if not run_phase("quantize_and_save.py", quant_args, "Quantization"):
            return 1
        print_success("Quantization complete")
    print()
    
    # Phase 3: Quantized benchmark (CRITICAL: fresh process)
    print_phase(3, 4, "Benchmarking QUANTIZED model (FRESH isolated process)...")
    quantized_args = [
        "--output-dir", args.output_dir,
        "--log-level", args.log_level,
    ]
    if not run_phase("benchmark_quantized.py", quantized_args, "Quantized benchmark"):
        return 1
    print_success("Quantized benchmark complete")
    print()
    
    # Phase 4: Combine results
    print_phase(4, 4, "Combining results and generating report...")
    combine_args = [
        "--output-dir", args.output_dir,
        "--log-level", args.log_level,
    ]
    if not run_phase("combine_results.py", combine_args, "Combine results"):
        return 1
    print_success("Results combined")
    print()
    
    # Final summary
    elapsed = time.time() - start_time
    
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"{Colors.GREEN}BENCHMARK COMPLETE{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    print(f"Total time: {Colors.YELLOW}{elapsed:.1f} seconds{Colors.NC}")
    print()
    print("Output files:")
    print(f"  - {args.output_dir}/baseline_results.json")
    print(f"  - {args.output_dir}/quantization_results.json")
    print(f"  - {args.output_dir}/quantized_benchmark_results.json")
    print(f"  - {args.output_dir}/combined_results.json")
    print(f"  - {args.output_dir}/awq_benchmarks.csv")
    print()
    print(f"View results: {Colors.CYAN}cat {args.output_dir}/combined_results.json | python -m json.tool{Colors.NC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
