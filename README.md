# AWQ Quantization Benchmark Suite

Reproducible benchmarking for AWQ (Activation-aware Weight Quantization) in Keras/Keras-Hub.

**Features:**
- 🐳 One-command Docker run (no build required)
- 🔧 Test any keras/keras-hub fork at runtime
- 🧠 Process isolation prevents memory contamination
- 📊 Comprehensive metrics (perplexity, latency, throughput, memory)

## Quick Start

```bash
# Run benchmark with default keras/keras-hub
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b
```

Results are saved to `./outputs/`.

## Testing a Custom Fork (for PRs)

Pass your fork via environment variable — no rebuild needed:

```bash
# Test with custom keras fork
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -e KERAS_SOURCE="git+https://github.com/JyotinderSingh/keras@awq-2" \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b
```

```bash
# Test with both custom keras and keras-hub
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -e KERAS_SOURCE="git+https://github.com/USER/keras@BRANCH" \
    -e KERAS_HUB_SOURCE="git+https://github.com/USER/keras-hub@BRANCH" \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b
```

## For PR Reviewers

To verify benchmark results from a PR, just run:

```bash
# Use the fork URL provided by the PR author
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -e KERAS_SOURCE="git+https://github.com/PR_AUTHOR/keras@BRANCH" \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b

# Check results
cat outputs/combined_results.json | python -m json.tool
```

## Commands

```bash
# Full benchmark
docker run --gpus all -v $(pwd)/outputs:/workspace/outputs \
    jyotindersingh/awq-benchmark benchmark \
    --model-class CLASS --model-preset PRESET

# Show environment info
docker run --gpus all jyotindersingh/awq-benchmark info

# Interactive shell
docker run --gpus all -it jyotindersingh/awq-benchmark shell

# Individual phases (for debugging)
docker run ... jyotindersingh/awq-benchmark baseline --model-class CLASS --model-preset PRESET
docker run ... jyotindersingh/awq-benchmark quantize --model-class CLASS --model-preset PRESET
docker run ... jyotindersingh/awq-benchmark quantized
docker run ... jyotindersingh/awq-benchmark combine
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-class` | (required) | Full model class path |
| `--model-preset` | (required) | Model preset name |
| `--output-dir` | `/workspace/outputs` | Output directory |
| `--dataset-name` | `wikitext2` | Evaluation dataset |
| `--seq-len` | `128` | Sequence length |
| `--eval-batches` | `50` | Number of eval batches |
| `--calib-samples` | `128` | Calibration samples |
| `--num-grid-points` | `20` | AWQ grid search points |
| `--skip-baseline` | `false` | Skip baseline benchmark |
| `--skip-quantize` | `false` | Skip quantization |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KERAS_SOURCE` | Custom keras (e.g., `git+https://github.com/user/keras@branch`) |
| `KERAS_HUB_SOURCE` | Custom keras-hub (e.g., `git+https://github.com/user/keras-hub@branch`) |
| `KERAS_BACKEND` | Backend: `tensorflow` (default), `torch`, or `jax` |

## Output Files

```
outputs/
├── combined_results.json          # ← Full comparison (main result)
├── baseline_results.json          # Pre-quantized metrics
├── quantization_results.json      # Quantization stats
├── quantized_benchmark_results.json
├── awq_benchmarks.csv             # CSV for spreadsheets
├── {model}_baseline.keras         # Saved baseline model
└── {model}_awq.keras              # Saved quantized model
```

## Example Output

```
============================================================
BENCHMARK COMPARISON: Baseline vs AWQ Quantized
============================================================
Model: keras_hub.models.Gemma3CausalLM / gemma3_1b

Metric                         Baseline       Quantized       Delta    Delta %
----------------------------------------------------------------------
Perplexity                     172.4540       178.0304       5.5764     3.23%
Disk Size                        3.73 GB        1.12 GB     -2.61 GB  -69.94%
First Token Latency             23.73ms        18.42ms      -5.31ms  -22.38%
Throughput                   324.56 t/s     412.33 t/s    87.77 t/s   27.04%
Peak GPU Memory                  4.56 GB        1.75 GB     -2.81 GB  -61.62%
----------------------------------------------------------------------

Quantization Time: 228.09 seconds
```

## Why Process Isolation?

Benchmarking both models in a single process causes memory contamination:
- GPU memory pools retain allocations
- JIT caches persist
- `gc.collect()` doesn't fully release GPU memory

This suite runs **each phase in a separate subprocess**, matching the approach used by MLPerf and AutoGPTQ.

## Caching HuggingFace Models

To avoid re-downloading models on each run:

```bash
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b
```

## Supported Models

Any Keras-Hub causal LM with `.quantize("awq", ...)` support:

- `keras_hub.models.Gemma3CausalLM` (gemma3_1b, etc.)
- `keras_hub.models.GPT2CausalLM` (gpt2_base_en, etc.)
- `keras_hub.models.OPTCausalLM` (opt_125m_en, etc.)
- `keras_hub.models.LlamaCausalLM` (llama2_7b_en, etc.)

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA

## Troubleshooting

**No GPUs detected:**
```bash
# Verify nvidia-docker works
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**CUDA out of memory:**
- Use `--seq-len 64` and `--eval-batches 20`
- Try a smaller model preset

## License

Apache 2.0
