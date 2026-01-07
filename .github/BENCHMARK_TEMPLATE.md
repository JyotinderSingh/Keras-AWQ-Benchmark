# AWQ Benchmark Template for PRs

Use this template when submitting PRs that modify AWQ quantization.

## How to Run the Benchmark

```bash
# Run with your keras fork (replace USER/BRANCH)
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -e KERAS_SOURCE="git+https://github.com/YOUR_USER/keras@YOUR_BRANCH" \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b

# View results
cat outputs/combined_results.json | python -m json.tool
```

---

## PR Description Template

Copy this into your PR description:

````markdown
## AWQ Benchmark Results

**Test Configuration:**
- Model: `keras_hub.models.Gemma3CausalLM / gemma3_1b`
- Dataset: wikitext2
- Sequence Length: 128
- Hardware: [YOUR GPU, e.g., NVIDIA A100 40GB]

**Tested With:**
```bash
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -e KERAS_SOURCE="git+https://github.com/YOUR_USER/keras@YOUR_BRANCH" \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b
```

### Results

| Metric | Baseline | Quantized | Delta | Change |
|--------|----------|-----------|-------|--------|
| Perplexity | X.XX | X.XX | +X.XX | +X.XX% |
| Disk Size | X.XX GB | X.XX GB | -X.XX GB | -XX.XX% |
| First Token Latency | X.XX ms | X.XX ms | -X.XX ms | -XX.XX% |
| Throughput | X.XX tok/s | X.XX tok/s | +X.XX tok/s | +XX.XX% |
| Peak GPU Memory | X.XX GB | X.XX GB | -X.XX GB | -XX.XX% |

**Quantization Stats:**
- Time: X.XX seconds
- Peak GPU Memory: X.XX GB

<details>
<summary>Full JSON Results</summary>

```json
// Paste outputs/combined_results.json here
```

</details>
````

---

## For Reviewers

To verify these results:

```bash
docker run --gpus all \
    -v $(pwd)/outputs:/workspace/outputs \
    -e KERAS_SOURCE="git+https://github.com/PR_AUTHOR/keras@BRANCH" \
    jyotindersingh/awq-benchmark benchmark \
    --model-class keras_hub.models.Gemma3CausalLM \
    --model-preset gemma3_1b
```
