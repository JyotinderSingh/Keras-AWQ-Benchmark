#!/bin/bash
# =============================================================================
# Entrypoint for AWQ Benchmark Container
# =============================================================================
# Supports runtime installation of custom keras/keras-hub via environment vars:
#   KERAS_SOURCE="git+https://github.com/user/keras@branch"
#   KERAS_HUB_SOURCE="git+https://github.com/user/keras-hub@branch"
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_banner() {
    echo -e "${CYAN}"
    echo "============================================================"
    echo "  AWQ Quantization Benchmark Suite"
    echo "============================================================"
    echo -e "${NC}"
}

# Install custom keras/keras-hub if specified via environment variables
install_custom_packages() {
    local installed_custom=false
    
    if [[ -n "${KERAS_SOURCE:-}" && "${KERAS_SOURCE}" != "default" ]]; then
        echo -e "${YELLOW}Installing custom Keras: ${KERAS_SOURCE}${NC}"
        pip uninstall -y keras 2>/dev/null || true
        pip install --no-cache-dir "${KERAS_SOURCE}"
        installed_custom=true
    fi
    
    if [[ -n "${KERAS_HUB_SOURCE:-}" && "${KERAS_HUB_SOURCE}" != "default" ]]; then
        echo -e "${YELLOW}Installing custom Keras-Hub: ${KERAS_HUB_SOURCE}${NC}"
        pip uninstall -y keras-hub keras-nlp 2>/dev/null || true
        pip install --no-cache-dir "${KERAS_HUB_SOURCE}"
        installed_custom=true
    fi
    
    if [[ "$installed_custom" = true ]]; then
        echo -e "${GREEN}Custom package installation complete${NC}"
        echo ""
    fi
}

print_env_info() {
    echo -e "${BLUE}Environment Information:${NC}"
    echo -e "  Python:     $(python --version 2>&1)"
    echo -e "  Keras:      $(python -c 'import keras; print(keras.__version__)' 2>/dev/null || echo 'not installed')"
    echo -e "  Keras-Hub:  $(python -c 'import keras_hub; print(keras_hub.__version__)' 2>/dev/null || echo 'not installed')"
    echo -e "  TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'not installed')"
    echo -e "  Backend:    ${KERAS_BACKEND:-tensorflow}"
    echo ""
    
    if [[ -n "${KERAS_SOURCE:-}" && "${KERAS_SOURCE}" != "default" ]]; then
        echo -e "${BLUE}Custom Keras Source:${NC} ${KERAS_SOURCE}"
    fi
    if [[ -n "${KERAS_HUB_SOURCE:-}" && "${KERAS_HUB_SOURCE}" != "default" ]]; then
        echo -e "${BLUE}Custom Keras-Hub Source:${NC} ${KERAS_HUB_SOURCE}"
    fi
    echo ""
    
    # Check GPU
    echo -e "${BLUE}GPU Status:${NC}"
    python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  Found {len(gpus)} GPU(s):')
    for gpu in gpus:
        print(f'    - {gpu.name}')
else:
    print('  No GPUs detected')
" 2>/dev/null || echo "  Could not detect GPUs"
    echo ""
}

print_usage() {
    echo -e "${GREEN}Usage:${NC}"
    echo ""
    echo "  docker run --gpus all jyotindersingh/awq-benchmark [command] [options]"
    echo ""
    echo -e "${GREEN}Commands:${NC}"
    echo "  benchmark    Run the full benchmark suite"
    echo "  baseline     Run only baseline (pre-quantized) benchmark"
    echo "  quantize     Run only quantization step"
    echo "  quantized    Run only quantized model benchmark"
    echo "  combine      Combine results from previous runs"
    echo "  info         Show environment information"
    echo "  shell        Start interactive shell"
    echo "  --help       Show this help message"
    echo ""
    echo -e "${GREEN}Benchmark Options:${NC}"
    echo "  --model-class CLASS     Model class (e.g., keras_hub.models.Gemma3CausalLM)"
    echo "  --model-preset PRESET   Model preset (e.g., gemma3_1b)"
    echo "  --output-dir DIR        Output directory (default: /workspace/outputs)"
    echo "  --seq-len N             Sequence length (default: 128)"
    echo "  --eval-batches N        Number of eval batches (default: 50)"
    echo "  --calib-samples N       Calibration samples (default: 128)"
    echo "  --num-grid-points N              AWQ grid search points (default: 20)"
    echo "  --skip-baseline         Skip baseline benchmark"
    echo "  --skip-quantize         Skip quantization step"
    echo ""
    echo -e "${GREEN}Environment Variables for Custom Packages:${NC}"
    echo "  KERAS_SOURCE            Custom keras source (git URL or PyPI spec)"
    echo "  KERAS_HUB_SOURCE        Custom keras-hub source (git URL or PyPI spec)"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo ""
    echo "  # Run with default packages"
    echo "  docker run --gpus all -v \$(pwd)/outputs:/workspace/outputs \\"
    echo "    jyotindersingh/awq-benchmark benchmark \\"
    echo "    --model-class keras_hub.models.Gemma3CausalLM --model-preset gemma3_1b"
    echo ""
    echo "  # Run with custom keras fork"
    echo "  docker run --gpus all -v \$(pwd)/outputs:/workspace/outputs \\"
    echo "    -e KERAS_SOURCE='git+https://github.com/JyotinderSingh/keras@awq-2' \\"
    echo "    jyotindersingh/awq-benchmark benchmark \\"
    echo "    --model-class keras_hub.models.Gemma3CausalLM --model-preset gemma3_1b"
    echo ""
    echo "  # Show environment info"
    echo "  docker run --gpus all jyotindersingh/awq-benchmark info"
    echo ""
}

# Install custom packages if specified
install_custom_packages

# Main logic
case "${1:-}" in
    benchmark)
        shift
        print_banner
        print_env_info
        echo -e "${GREEN}Starting full benchmark...${NC}"
        echo ""
        exec python /workspace/scripts/run_benchmark.py "$@"
        ;;
    baseline)
        shift
        print_banner
        print_env_info
        echo -e "${GREEN}Running baseline benchmark...${NC}"
        exec python /workspace/scripts/benchmark_baseline.py "$@"
        ;;
    quantize)
        shift
        print_banner
        print_env_info
        echo -e "${GREEN}Running quantization...${NC}"
        exec python /workspace/scripts/quantize_and_save.py "$@"
        ;;
    quantized)
        shift
        print_banner
        print_env_info
        echo -e "${GREEN}Running quantized benchmark...${NC}"
        exec python /workspace/scripts/benchmark_quantized.py "$@"
        ;;
    combine)
        shift
        print_banner
        echo -e "${GREEN}Combining results...${NC}"
        exec python /workspace/scripts/combine_results.py "$@"
        ;;
    info)
        print_banner
        print_env_info
        ;;
    shell)
        print_banner
        print_env_info
        echo -e "${GREEN}Starting interactive shell...${NC}"
        exec /bin/bash
        ;;
    --help|-h|"")
        print_banner
        print_env_info
        print_usage
        ;;
    *)
        # Pass through to python or bash
        exec "$@"
        ;;
esac
