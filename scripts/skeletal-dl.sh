#!/bin/bash
# ============================================================================
#  One-click training & evaluation for skeleton-based action recognition (DL)
#
#  Usage:
#    bash scripts/skeletal-dl.sh train              # Train all 7 experiments
#    bash scripts/skeletal-dl.sh train stgcn agcn   # Train specific models
#    bash scripts/skeletal-dl.sh eval               # Evaluate + compare
#    bash scripts/skeletal-dl.sh demo <file>        # Visualize prediction
#    bash scripts/skeletal-dl.sh all                # Train all + eval + compare
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# -- Config paths --
CONFIG_DIR="config/neu"
declare -A CONFIGS=(
    ["stgcn"]="$CONFIG_DIR/train_joint_stgcn.yaml"
    ["stgcn_aug"]="$CONFIG_DIR/train_joint_stgcn_aug.yaml"
    ["stgcn_lr"]="$CONFIG_DIR/train_joint_stgcn_lr.yaml"
    ["agcn"]="$CONFIG_DIR/train_joint_agcn.yaml"
    ["agcn_dropout"]="$CONFIG_DIR/train_joint_agcn_dropout.yaml"
    ["stgin"]="$CONFIG_DIR/train_joint_stgin.yaml"
    ["resnet"]="$CONFIG_DIR/train_joint_resnet.yaml"
)
ORDERED_KEYS=("stgcn" "stgcn_aug" "stgcn_lr" "agcn" "agcn_dropout" "stgin" "resnet")

# -- Colors --
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
check_env() {
    log "Checking environment..."
    if ! command -v uv &>/dev/null; then
        err "uv not found. Install: pip install uv"
        exit 1
    fi
    if [ ! -f "pyproject.toml" ]; then
        err "Not in project root. Run from NEU_skeletal-ml/"
        exit 1
    fi
    uv run python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    log "Environment OK."
}

# ============================================================================
check_data() {
    log "Checking data..."
    if [ ! -f "data/neu/train_data_joint.npy" ]; then
        warn "train_data_joint.npy not found, generating..."
        uv run skeletal-gendata
    fi
    log "Data OK."
}

# ============================================================================
train_one() {
    local key="$1"
    local config="${CONFIGS[$key]}"

    if [ ! -f "$config" ]; then
        err "Config not found: $config"
        return 1
    fi

    # Skip resnet if nnAudio not available
    if [ "$key" = "resnet" ]; then
        if ! uv run python -c "import nnAudio" 2>/dev/null; then
            warn "nnAudio not installed. Skipping ResNet training."
            warn "  Install with: uv sync"
            return 0
        fi
    fi

    log "Training: $key  (config: $config)"
    uv run skeletal-train-dl --config "$config"
    log "Done: $key"
}

# ============================================================================
train_all() {
    check_env
    check_data

    local keys=("${ORDERED_KEYS[@]}")
    if [ $# -gt 0 ]; then
        keys=("$@")
    fi

    local total=${#keys[@]}
    local i=1
    for key in "${keys[@]}"; do
        echo ""
        echo -e "${CYAN}========================================================${NC}"
        echo -e "${CYAN}  [$i/$total]  $key${NC}"
        echo -e "${CYAN}========================================================${NC}"
        train_one "$key"
        i=$((i + 1))
    done

    echo ""
    echo -e "${GREEN}========================================================${NC}"
    echo -e "${GREEN}  All training done!${NC}"
    echo -e "${GREEN}========================================================${NC}"
}

# ============================================================================
eval_all() {
    check_env
    log "Running evaluations..."

    # 1. Pre-trained ST-GCN zero-shot evaluation
    log "1/3  Pre-trained ST-GCN zero-shot eval..."
    PT_CKPT="D:/ACTIVE/NEUAI/mmskeleton/checkpoints/st_gcn.ntu-xsub-300b57d4.pth"
    if [ -f "$PT_CKPT" ]; then
        uv run python src/skeletal_dl/evaluate_pretrained.py eval
    else
        warn "Pre-trained checkpoint not found: $PT_CKPT"
        warn "  Update CHECKPOINT_PATH in src/skeletal_dl/evaluate_pretrained.py"
    fi

    # 2. Multi-model comparison (auto-discover checkpoints)
    log "2/3  Model comparison..."
    uv run python src/skeletal_dl/compare_models.py

    # 3. Per-model test (using trainer in test mode)
    log "3/3  Per-model test evaluation..."
    for key in "${ORDERED_KEYS[@]}"; do
        local config="${CONFIGS[$key]}"
        local latest_ckpt=$(ls -t runs/neu_${key}_joint-*.pt 2>/dev/null | head -1)
        if [ -n "$latest_ckpt" ]; then
            log "  Testing $key with $latest_ckpt"
            uv run skeletal-train-dl --config "$config" \
                --phase test --weights "$latest_ckpt" \
                --model_saved_name "./runs/neu_${key}_joint_test" || true
        else
            warn "  No checkpoint found for $key (skip)"
        fi
    done

    log "Evaluation complete."
    log "  Results: outputs/pretrained_stgcn_eval.txt"
    log "  Results: outputs/model_comparison.txt"
    log "  Charts:  outputs/model_comparison.png"
}

# ============================================================================
demo() {
    check_env
    local file="$1"
    if [ -z "$file" ]; then
        file=$(ls data/test/*.skeleton 2>/dev/null | head -1)
        if [ -z "$file" ]; then
            err "No .skeleton file specified and none found in data/test/"
            echo "Usage: bash scripts/skeletal-dl.sh demo <path/to/file.skeleton>"
            exit 1
        fi
        log "Auto-selected demo file: $file"
    fi
    if [ ! -f "$file" ]; then
        err "File not found: $file"
        exit 1
    fi

    log "Running demo on: $file"

    # Pre-trained model
    log "  [1/2] Pre-trained ST-GCN..."
    uv run python src/skeletal_dl/demo_predict.py "$file" --gif

    # Best trained model (auto-find)
    log "  [2/2] Best trained checkpoint..."
    local best_ckpt=""
    local best_model=""
    for key in stgcn_lr agcn_dropout stgin; do
        local ckpt=$(ls -t runs/neu_${key}_joint-*.pt 2>/dev/null | head -1)
        if [ -n "$ckpt" ]; then
            best_ckpt="$ckpt"
            case "$key" in
                stgcn*|stgcn_lr) best_model="skeletal_dl.model.stgcn.Model" ;;
                agcn*)           best_model="skeletal_dl.model.agcn_dropout.Model" ;;
                stgin)           best_model="skeletal_dl.model.stgin.Model" ;;
            esac
            break
        fi
    done
    if [ -n "$best_ckpt" ]; then
        uv run python src/skeletal_dl/demo_predict.py "$file" --gif \
            --checkpoint "$best_ckpt" --model "$best_model" --num-class 10
    else
        warn "  No trained checkpoint found, skipping trained-model demo."
    fi

    log "Demo complete. Outputs in outputs/"
}

# ============================================================================
cmd_all() {
    train_all "$@"
    eval_all
}

# ============================================================================
print_help() {
    echo ""
    echo "skeletal-dl.sh — One-click DL experiment pipeline"
    echo ""
    echo "Usage: bash scripts/skeletal-dl.sh <command> [args]"
    echo ""
    echo "Commands:"
    echo "  train [model]     Train all models, or a specific one"
    echo "                    models: stgcn | stgcn_aug | stgcn_lr | agcn |"
    echo "                            agcn_dropout | stgin | resnet"
    echo "  eval              Run all evaluations (pre-trained + comparison)"
    echo "  demo <file>       Visualize prediction on a .skeleton file"
    echo "  all               Train all + evaluate"
    echo ""
    echo "Examples:"
    echo "  bash scripts/skeletal-dl.sh train                    # Train all 7"
    echo "  bash scripts/skeletal-dl.sh train stgcn agcn         # Train two"
    echo "  bash scripts/skeletal-dl.sh eval                     # Evaluate"
    echo "  bash scripts/skeletal-dl.sh demo data/test/A005.skeleton"
    echo ""
    echo "Ordered experiments (train order):"
    for i in "${!ORDERED_KEYS[@]}"; do
        local key="${ORDERED_KEYS[$i]}"
        echo "  $((i+1)). $key  ->  ${CONFIGS[$key]}"
    done
    echo ""
}

# ============================================================================
# Main
# ============================================================================
case "${1:-help}" in
    train)
        shift
        if [ $# -gt 0 ]; then
            train_all "$@"
        else
            train_all
        fi
        ;;
    eval)
        eval_all
        ;;
    demo)
        demo "$2"
        ;;
    all)
        shift
        cmd_all "$@"
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        err "Unknown command: ${1:-none}"
        print_help
        exit 1
        ;;
esac
