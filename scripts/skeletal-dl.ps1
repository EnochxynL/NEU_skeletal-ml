# ============================================================================
#  One-click training & evaluation for skeleton-based action recognition (DL)
#
#  Usage:
#    .\scripts\skeletal-dl.ps1 train              # Train all 7 experiments
#    .\scripts\skeletal-dl.ps1 train stgcn agcn   # Train specific models
#    .\scripts\skeletal-dl.ps1 eval               # Evaluate + compare
#    .\scripts\skeletal-dl.ps1 demo <file>        # Visualize prediction
#    .\scripts\skeletal-dl.ps1 all                # Train all + eval + compare
# ============================================================================
param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Remaining
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$env:PYTHONPATH = "$ProjectDir\src;$env:PYTHONPATH"

# -- Config paths --
$ConfigDir = "config/neu"
$CONFIGS = @{
    "stgcn"          = "$ConfigDir/train_joint_stgcn.yaml"
    "stgcn_aug"      = "$ConfigDir/train_joint_stgcn_aug.yaml"
    "stgcn_lr"       = "$ConfigDir/train_joint_stgcn_lr.yaml"
    "agcn"           = "$ConfigDir/train_joint_agcn.yaml"
    "agcn_dropout"   = "$ConfigDir/train_joint_agcn_dropout.yaml"
    "agcn_dropout2d" = "$ConfigDir/train_joint_agcn_dropout2d.yaml"
    "stgin"          = "$ConfigDir/train_joint_stgin.yaml"
    "resnet"         = "$ConfigDir/train_joint_resnet.yaml"
}
$ORDERED_KEYS = @("stgcn", "stgcn_aug", "stgcn_lr", "agcn", "agcn_dropout", "agcn_dropout2d", "stgin", "resnet")

# -- Checkpoint glob patterns (handles non-standard naming like neu_stgcn_joint_aug-*.pt) --
function Get-CkptPattern($key) {
    $CKPT_PATTERNS = @{
        "stgcn_aug"      = "runs/neu_stgcn_joint_aug-*.pt"
        "stgcn_lr"       = "runs/neu_stgcn_joint_lr-*.pt"
        "agcn_dropout"   = "runs/neu_agcn_joint_dropout-*.pt"
        "agcn_dropout2d" = "runs/neu_agcn_joint_dropout2d-*.pt"
    }
    if ($CKPT_PATTERNS.ContainsKey($key)) {
        return $CKPT_PATTERNS[$key]
    }
    return "runs/neu_${key}_joint-*.pt"
}

# -- Helpers --
function log($msg)   { Write-Host "[$(Get-Date -Format HH:mm:ss)] $msg" -ForegroundColor Green }
function warn($msg)  { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function err($msg)   { Write-Host "[ERROR] $msg" -ForegroundColor Red }
function cyan($msg)  { Write-Host $msg -ForegroundColor Cyan }

# ============================================================================
function Check-Env {
    log "Checking environment..."
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        err "uv not found. Install: pip install uv"
        exit 1
    }
    if (-not (Test-Path "pyproject.toml")) {
        err "Not in project root. Run from NEU_skeletal-ml\"
        exit 1
    }
    $pyVer = uv run python -c "import sys; print(sys.version.split()[0])"
    $torchInfo = uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    log "Python $pyVer, $torchInfo"
    log "Environment OK."
}

# ============================================================================
function Check-Data {
    log "Checking data..."
    if (-not (Test-Path "data/neu/train_data_joint.npy")) {
        warn "train_data_joint.npy not found, generating..."
        uv run skeletal-gendata
    }
    log "Data OK."
}

# ============================================================================
function Train-One {
    param([string]$key)

    $config = $CONFIGS[$key]
    if (-not (Test-Path $config)) {
        err "Config not found: $config"
        return
    }

    if ($key -eq "resnet") {
        $hasNnAudio = uv run python -c "import nnAudio" 2>$null
        if (-not $?) {
            warn "nnAudio not installed. Skipping ResNet training."
            warn "  Install with: uv sync"
            return
        }
    }

    log "Training: $key  (config: $config)"
    uv run skeletal-train-dl --config $config
    if ($LASTEXITCODE -ne 0) {
        warn "Training $key finished with exit code $LASTEXITCODE"
    } else {
        log "Done: $key"
    }
}

# ============================================================================
function Train-All {
    Check-Env
    Check-Data

    $keys = $ORDERED_KEYS
    if ($Remaining.Count -gt 0) {
        $keys = $Remaining
    }

    $total = $keys.Count
    $i = 1
    foreach ($key in $keys) {
        Write-Host ""
        cyan "========================================================"
        cyan "  [$i/$total]  $key"
        cyan "========================================================"
        Train-One $key
        if ($LASTEXITCODE -ne 0) {
            warn "Training $key failed (exit code $LASTEXITCODE), continuing..."
        }
        $i++
    }

    Write-Host ""
    log "========================================================"
    log "  All training done!"
    log "========================================================"
}

# ============================================================================
function Eval-All {
    Check-Env
    log "Running evaluations..."

    # 1. Pre-trained ST-GCN zero-shot evaluation
    log "1/3  Pre-trained ST-GCN zero-shot eval..."
    $PT_CKPT = "D:/ACTIVE/NEUAI/mmskeleton/checkpoints/st_gcn.ntu-xsub-300b57d4.pth"
    if (Test-Path $PT_CKPT) {
        uv run python src/skeletal_dl/evaluate_pretrained.py eval
    } else {
        warn "Pre-trained checkpoint not found: $PT_CKPT"
        warn "  Update CHECKPOINT_PATH in src/skeletal_dl/evaluate_pretrained.py"
    }

    # 2. Multi-model comparison (auto-discover checkpoints)
    log "2/3  Model comparison..."
    uv run python src/skeletal_dl/compare_models.py

    # 3. Per-model test (using trainer in test mode)
    log "3/3  Per-model test evaluation..."
    foreach ($key in $ORDERED_KEYS) {
        $config = $CONFIGS[$key]
        $latestCkpt = Get-ChildItem (Get-CkptPattern $key) -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestCkpt) {
            log "  Testing $key with $($latestCkpt.Name)"
            uv run skeletal-train-dl --config $config `
                --phase test --weights $latestCkpt.FullName `
                -model_saved_name "./runs/neu_${key}_joint_test"
        } else {
            warn "  No checkpoint found for $key (skip)"
        }
    }

    log "Evaluation complete."
    log "  Results: outputs/pretrained_stgcn_eval.txt"
    log "  Results: outputs/model_comparison.txt"
    log "  Charts:  outputs/model_comparison.png"
}

# ============================================================================
function Demo {
    Check-Env
    param([string]$file)

    if (-not $file) {
        $file = Get-ChildItem "data/test/*.skeleton" -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $file) {
            err "No .skeleton file specified and none found in data/test/"
            Write-Host "Usage: .\scripts\skeletal-dl.ps1 demo <path\to\file.skeleton>"
            exit 1
        }
        $file = $file.FullName
        log "Auto-selected demo file: $file"
    }
    if (-not (Test-Path $file)) {
        err "File not found: $file"
        exit 1
    }

    log "Running demo on: $file"

    # Pre-trained model
    log "  [1/2] Pre-trained ST-GCN..."
    uv run python src/skeletal_dl/demo_predict.py $file --gif

    # Best trained model (auto-find)
    log "  [2/2] Best trained checkpoint..."
    $bestCkpt = $null
    $bestModel = ""
    foreach ($key in @("stgcn_lr", "agcn_dropout", "stgin")) {
        $ckpt = Get-ChildItem (Get-CkptPattern $key) -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($ckpt) {
            $bestCkpt = $ckpt.FullName
            switch ($key) {
                { $_ -like "stgcn*" } { $bestModel = "skeletal_dl.model.stgcn.Model" }
                { $_ -like "agcn*" }  { $bestModel = "skeletal_dl.model.agcn_dropout.Model" }
                "stgin"              { $bestModel = "skeletal_dl.model.stgin.Model" }
            }
            break
        }
    }
    if ($bestCkpt) {
        uv run python src/skeletal_dl/demo_predict.py $file --gif `
            --checkpoint $bestCkpt --model $bestModel --num-class 10
    } else {
        warn "  No trained checkpoint found, skipping trained-model demo."
    }

    log "Demo complete. Outputs in outputs/"
}

# ============================================================================
function Cmd-All {
    Train-All
    Eval-All
    Demo
}

# ============================================================================
function Print-Help {
    Write-Host ""
    Write-Host "skeletal-dl.ps1 - One-click DL experiment pipeline"
    Write-Host ""
    Write-Host "Usage: .\scripts\skeletal-dl.ps1 <command> [args]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  train [model]     Train all models, or a specific one"
    Write-Host "                    models: stgcn | stgcn_aug | stgcn_lr | agcn |"
    Write-Host "                            agcn_dropout | stgin | resnet"
    Write-Host "  eval              Run all evaluations (pre-trained + comparison)"
    Write-Host "  demo <file>       Visualize prediction on a .skeleton file"
    Write-Host "  all               Train all + evaluate + demo"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\skeletal-dl.ps1 train                        # Train all 7"
    Write-Host "  .\scripts\skeletal-dl.ps1 train stgcn agcn             # Train two"
    Write-Host "  .\scripts\skeletal-dl.ps1 eval                         # Evaluate"
    Write-Host "  .\scripts\skeletal-dl.ps1 demo data\test\A005.skeleton"
    Write-Host ""
    Write-Host "Ordered experiments (train order):"
    for ($i = 0; $i -lt $ORDERED_KEYS.Count; $i++) {
        $key = $ORDERED_KEYS[$i]
        Write-Host "  $($i+1). $key  ->  $($CONFIGS[$key])"
    }
    Write-Host ""
}

# ============================================================================
# Main
# ============================================================================
switch ($Command) {
    "train" {
        Train-All
    }
    "eval" {
        Eval-All
    }
    "demo" {
        $demoFile = if ($Remaining.Count -gt 0) { $Remaining[0] } else { $null }
        Demo $demoFile
    }
    "all" {
        Cmd-All
    }
    { @("help", "--help", "-h") -contains $_ } {
        Print-Help
    }
    default {
        err "Unknown command: $Command"
        Print-Help
        exit 1
    }
}
