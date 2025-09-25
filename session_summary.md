# Session Summary: CRNN Training Setup + SRP-PHAT Analysis

## Context
- Working on thesis combining neural networks (CRNN) with classical methods (SRP-PHAT) for sound source localization
- Using M4 Pro Mac with MPS acceleration
- Dataset: RealMAN_dataset_T60_08 with 9-microphone circular array (6cm diameter)
- Goal: Find scenarios where classical methods complement neural networks

## Work Completed

### 1. CRNN Training Pipeline Setup
**Problem**: Original code was designed for CUDA/Linux, needed Mac/MPS compatibility

**Solutions Implemented**:
- Updated `requirements.txt` for Mac compatibility with MPS-compatible PyTorch
- Fixed device detection: `device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'`
- Fixed data type issues: converted float64 to float32 for MPS compatibility in `RecordData.py`
- Replaced Rich progress bar with TQDMProgressBar for stability
- Fixed strategy configuration: DDP doesn't work with MPS, switched to "auto" strategy

**Files Modified**:
- `/Users/danieltoberman/Documents/git/Thesis/requirements.txt`
- `/Users/danieltoberman/Documents/git/Thesis/SSL/run_CRNN.py`
- `/Users/danieltoberman/Documents/git/Thesis/SSL/Module.py`
- `/Users/danieltoberman/Documents/git/Thesis/SSL/RecordData.py`
- `/Users/danieltoberman/Documents/git/Thesis/SSL/utils/my_rich_progress_bar.py`

### 2. Weights & Biases Integration
**Problem**: Wandb logging not working due to incorrect argument parsing

**Solution**:
- Fixed argparse: `parser.add_argument("--use_wandb", action="store_true", default=False)` instead of `type=bool`
- Fixed config access: `use_wandb = bool(self.config.get('fit', {}).get('use_wandb', False))`
- Added proper logger switching logic and URL display

**Result**: Wandb now works correctly with `python run_CRNN.py fit --use_wandb`

### 3. SRP-PHAT Setup and Optimization
**Initial Setup**:
- Adapted `run_SRP.py` from Windows to Mac paths
- Fixed import issues (use `python -m xsrpMain.xsrp.run_SRP` syntax)
- Converted .flac to .wav file extensions

**Performance Issues Discovered**:
- Initial SRP-PHAT results were poor: ~70-150° errors
- CRNN achieves 4° MAE vs SRP-PHAT getting much higher errors

**Optimization Process**:
1. **Initial config**: 10 avg samples, 360 grid cells, gcc_phat_freq mode → ~90-170° errors
2. **Increased averaging**: 50 avg samples → ~45-120° errors
3. **Time domain**: gcc_phat_time, 100 avg samples → ~67-120° errors
4. **Final optimized config**:
   - 720 grid cells (0.5° resolution)
   - 200 averaging samples
   - gcc_phat_freq mode
   - 1024 DFT bins
   - **Result**: 8-20° errors for good cases, but still highly variable (1-175° range)

### 4. Key Findings

**CRNN Performance**: 4° MAE (highly consistent)
**Optimized SRP-PHAT Performance**:
- Best cases: 1-10° error
- Typical: 20-80° error
- Worst cases: 100-175° error
- **Very inconsistent** - performance varies dramatically by acoustic conditions

**Array Configuration**:
- 9 microphones in circular array
- 6cm diameter (very small for classical methods)
- 16 kHz sampling rate

## Current Status
- CRNN training pipeline fully functional on Mac with MPS
- Wandb integration working
- SRP-PHAT optimized but still underperforming vs CRNN
- Ready for full model training to get detailed CRNN results

## Next Steps (When Model is Trained)
1. **Analyze CRNN failures**: Get per-example CRNN results to identify failure cases
2. **Find complementary scenarios**: Identify where SRP-PHAT might add value:
   - Computational constraints
   - Out-of-distribution conditions
   - High uncertainty cases
   - Real-time requirements
3. **Design hybrid architecture**: Based on failure analysis, create systems that:
   - Use confidence-based switching
   - Employ ensemble methods
   - Provide computational/accuracy trade-offs

## Thesis Direction
**Research Goal**: Combine neural networks with classical methods for sound source localization
**Challenge**: CRNN significantly outperforms classical methods
**Strategy**: Find niche scenarios where classical methods provide value (computational efficiency, interpretability, out-of-distribution robustness, uncertainty estimation)

## Technical Environment
- **Hardware**: M4 Pro Mac
- **Python Environment**: `/Users/danieltoberman/Documents/git/Thesis/.venv/`
- **Key Dependencies**: PyTorch with MPS support, Lightning, Wandb
- **Dataset Path**: `/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/`
- **Working Directory**: `/Users/danieltoberman/Documents/git/Thesis/`

## Commands Ready to Use
```bash
# CRNN Training with wandb
python run_CRNN.py fit --use_wandb

# CRNN Training without wandb
python run_CRNN.py fit

# SRP-PHAT evaluation
python -m xsrpMain.xsrp.run_SRP --n 50 --random

# SRP-PHAT with plots
python -m xsrpMain.xsrp.run_SRP --idx <index> --plots --save_gcc --outdir <output_dir>
```