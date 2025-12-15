# Backup Guide for Computer Migration

## 1. CODE FILES (Add to Git)

### SSL Directory - Modified files to commit:
```bash
cd /Users/danieltoberman/Documents/git/Thesis/SSL
git add SingleTinyIPDnet.py         # Added forward_with_intermediates()
git add Module.py                    # Fixed float32 for MPS (line 637)
git add run_IPDnet_6cm.py           # New training script with batch printing
git add test_ipdnet_3x12cm.py       # Baseline evaluation script
git add RecordData.py               # If modified
git commit -m "IPDnet training setup and MPS compatibility fixes"
git push
```

### Hybrid System Directory - Scripts to commit:
```bash
cd /Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection
git add extract_ipdnet_features.py
git add compare_crnn_ipdnet.py
git add OOD_METHODS_README.md
git add research_summary.md
git add analyze_*.py
git add evaluate_*.py
git add *_routing.py
git commit -m "IPDnet feature extraction and comparison scripts"
git push
```

## 2. LARGE FILES (Save to Google Drive - TOO BIG FOR GIT)

### Critical Time-Consuming Outputs (MUST SAVE):

#### A. Model Checkpoints (~500MB-2GB each)
```
# CRNN checkpoint (already trained)
/Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/best_valid_loss0.0490.ckpt

# IPDnet checkpoint (TRAINING NOW - save when done!)
/Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/version_*/checkpoints/best*.ckpt
```
**Priority: CRITICAL** - Training takes 4-6 hours without GPU!

#### B. Extracted Features (~100-500MB each)
```
# CRNN features (already extracted)
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/test_3x12cm_consecutive_features.npz
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/train_6cm_features.npz

# IPDnet features (extract after training completes)
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/ipdnet/train_6cm_ipdnet_features.npz
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/ipdnet/test_3x12cm_consecutive_ipdnet_features.npz
```
**Priority: HIGH** - Feature extraction takes 30-60 minutes per dataset

#### C. SRP-PHAT Results (~50-100MB)
```
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/features/test_3x12cm_srp_results.pkl
```
**Priority: HIGH** - SRP computation takes 1-2 hours

#### D. OOD Method Results (~100-500MB total)
```
/Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection/results/
├── ood_distributions/          # Distribution analysis for all 19 methods
├── optimal_thresholds/         # Hybrid evaluation results
├── ood_methods/                # Individual method results (Energy, MC Dropout, etc.)
└── model_comparison/           # CRNN vs IPDnet comparison
```
**Priority: MEDIUM** - OOD analysis takes 2-4 hours total

### Optional (Can Regenerate):

#### E. Training Logs
```
/Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/  # All version folders
```
**Priority: LOW** - Useful for debugging but can be regenerated

## 3. DATASET (30-50GB - Keep on External Drive)

```
/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/
/Users/danieltoberman/Documents/RealMAN_9_channels/
```
**Note:** Dataset is too large for git or Google Drive. Keep on external drive or re-download from source.

## 4. RECOMMENDED GOOGLE DRIVE STRUCTURE

```
ThesisBackup/
├── checkpoints/
│   ├── crnn_best_valid_loss0.0490.ckpt
│   └── ipdnet_best_epoch_XX.ckpt
├── features/
│   ├── crnn/
│   │   ├── train_6cm_features.npz
│   │   └── test_3x12cm_consecutive_features.npz
│   └── ipdnet/
│       ├── train_6cm_ipdnet_features.npz
│       └── test_3x12cm_consecutive_ipdnet_features.npz
├── srp_results/
│   └── test_3x12cm_srp_results.pkl
└── ood_results/
    └── [all result directories]
```

## 5. QUICK BACKUP COMMANDS

### After IPDnet training completes:
```bash
# Find the best checkpoint
ls -lh /Users/danieltoberman/Documents/git/Thesis/SSL/lightning_logs/version_*/checkpoints/

# Upload to Google Drive (replace with your method)
# Option 1: Google Drive desktop app
cp /path/to/checkpoint ~/Google\ Drive/ThesisBackup/checkpoints/

# Option 2: rclone (if configured)
rclone copy /path/to/checkpoint gdrive:ThesisBackup/checkpoints/
```

## 6. VERIFICATION CHECKLIST

Before switching computers, verify you have:
- [ ] All modified .py files committed to git
- [ ] CRNN checkpoint saved
- [ ] IPDnet checkpoint saved (after training)
- [ ] CRNN features saved (train + test)
- [ ] IPDnet features saved (train + test, after extraction)
- [ ] SRP results saved
- [ ] OOD distribution results saved
- [ ] Dataset accessible (external drive or download link)

## 7. RESTORATION ON NEW COMPUTER

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Thesis.git
cd Thesis

# 2. Download large files from Google Drive to correct locations
# Place checkpoints in: SSL/lightning_logs/
# Place features in: hybrid_system/advanced_failure_detection/features/
# Place SRP results in: hybrid_system/advanced_failure_detection/features/

# 3. Install dependencies
pip install -r requirements.txt

# 4. Continue with evaluation (no retraining needed!)
cd hybrid_system/advanced_failure_detection
python3 evaluate_all_methods_optimal_thresholds.py --test_features features/ipdnet/test_3x12cm_consecutive_ipdnet_features.npz
```

## 8. ESTIMATED SAVINGS BY BACKING UP

| Item | Time to Regenerate | Size | Priority |
|------|-------------------|------|----------|
| IPDnet checkpoint | 4-6 hours (no GPU) | 500MB-1GB | CRITICAL |
| CRNN checkpoint | Already done | 500MB-1GB | CRITICAL |
| IPDnet features | 30-60 min | 200-400MB | HIGH |
| CRNN features | Already done | 200-400MB | HIGH |
| SRP results | 1-2 hours | 50-100MB | HIGH |
| OOD results | 2-4 hours | 100-500MB | MEDIUM |
| **TOTAL SAVINGS** | **8-13 hours** | **1.5-3.4GB** | - |
