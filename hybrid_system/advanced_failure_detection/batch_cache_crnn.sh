#!/bin/bash
#
# Batch cache CRNN predictions for 30 different microphone configurations
#

cd /Users/danieltoberman/Documents/git/Thesis/hybrid_system/advanced_failure_detection || exit 1

echo "=========================================="
echo "Batch CRNN Caching for 30 Configurations"
echo "=========================================="
echo ""

CONFIGS_2MIC=(
    "9,10,3,4,5,6,7,8,0"
    "1,10,11,4,5,6,7,8,0"
    "1,2,11,12,5,6,7,8,0"
    "9,2,3,4,13,6,7,8,0"
    "1,10,3,4,5,14,7,8,0"
    "9,2,11,4,5,6,7,8,0"
    "1,10,3,12,5,6,7,8,0"
    "1,2,11,4,13,6,7,8,0"
    "1,2,3,12,5,14,7,8,0"
    "1,2,3,4,13,6,15,8,0"
)

CONFIGS_3MIC=(
    "9,10,11,4,5,6,7,8,0"
    "1,10,11,12,5,6,7,8,0"
    "1,2,11,12,13,6,7,8,0"
    "9,2,3,12,5,6,15,8,0"
    "1,10,3,4,13,6,7,16,0"
    "9,2,11,4,5,14,7,8,0"
    "1,10,3,12,5,6,15,8,0"
    "9,2,3,4,13,6,15,8,0"
    "1,10,11,4,5,6,15,8,0"
    "9,10,3,12,5,6,7,16,0"
)

CONFIGS_4MIC=(
    "9,10,11,12,5,6,7,8,0"
    "1,10,11,12,13,6,7,8,0"
    "1,2,11,12,13,14,7,8,0"
    "9,2,11,4,13,6,15,8,0"
    "1,10,3,12,5,14,7,16,0"
    "9,10,11,4,5,14,15,8,0"
    "9,10,3,12,13,6,7,8,0"
    "1,10,11,4,13,14,7,8,0"
    "9,2,11,12,5,6,15,8,0"
    "1,10,3,4,13,14,15,8,0"
)

ALL_CONFIGS=("${CONFIGS_2MIC[@]}" "${CONFIGS_3MIC[@]}" "${CONFIGS_4MIC[@]}")

TOTAL=${#ALL_CONFIGS[@]}
COUNT=0

for MIC_CONFIG in "${ALL_CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    MIC_STR=$(echo "$MIC_CONFIG" | tr ',' '_')
    OUTPUT_FILE="features/crnn_results_mics_${MIC_STR}.pkl"

    if [ -f "$OUTPUT_FILE" ]; then
        echo "[$COUNT/$TOTAL] Skipping (exists): $MIC_CONFIG"
        continue
    fi

    echo "[$COUNT/$TOTAL] Processing: $MIC_CONFIG"

python3 - <<EOF
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from scipy import signal
from pathlib import Path
from tqdm import tqdm

sys.path.append("/Users/danieltoberman/Documents/git/Thesis/SSL")
from CRNN import CRNN

DATA_ROOT = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/extracted")
CSV_PATH = Path("/Users/danieltoberman/Documents/RealMAN_dataset_T60_08/test/test_static_source_location_08.csv")
CHECKPOINT_PATH = "/Users/danieltoberman/Documents/git/Thesis/08_CRNN/checkpoints/best_valid_loss0.0220.ckpt"
OUTPUT_PATH = Path("$OUTPUT_FILE")
MIC_ORDER_CRNN = [${MIC_CONFIG}]

CRNN_CONFIG = dict(win_len=512, nfft=512, win_shift_ratio=0.625, fre_range_used=range(1,257), eps=1e-6)

def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = CRNN()
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict({k.replace("model.","").replace("arch.",""):v for k,v in state.items()})
    model.eval()
    return model

def load_audio(path, mic_ids, fs=16000):
    base = path.replace("_CH0.wav","")
    sigs=[]
    for m in mic_ids:
        s,f = sf.read(f"{base}_CH{m}.wav",dtype="float32")
        if f!=fs: s=signal.resample(s,int(len(s)*fs/f))
        sigs.append(s)
    return np.stack(sigs)

model = load_model()
df = pd.read_csv(CSV_PATH)

results=[]

for idx,row in tqdm(df.iterrows(), total=len(df)):
    path = DATA_ROOT / row["filename"].replace(".flac","_CH0.wav")
    try:
        audio = load_audio(str(path), MIC_ORDER_CRNN)
    except:
        continue

    x = torch.from_numpy(audio.T).unsqueeze(0).float()
    with torch.no_grad():
        logits, feats = model.forward_with_intermediates(x)

    logits = logits[0]        # (T,360)
    feats  = feats[0]         # (T,256)

    assert feats.ndim==2 and feats.shape[1]==256, feats.shape
    assert logits.ndim==2 and logits.shape[1]==360, logits.shape

    probs = torch.sigmoid(logits)
    avg = probs.mean(dim=0)

    pred = int(torch.argmax(avg))
    err = abs(pred-row["angle(°)"])
    if err>180: err=360-err

    results.append(dict(
        sample_idx=idx,
        gt_angle=row["angle(°)"],
        crnn_pred=pred,
        crnn_error=err,
        logits_pre_sig=logits.cpu().numpy(),
        penultimate_features=feats.cpu().numpy(),
        predictions=probs.cpu().numpy(),
        avg_prediction=avg.cpu().numpy()
    ))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH,"wb") as f:
    pickle.dump(results,f)

print(f"Saved {len(results)} samples to {OUTPUT_PATH}")
EOF

done

echo "=========================================="
echo "Batch CRNN caching complete!"
echo "=========================================="
