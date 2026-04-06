#!/bin/bash

CHECKPOINT="ckpts/pad/pad_best_ace.pth"
SPLIT_FILE="data/LivDet/livet_pad_splits.json"

SENSORS=(
    test_livdet2011_Biometrika
    test_livdet2011_Digital
    test_livdet2011_Italdata
    test_livdet2011_Sagem
    test_livdet2013_Biometrika
    test_livdet2013_CrossMatch
    test_livdet2013_Italdata
    test_livdet2015_CrossMatch
    test_livdet2015_DigitalPersona
    test_livdet2015_GreenBit
    test_livdet2015_HiScan
)

for SENSOR in "${SENSORS[@]}"; do
    echo "Running evaluation for $SENSOR"

    python evaluate_pad.py \
        --checkpoint "$CHECKPOINT" \
        --split "$SENSOR" \
        --output-dir "results/pad/$SENSOR" \
        --batch-size 128
done