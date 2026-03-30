#!/bin/bash

CHECKPOINT="ckpts/recog/recog_best_eer.pth"
SPLIT_FILE="data/FVC/fvc_splits.json"

DBS=(
    test_fvc2000_db1
    test_fvc2000_db2
    test_fvc2002_db1
    test_fvc2002_db2
    test_fvc2002_db3
    test_fvc2004_db1
    test_fvc2004_db2
)

for DB in "${DBS[@]}"; do
    echo "Running evaluation for $DB"

    python evaluate_recog.py \
        --checkpoint "$CHECKPOINT" \
        --split-file "$SPLIT_FILE" \
        --split "$DB" \
        --output-dir "results/recog/$DB" \
        --batch-size 128
done

SPLIT_FILE="data/SD302/sd302_splits.json"
DB=test

echo "Running evaluation for test_sd302"

python evaluate_recog.py \
    --checkpoint "$CHECKPOINT" \
    --split-file "$SPLIT_FILE" \
    --split "$DB" \
    --output-dir "results/recog/test_sd302" \
    --batch-size 128