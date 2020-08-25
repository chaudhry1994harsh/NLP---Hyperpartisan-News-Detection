#!/usr/bin/env bash

echo "Starting the prediction"

INPUTDIR=${1?Error no input given}
OUTDIR=${2?Error no output given}

echo "conda activate projenv"
conda activate projenv

echo "python /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semeval-pan-2019-meta-random-baseline.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR"
python /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semeval-pan-2019-meta-random-baseline.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR