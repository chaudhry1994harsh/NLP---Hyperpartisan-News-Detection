#!/usr/bin/env bash

echo "Starting the prediction"

INPUTDIR=${1?Error no input given}
OUTDIR=${2?Error no output given}
MODELTYPE=${3?Error no model given}

echo "/home/zenith-kaju/miniconda3/envs/projenv/bin/python3.6 /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semval_Har.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR --modelType=$MODELTYPE"
#python /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semeval-pan-2019-random-baseline.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR
/home/zenith-kaju/miniconda3/envs/projenv/bin/python3.6 /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semval_Har.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR --modelType=$MODELTYPE