#!/usr/bin/env bash

echo "Starting the prediction"

INPUTDIR=${1?Error no input given}
OUTDIR=${2?Error no output given}
MODELTYPE=${3?Error no model given}
echo "/home/zenith-kaju/miniconda3/condabin/conda init bash"
#/home/zenith-kaju/miniconda3/condabin/conda init bash

#source /home/zenith-kaju/.bashrc

echo "/home/zenith-kaju/miniconda3/condabin/conda activate projenv"
#/home/zenith-kaju/miniconda3/condabin/conda activate projenv

echo "python /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semeval-pan-2019-random-baseline.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR"
#python /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semeval-pan-2019-random-baseline.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR
/home/zenith-kaju/miniconda3/envs/projenv/bin/python3.6 /home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/semeval-pan-2019-random-baseline.py --inputDataset=$INPUTDIR --outputDir=$OUTDIR --modelType=$MODELTYPE