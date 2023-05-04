#!/bin/bash
echo "Starting training..."
python -m src.train \
    --gpu_ids='2'\
    --bert_seq_length=128\
    --learning_rate=1e-4\
    --bert_learning_rate=2e-5\
    --savedmodel_path='data/checkpoint/0404_covid-twitter-bert'\
    --max_epochs=10\
    --swa_start=5\
    --fgm=1\
    --bert_dir="digitalepidemiologylab/covid-twitter-bert"\
