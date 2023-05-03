# Final_Project
201918020227
Project operation requirements:
GPU: NVIDIA GeForce RTX 3090 with 24G of Video memory
CPU: Intel(R) Xeon(R) Gold 6142M CPU @ 2.60GHz

RoBERTa-base model:
python -m src.train --gpu_ids='2' --bert_seq_length=128 --learning_rate=1e-4 --bert_learning_rate=5e-5 --savedmodel_path='data/checkpoint/0703_swa_10' --max_epochs=10  --adam_epsilon=1e-6 --weight_decay=0.01 --test_batch_size=64 --val_batch_size=64 --batch_size=16 --swa_start=5 --swa_lr=2e-5 --fgm=1

covid-twitter-bert-v2 model: 
python -m src.train --gpu_ids='2' --bert_seq_length=128 --learning_rate=1e-4 --bert_learning_rate=5e-5 --savedmodel_path='data/checkpoint/0703_swa_10' --max_epochs=10  --adam_epsilon=1e-6 --weight_decay=0.01 --test_batch_size=64 --val_batch_size=64 --batch_size=16 --swa_start=5 --swa_lr=2e-5 --fgm=1 --bert_dir="digitalepidemiologylab/covid-twitter-bert-v2"
