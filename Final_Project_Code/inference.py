from src.models import FtModel
from src.config import parse_args
from src.dataset import FinetuneDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, WeightedRandomSampler
import os
from tqdm import tqdm,trange
import pandas as pd
import torch
from src.utils import *
def inference():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    setup_device(args)
    setup_seed(args)
    test_dataset = FinetuneDataset(args, args.test_path, True)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=test_sampler,
                                    drop_last=False,
                                    pin_memory=True)
    
    print('The test data length: ',len(test_dataloader))
    model = FtModel(args)
    model = model.to(args.device)
    if args.distributed_train:
        model = torch.nn.parallel.DataParallel(model)
    ckpoint = torch.load(args.ckpt_file)
    model.load_state_dict(ckpoint['model_state_dict'])
    print("The epoch {} and the best mean f1 {:.4f} of the validation set.".format(ckpoint['epoch'],ckpoint['mean_f1']))
    
    if args.ema_start >= 0:
        ema = EMA(model, args.ema_decay)
        ema.resume(ckpoint['shadow'][0], ckpoint['backup'][0])
        # ema.shadow = 
        ema.apply_shadow()
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,desc="Evaluating")):
            for k in batch:
                batch[k] = batch[k].cuda()

            probability = model(batch,True)
            pred_label_id = torch.argmax(probability, dim=1)
            predictions.extend(pred_label_id.cpu().numpy())
    with open(f"data/{args.result_file}","w+") as f:
        if args.premise:
            f.write(f"id\ttext\tClaim\tPremise\n")
        else:
            f.write(f"id\ttext\tClaim\tStance\n")
        for i in trange(len(predictions)):
            i_d = test_dataset.data['id'].iloc[i]
            text = test_dataset.data['Tweet'].iloc[i]
            claim = test_dataset.data['Claim'].iloc[i]
            if args.premise:
                label = int(predictions[i])
            else:
                label = test_dataset.label2stance[str(int(predictions[i]))]

            
            f.write(f"{i_d}\t{text}\t{claim}\t{label}\n")
if __name__ == "__main__":
    inference()