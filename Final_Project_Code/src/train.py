from src.config import parse_args
from src.trainer import FtTrainer
import os
if __name__ == "__main__":
    args = parse_args()
    print("The GPUS ids: ",args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    trainer = FtTrainer(args)
    trainer.train()