import logging
from multiprocessing.reduction import ForkingPickler
import random

import sys
import os
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import dataloader



def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_pretrain_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 'lr': args.learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler
def build_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    roberta = ["roberta"]
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and not any(item in n for item in  roberta))],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},

        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and any(item in n for item in  roberta))], 
         'weight_decay': args.weight_decay, 'lr': args.bert_learning_rate},

        {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and (not any(item in n for item in  roberta)) )], 
         'weight_decay': 0.0, 'lr': args.learning_rate},

        {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and any(item in n for item in  roberta))], 
         'weight_decay': 0.0, 'lr': args.bert_learning_rate}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    args.constant_lr = False
    if args.constant_lr:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler


def build_grouped_parameters_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']    
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': args.weight_decay, 'lr': args.learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': args.weight_decay, 'lr': args.learning_rate*2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': args.learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': args.learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if args.model_type not in n], 'lr':args.learning_rate*20, "weight_decay": 0.0},
    ]
    return optimizer_grouped_parameters