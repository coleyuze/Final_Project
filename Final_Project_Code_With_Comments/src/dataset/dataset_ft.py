import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, WeightedRandomSampler
from transformers import AutoTokenizer


import torch

class FinetuneDataset(Dataset):
    def __init__(self, args, data_path, test_model = False):
        self.args = args
        self.data = pd.read_csv(data_path,sep='\t', quoting=csv.QUOTE_NONE)
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_model
        self.stance2label = {"AGAINST":0,"FAVOR":1,"NONE":2}
        self.label2stance = {"0":"AGAINST","1":"FAVOR","2":"NONE"}
        self.premise = args.premise
        
    def __getitem__(self,index):
        texta  = self.data['Tweet'].iloc[index]
        text = str(texta);
        claima = self.data['Claim'].iloc[index]
        claim=str(claima);
        input_data = self.tokenizer(text,claim,max_length=self.bert_seq_length, \
                                    padding='max_length', \
                                    truncation='longest_first')
        input_ids = torch.tensor(input_data['input_ids'],dtype=torch.long)
        attention_mask = torch.tensor(input_data['attention_mask'],dtype=torch.long)
        data = dict(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        if self.test_mode:
            return data
        if self.premise:
            premise = self.data['Premise'].iloc[index]
            label_id = torch.tensor(premise,dtype=torch.long)
        else:
            stance = self.data['Stance'].iloc[index]
            label = self.stance2label[stance]
            label_id = torch.tensor(label,dtype=torch.long)
        data['label'] = label_id
        return data
    def __len__(self):
        return self.data.shape[0]
    
    @classmethod
    def create_dataloaders(cls, args):# cls is the class parameter, args is the input parameter
        train_dataset = cls(args, args.train_path)# create the training dataset with args
        valid_dataset = cls(args, args.valid_path)# create the training dataset with args
        train_sampler = RandomSampler(train_dataset)# Encapsulate the training dataset as a RandomSampler to randomly sample the data
        valid_sampler = SequentialSampler(valid_dataset)# Encapsulate the validation dataset as a SequentialSampler to sequentially sample the data one by one
        
        train_dataloader = DataLoader(train_dataset,# Encapsulate the training dataset as DataLoader
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True,# True indicates that if the data is not evenly distributed across batches, the last batch with less than batch_size will be discarded. False Not discarded
                                        pin_memory=True)# Copy data to a fixed memory area for faster data loading
        valid_dataloader = DataLoader(valid_dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=valid_sampler,
                                    drop_last=False,
                                    pin_memory=True)
        print('The train data length: ',len(train_dataloader))
        print('The valid data length: ',len(valid_dataloader))
        
        return train_dataloader, valid_dataloader# Return dataloaders for the training and validation sets

