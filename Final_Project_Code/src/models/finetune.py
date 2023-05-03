import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig,RobertaModel
from transformers import BertConfig,BertModel
import numpy as np
from src.utils import FocalLoss
from transformers import AutoModel, AutoConfig

class FtModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.pretrain_model_path is not None:
            print(f"Continuous pre-training model:{args.pretrain_model_path}")
            if "roberta" in  args.bert_dir:
                self.config = RobertaConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = RobertaModel(self.config, add_pooling_layer=False) 
            else:
                self.config = BertConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = BertModel(self.config, add_pooling_layer=False) 
            
            ckpoint = torch.load(args.pretrain_model_path)
            self.roberta.load_state_dict(ckpoint["model_state_dict"])
        else:
            self.config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
            self.roberta = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        if args.premise:
            args.class_label = 2
        self.cls = nn.Linear(self.config.hidden_size, args.class_label)
        if args.gamma_focal > 0:
            self.focal_loss = FocalLoss(class_num=args.class_label, gamma = args.gamma_focal)

        
    def forward(self,input_data,inference=False):
        outputs = self.roberta(input_data['input_ids'], input_data['attention_mask'], output_hidden_states = True)
        last_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        h12 = hidden_states[-1]
        h11 = hidden_states[-2]
        h10 = hidden_states[-3]
        h09 = hidden_states[-4]

        h12_mean = torch.mean(h12 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h11_mean = torch.mean(h11 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h10_mean = torch.mean(h10 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h09_mean = torch.mean(h09 * input_data['attention_mask'].unsqueeze(-1) , dim=1)

        logits = self.cls(h12_mean)
        probability = nn.functional.softmax(logits)
        if inference:
            return probability
        loss, accuracy, pred_label_id = self.cal_loss(logits, input_data['label'])
        return loss, accuracy, pred_label_id
        
        

    def cal_loss(self, logits, label):
        if self.args.gamma_focal > 0:
            loss = self.focal_loss(logits, label)
        else:
            loss = F.cross_entropy(logits, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(logits, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        if self.args.use_rdrop:
            return loss, logits, accuracy, pred_label_id
        return loss, accuracy, pred_label_id
    
    
class AttentionHead(nn.Module):
    def __init__(self, cat_size, hidden_size=768):
        super().__init__()
        self.W = nn.Linear(cat_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)        
        
    def forward(self, hidden_states):
        att = torch.tanh(self.W(hidden_states))
        score = self.V(att)
        att_w = torch.softmax(score, dim=1)
        context_vec = att_w * hidden_states
        context_vec = torch.sum(context_vec,dim=1)
        
        return context_vec