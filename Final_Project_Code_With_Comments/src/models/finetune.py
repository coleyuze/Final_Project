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
        # If pre-trained model path is provided
        if args.pretrain_model_path is not None:
            # Print the path
            print(f"Continuously Pretrain model paths:{args.pretrain_model_path}")
            if "roberta" in  args.bert_dir:# If RoBERTa is in the BERT directory
                # Load the RoBERTa configuration and model
                self.config = RobertaConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = RobertaModel(self.config, add_pooling_layer=False)
            else:# Otherwise, load the BERT configuration and model
                self.config = BertConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = BertModel(self.config, add_pooling_layer=False)
            # Load the pre-trained model state dictionary
            ckpoint = torch.load(args.pretrain_model_path)
            self.roberta.load_state_dict(ckpoint["model_state_dict"])
        else:# If no pre-trained model path is provided
            # Load the auto configuration and model
            self.config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
            self.roberta = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        # self.att_head = AttentionHead(self.config.hidden_size * 4, self.config.hidden_size)
        if args.premise:# Define the classification layer based on the hidden size and class label The classification layer is defined based on the hidden size and the class label
            args.class_label = 2
        #nn.LinearDefine a linear layer of a neural network to create a linear layer for a given hidden layer dimension and class label
        self.cls = nn.Linear(self.config.hidden_size, args.class_label)
        # self.test_model = args.test_model
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if args.gamma_focal > 0:# If gamma focal loss is greater than 0, initialize the focal loss
            self.focal_loss = FocalLoss(class_num=args.class_label, gamma = args.gamma_focal)

    # Note:
    # 1. Define the forward propagation function
    # 2. Take input_data as input and generate output using roberta
    # 3. Extract roberta's last hidden layer, last_hidden_state, and all hidden_states
    # 4. Extract the last 4 hidden layers and multiply them with attention_mask of input_data and average them
    # 5. Calculate logits using cls layer
    # 6. Calculate the loss function, accuracy, and predicted label idpred_label_id based on the predictions
    # 7. In inference mode, only return probability
    def forward(self,input_data,inference=False):
        # Get the roberta model outputs, hidden states using input_data (input id and attention mask)
        outputs = self.roberta(input_data['input_ids'], input_data['attention_mask'], output_hidden_states = True)
        outputs_nohidden = self.roberta(input_data['input_ids'], input_data['attention_mask'],
                                        output_hidden_states=False)
        last_hidden_states = outputs.last_hidden_state
        # First evaluation method
        pooled_output = outputs_nohidden[1]  # 句向量
        # Get the last 4 hidden states of the output
        hidden_states = outputs.hidden_states
        # Set h12 as the last hidden state, i.e. get the last state from the array of hidden states
        h12 = hidden_states[-1]
        h11 = hidden_states[-2]
        h10 = hidden_states[-3]
        h09 = hidden_states[-4]
        # Calculate the average of h12 and multiply the attention mask by h12
        h12_mean = torch.mean(h12 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h11_mean = torch.mean(h11 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h10_mean = torch.mean(h10 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h09_mean = torch.mean(h09 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        # Calculate logits based on h12
        # logits = self.cls(h12_mean)
        # Calculate probability using softmax function
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
        with torch.no_grad():# Without gradient computation, compute the index of the maximum value of each row in the output logits as the predicted label
            # id denotes the index of the maximum value (per row) of the tensor in dimension dim=1 of the matrix
            pred_label_id = torch.argmax(logits, dim=1)# Calculate the accuracy between the predicted and true labels, normalized to 0-1
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        if self.args.use_rdrop:
            return loss, logits, accuracy, pred_label_id
        return loss, accuracy, pred_label_id
    
    
class AttentionHead(nn.Module):
    def __init__(self, cat_size, hidden_size=768):
        super().__init__()
        # Create two linear layers with cat_size as 1D input and hidden_size as 1D output
        # self.W are the input parameters, output parameters, and weights
        self.W = nn.Linear(cat_size, hidden_size)
        # Create a linear layer whose 1D input is hidden_size and 1D output is 1
        # self.V is the input parameters, output parameters, and weights
        self.V = nn.Linear(hidden_size, 1)
        # #1. Score first

    # #2. Amplify differences in scores (differentiation, e^x)
    # #3. Normalize to get the probability
    # #4. Calculate the loss
    def forward(self, hidden_states):
        # Calculate attention score
        att = torch.tanh(self.W(hidden_states))# Get attention score using linear representation W(h) and tanh
        score = self.V(att)# A function that gets the score using another linear representation V(W(h))
        att_w = torch.softmax(score, dim=1)# softmax score to get the vector att_w
        # compute the context vector
        context_vec = att_w * hidden_states# Multiply the hidden_state and attention weights
        context_vec = torch.sum(context_vec,dim=1)# Sum the results to get the context vector
        
        return context_vec