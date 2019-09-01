import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
MAX_SENTENCE_LENGTH = 1000
import nltk
import numpy as np
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
from torch.autograd import Variable
MAX_VOCAB = 50000
PAD_IDX = 0
UNK_IDX = 1


class Attention(nn.Module):
    def __init__(self, emb_size, num_classes, vocab_size):
        super(Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx = PAD_IDX)
        self.hidden_size = 50
        self.num_layers = 1
        self.gru = nn.GRU(input_size=emb_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first = True,bidirectional=True)
        self.att_weight = nn.Parameter(torch.rand(2*self.hidden_size,1))
        self.sm = nn.Softmax(dim=1)
        self.lin_bias = nn.Linear(2*self.hidden_size,2*self.hidden_size)
        self.dp = nn.Dropout(p=0.6) 
        self.linear1 = nn.Linear(2*self.hidden_size,500)
        self.linear2 = nn.Linear(500,1000)
        
        self.linear_out = nn.Linear(1000,num_classes)
        
    def init_hidden(self, batch_size):
        hidden = torch.randn(2*self.num_layers, batch_size, self.hidden_size)
        return hidden.cuda()
    
    def forward(self,x):
        bs = x.size(0)
        emb = self.embedding(x)
        hidden1 = self.init_hidden(bs)
        out, hidden = self.gru(emb,hidden1)
        
        mlp_out = F.tanh(self.lin_bias(out))

        attn_out = torch.bmm(mlp_out,self.att_weight.repeat(bs,1,1)).squeeze(dim=2)
        sm_out2 = self.sm(attn_out)
        
        word_attn_vectors = torch.bmm(sm_out2.unsqueeze(dim=1),out).squeeze(dim=1)
        
        out_o = F.relu(self.linear1(word_attn_vectors))
        out_o = self.dp(F.relu(self.linear2(out_o)))        
        out_o = self.linear_out(out_o)
        return out_o,self.sm(out_o),sm_out2,word_attn_vectors
        
