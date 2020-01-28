import os
import sys
import torch
from utils.evaluation import *
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class Net(nn.Module):
    def __init__(self, vocab_size=None, device='cpu'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        self.fc = nn.Linear(768, vocab_size)

        self.device = device

    def forward(self, x):
        '''

        Parameters
        ----------
        x : Input tensor of form (batch_size, sentence_length)

        Returns
        -------
        Returns the output of fully connected layer
        '''
        x = x.to(self.device)

            # print("->bert.train()")
        self.bert.train()
        encoded_layers, _ = self.bert(x)
        enc = encoded_layers[-1]
        logits = self.fc(enc)
        
        return logits