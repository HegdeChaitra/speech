import torch
from pytorch_pretrained_bert import BertModel
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, vocab_size=None, device='cpu'):
        '''

        Parameters
        ----------
        vocab_size : Number of target classes.
        device : 'cpu' or 'gpu'
        '''
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(768, vocab_size)
        self.dp = nn.Dropout(p=0.2)

        self.device = device

    def forward(self, x):
        '''

        Parameters
        ----------
        x : Input torch tensor of form (batch_size, sentence_length)

        Returns
        -------
        Returns the output of fully connected layer for '[CLS]' token
        '''
        x = x.to(self.device)
        self.bert.train()
        encoded_layers, _ = self.bert(x)
        enc = encoded_layers[-1]
        logits = self.fc(self.dp(enc))
        return logits[:, 0]
