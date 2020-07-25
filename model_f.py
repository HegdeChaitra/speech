import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.parallel
from collections import OrderedDict
# for module definition
from torch.nn.parameter import Parameter
import scipy


class MLP(nn.Module):
    def __init__(self, n_mlp_layer, n_hidden, n_class, n_emb, dropout, device='cpu', ngpu=1):
        super(MLP, self).__init__()
        self.n_mlp_layer = n_mlp_layer
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.n_emb = n_emb
        self.device = device
        self.dropout = dropout
        self.ngpu = ngpu
        self.linear1 = nn.Linear(self.n_emb, self.n_hidden)
        self.loss = nn.NLLLoss()
        if n_mlp_layer == 1:
            self.main = nn.Sequential(
                #nn.BatchNorm1d(self.n_emb),
                self.linear1,
                nn.Dropout(self.dropout),
                nn.Tanh(),
                nn.Linear(self.n_hidden, self.n_class),
                nn.LogSoftmax(dim=1))
        else:
            assert n_mlp_layer > 1, 'Number of mlp layers must be at least 1'
            self.mlp_layers = [('Batch normalization', nn.BatchNorm1d(self.n_emb)),
                               ('MLP0', self.linear1),
                               ('Activation0', nn.Tanh())]
            for i in range(1, n_mlp_layer, 1):
                self.mlp_layers.append(('MLP%d' % i, nn.Linear(self.n_hidden, self.n_hidden)))
                self.mlp_layers.append(('Activation%d' % i, nn.Tanh()))
            self.mlp_layers.append(('Dropout', nn.Dropout(self.dropout)))
            self.mlp_layers.append(('Decoder', nn.Linear(self.n_hidden, self.n_class)))
            self.mlp_layers.append(('LogSoftmax', nn.LogSoftmax()))
            self.main = nn.Sequential(OrderedDict(self.mlp_layers))

    def forward(self, x):
        if self.device == 'gpu':
            x = x.cuda()
        if self.device == 'gpu' and self.ngpu > 1:
            emb = nn.parallel.data_parallel(x, range(self.ngpu))
            logits = nn.parallel.data_parallel(self.main, emb, range(self.ngpu))
        else:
            logits = self.main(x)
        return logits
