import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from torch.autograd import Variable
import pickle
import argparse

from def_attn_model_word_eff import Attention
from define_word_training import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



parser = argparse.ArgumentParser()

parser.add_argument('--max_len', type=int, default=500,
                    help='Maximum length of the sentence. default=500')

parser.add_argument('--bs', type=int, default=32,
                    help='Batch size. default=32')

parser.add_argument('--n_vocab', type=int, default=50000,
                    help='Vocabulary size. default=50000')

parser.add_argument('--lr', type=float, default=5e-4,
                    help='Learning rate. default=5e-4')

parser.add_argument('--n_epoch', type=int, default=30,
                    help='Number of epochs. default=30')

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training. default=False")

parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set. default=False")

parser.add_argument('--source_path', type=str, default='./processed_data/',
                    help='Path to dataset. default=./processed_data/')

parser.add_argument('--train_data', type=str, default='idized_hollow_train.csv',
                    help='Name of train dataset. default=idized_hollow_train.csv')

parser.add_argument('--val_data', type=str, default='idized_hollow_val.csv',
                    help='Name of val dataset. default=idized_hollow_val.csv')

parser.add_argument('--token_id', type=str, default='token_and_id.pk',
                    help='Name of token&id pickle file. default=token_and_id.pk')

parser.add_argument('--emb_dim', type=int, default=200,
                    help='embedding dimension. default=200')

parser.add_argument('--save_path', type=str, default='./saved_models/',
                    help='Directory where models are to be saved')

parser.add_argument('--model_load', type=str, default=None,
                    help='Complete path to the Model to load for continued training/ evaluation. default=None')

parser.add_argument('--model_save', type=str, default = "model_sample.m",
                    help="Name of the model to be saved. default = 'model_sample.m'")


args = parser.parse_args()

MAX_SENTENCE_LENGTH = args.max_len
MAX_VOCAB = args.n_vocab
PAD_IDX = 0
UNK_IDX = 1

class HollowDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = csv_file

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
#        token_idx = self.data_frame.iloc[idx]["idized"]
        token_idx = np.array(self.data_frame.iloc[idx]["idized"][1:-1].split(',')).astype(int)
        label = self.data_frame.iloc[idx]['topic_id'].astype(int)
        return [token_idx, len(token_idx), label]

def pad_fun(batch):
    data_list = []
    label_list = []
    length_list = []
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    for datum in batch:
        if datum[1]>MAX_SENTENCE_LENGTH:
            padded_vec = np.array(datum[0][:MAX_SENTENCE_LENGTH])
        else:
            padded_vec = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_SENTENCE_LENGTH - datum[1])),
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.from_numpy(np.array(length_list)), torch.from_numpy(np.array(label_list))]


if __name__=='__main__':
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    print("begin")
    
    df_train = pd.read_csv(args.source_path+args.train_data)
    df_val = pd.read_csv(args.source_path+args.val_data)
    
#     df_train = pd.read_csv("./processed_data/idized_hollow_train.csv")
#     df_val = pd.read_csv("./processed_data/idized_hollow_val.csv")

    with open(args.source_path+args.token_id,'rb') as f:
        token_2id = pickle.load(f)
        id2_token = pickle.load(f)
        all_toks = pickle.load(f)
        
        
    BATCH_SIZE = args.bs
    
    if args.do_train:
        train_dataset = HollowDataset(df_train)
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           collate_fn = pad_fun,
                                           shuffle = True)

    val_dataset = HollowDataset(df_val)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                           batch_size = BATCH_SIZE,
                                           collate_fn = pad_fun,
                                           shuffle = True)
    
    if args.do_train:
        num_classes = df_train['topic_id'].unique().shape[0]
        dataloaders = [train_loader, val_loader]
    else:
        num_classes = 5842
    
    print("number of classes",num_classes)

    if args.model_load:
        print("model loaded")
        model = torch.load(args.model_load)
        model_p = model.state_dict()
        model = Attention(args.emb_dim,num_classes,len(token_2id))
        model.load_state_dict(model_p)
    else:
        print("model defined")
        model = Attention(args.emb_dim,num_classes,len(token_2id))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    print("training begun")
    print(model)
    
    if args.do_train:
        loss_hists,acc_hist = train_model(optimizer, criterion, model, dataloaders, device,
                                      args.save_path+args.model_save, args.save_path, args.n_epoch)
        print("Model saved at ", args.save_path+args.model_save")

    if args.do_eval:
        print("Model Evaluation")
        net_evaluation(model, val_loader, device)