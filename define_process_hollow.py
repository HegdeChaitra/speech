import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
from torch.autograd import Variable
import string
import argparse

punctuations = string.punctuation

stop_words = ["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","to","from","up","down","in","on","off","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","other","some","such","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]

parser = argparse.ArgumentParser()

parser.add_argument('--source_path', type=str, default="./processed_data/",
                    help='path for the dataset. default=./processed_data/')

parser.add_argument('--dest_path', type=str, default='./processed_data/',
                    help='path where the processed dataset has to be stored, default=./processed_data') 

parser.add_argument('--df_name', type=str, default="df_train.csv",
                    help='Name of the train dataset to process. default=df_train.csv')

parser.add_argument('--df_val', type=str, default="df_val.csv",
                    help='Name of the validation dataset to process. default=df_val.csv')

parser.add_argument('--df_save_name', type=str, default="idized_hollow_train.csv",
                    help='Name of the df to be saved. default=idized_hollow_train.csv')

parser.add_argument('--val_save_name', type=str, default="idized_hollow_val.csv",
                    help='Name of the validation df to be saved. default=idized_hollow_val.csv')

parser.add_argument('--n_vocab', type=int, default=50000,
                    help='Size of the vocabulary. default=50000')

parser.add_argument('--load_tok_id', type=str, default="None",
                    help='Token and ids of vocabulary. default=token_and_id.pk')

args = parser.parse_args()

def fun_apply(x):
    return [tok.lower() for tok in x.split(" ") if(tok not in punctuations and tok not in stop_words)]

def split(df):
    df['tokenized'] = df['extracted_content'].apply(fun_apply)
    return df

def build_vocab(all_tokens):
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(MAX_VOCAB))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token


def token2index_dataset(df,token2id):
    indices_data = []
    for tokens in df['tokenized']:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    df['idized'] = indices_data
    return df

if __name__=='__main__':
    
    MAX_VOCAB = args.n_vocab
    PAD_IDX = 0
    UNK_IDX = 1
    print("reading train data")
    df_train = pd.read_csv(args.source_path+args.df_name)
    df_train = split(df_train)
    
    if args.load_tok_id=="None":
        all_toks = []
        for i,row in df_train.iterrows():
            all_toks+=row['tokenized']
        
    
        token_2id, id2_token = build_vocab(all_toks)
        
        
        with open(args.dest_path+"token_and_id.pk",'wb') as f:
            pickle.dump(token_2id,f)
            pickle.dump(id2_token,f)
            pickle.dump(all_toks,f)

    else:
        with open(args.load_tok_id,'rb') as f:
            token_2id = pickle.load(f)
            id2_token = pickle.load(f)
            all_toks = pickle.load(f)
        
    df_train = token2index_dataset(df_train,token_2id)
    print("reading val data")
    if args.df_val!="None":
        df_val = pd.read_csv(args.source_path+args.df_val)
        df_val = split(df_val)
    
        df_val = token2index_dataset(df_val,token_2id)
        df_val.to_csv(args.dest_path+args.val_save_name, index=None)
        
    df_train.to_csv(args.dest_path+args.df_save_name,index=None)
    
    
    
    