import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import time
from pytorch_pretrained_bert.tokenization import BertTokenizer

import torch.nn as nn
from torch.optim import Adam
from model import Net

parser = argparse.ArgumentParser()

parser.add_argument("--train_dataset", default="../english_classification/hollow_dataset/df_train.csv", type=str,
                    help="Path of train dataset")

parser.add_argument("--val_dataset", default="../english_classification/hollow_dataset/df_val.csv", type=str,
                    help="Path to validation dataset")

parser.add_argument("--output_dir",
                    default="./save_mod/",
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training.")

parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set.")

parser.add_argument("--batch_size",
                    default=20,
                    type=int,
                    help="batch size.")

parser.add_argument("--lr",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument("--n_epochs",
                    default=3,
                    type=int,
                    help="Total number of training epochs to perform.")

parser.add_argument("--save_name",
                   default="bert_sample2",
                   type=str,
                   help ="name of the model to save default='bert_sample2'")

parser.add_argument("--load_model",
                   default = None,
                   type = str,
                   help ="name of the model to load default = None")

args = parser.parse_args()

class HollowDataset(Dataset):
    def __init__(self, csv_file):
        '''

        Parameters
        ----------
        csv_file : Pandas dataframe object with fields=['extracted_content','topic_id'].
        'extracted content' is strings and 'topic_id' is integers
        '''
        self.data_frame = csv_file
        tokenizer.max_len = 5000
        self.data_frame['tokenized'] = self.data_frame['extracted_content'].apply(lambda x: tokenizer.tokenize(x)[:510])
        self.data_frame['idized'] = self.data_frame['tokenized'].apply(lambda x: tokenizer.convert_tokens_to_ids(x))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        max_seq_length = 512
        tokenized = self.data_frame.iloc[idx]['idized']        
        idized = [101]+tokenized+[102]

        label = self.data_frame.iloc[idx]['topic_id'].astype(int)

        segment_ids = [0] * len(idized)
        input_mask = [1] * len(idized)
        padding = [0] * (max_seq_length - len(idized))

        idized += padding
        input_mask += padding
        segment_ids += padding

        return [np.array(idized), np.array(input_mask), np.array(segment_ids), np.array(label)]


def train(model, iterator, optimizer, criterion, device):
    '''
    Function to train the model for one epoch
    Parameters
    ----------
    model : Instance of type Net
    iterator : Instance of type torch.utils.data.DataLoader
    optimizer : Adam optimizer instance
    criterion : Instance of type CrossEntropyLoss
    device : 'cpu' or 'gpu'

    Returns
    -------
    Returns the model after one epoch of training
    '''
    model.train()
    epoch_loss = 0
    for data in iterator:
        optimizer.zero_grad()
        x = data[0].to(device)
        y = data[-1].to(device).long()

        m_out = model(x)
        loss = criterion(m_out,y)

        epoch_loss += loss.mean().item()

        loss.backward()
        optimizer.step()
    print("Train Loss = ", epoch_loss)
    return model


def eval(model, iterator, criterion, device='cpu'):
    '''
    Function to calculate accuracy of the trained model
    Parameters
    ----------
    model : Instance of type Net
    iterator : Instance of type torch.utils.data.DataLoader
    criterion : CrossEntropyLoss object
    device : 'cpu' or 'gpu'

    Returns
    -------
    accuracy : Returns accuracy of the model
    '''
    model.eval()

    epoch_loss = 0
    total = 0
    correct = 0
    for data in iterator:
        x = data[0].to(device)
        y = data[-1].to(device).long()

        with torch.no_grad():
            logits = model(x)

        loss = criterion(logits, y)
        epoch_loss+=loss.mean().item()

        _, top1_predicted = torch.max(logits, dim=1)
        correct += int((top1_predicted == y).sum())
        total += x.size(0)

    accuracy = correct/total
    print("Eval Loss = ", epoch_loss)
    print("Eval Accuracy = ", accuracy)
    return accuracy


def final_eval(model, data_loader, device='cpu'):
    '''
    Function to evaluate the trained model
    Parameters
    ----------
    model : Trained model. Instance of type Net()
    device : 'cpu' or 'gpu'
    data_loader : Instance of type torch.utils.data.DataLoader

    Returns : Prints Top1, Top5 and Top10 accuracy along with time taken to run the model and number of examples evaluated
    -------
    '''

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0

    total = 0
    model.eval()
    start = time.time()
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, y_batch = batch
        with torch.no_grad():
            output = model(input_ids)

        _, top1_predicted = torch.max(output, dim=1)
        top1_correct += int((top1_predicted == y_batch).sum())

        _, topn_predicted = torch.topk(output, k=5, dim=1, largest=True)
        for col in range(5):
            top5_correct += int((topn_predicted[:, col] == y_batch).sum())

        _, topn_predicted = torch.topk(output, k=10, dim=1, largest=True)
        for col in range(10):
            top10_correct += int((topn_predicted[:, col] == y_batch).sum())

        total += input_ids.size(0)

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    top10_acc = top10_correct / total
    print("-" * 100)
    print("top1_acc = ", round(100 * top1_acc, 2))
    print("top5_acc = ", round(100 * top5_acc, 2))
    print("top10_acc = ", round(100 * top10_acc, 2))
    print("Time taken = ", round(time.time() - start, 2), " seconds")
    print("Number of documents = ", total)


if __name__ == '__main__':
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    print("reading data")
    if args.do_train:
        df_train = pd.read_csv(args.train_dataset)
        num_labels = max(df_train['topic_id'].unique())+1
        print("Number of Classes = ",num_labels)
#         df_train = df_train.sample(n=10)
    df_val = pd.read_csv(args.val_dataset)
#     df_val = df_val.sample(n=15)

    if args.load_model is not None:
        print("Loading the model")
        model = torch.load(args.load_model).to(device)
    else:
        print("Creating the model")
        model = Net(num_labels, device).to(device)
    model = nn.DataParallel(model)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("Creating Dataset objects. Sit back and relax. The tokenization will take some time.")
    if args.do_train:
        train_dataset = HollowDataset(df_train)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True)

    val_dataset = HollowDataset(df_val)
    eval_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=True)

    best_acc = 0
    best_param = None
    
    if args.do_train:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        print("Training")
        for epoch in range(args.n_epochs):
            print("-"*10)
            print("Epoch = ", epoch+1)
            model = train(model, train_dataloader, optimizer, criterion, device)
            accuracy = eval(model, eval_dataloader, criterion, device)

            if accuracy>best_acc:
                best_acc = accuracy
                torch.save(model, args.output_dir+args.save_name)
                best_param = model.state_dict()

        try:
            model.load_state_dict(best_param)
        except:
            pass

    if args.do_eval:
        print()
        print("Final Evaluation")
        final_eval(model, eval_dataloader, device)
