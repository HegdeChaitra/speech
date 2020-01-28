import os
import sys
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.evaluation import *
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from bert_model import Net
from utils.evaluation import create_biluo_tag_from_ner_types
from spacy.gold import iob_to_biluo
from pytorch_pretrained_bert import BertTokenizer

parser = argparse.ArgumentParser()


parser.add_argument("--trainset", default="./data/raw/CoNLL2003/train.txt", type=str,
                    help="Path to dataset. default=./data/raw/CoNLL2003/train.txt")

parser.add_argument("--validset", default="./data/raw/CoNLL2003/valid.txt", type=str,
                    help="Path to dataset. default=./data/raw/CoNLL2003/valid.txt")

parser.add_argument("--save_path", default="./saved_models/", type=str,
                    help="Path to directory where model should be saved. default='./saved_models/'")

parser.add_argument("--save_name", type=str, required = False, default = "model1.m",
                    help="name of the model to save. deault=model1.m")

parser.add_argument("--load_trained_model", type=str, required=False, default = None,
                    help="complete path of the already traned model to continue training or evaluating. default = None")

parser.add_argument("--do_train",
                    action='store_true',
                    help="Whether to run training. default=False")

parser.add_argument("--do_eval",
                    action='store_true',
                    help="Whether to run eval on the dev set. default=False")

parser.add_argument("--batch_size",
                    default=100,
                    type=int,
                    help="Total batch size. default=100")

parser.add_argument("--n_epochs",
                    default=5,
                    type=int,
                    help="Number of epochs. default=5")

parser.add_argument("--lr",
                    default=5e-5,
                    type=float,
                    help="Learning rate. default=5e-5")




class NerDataset(data.Dataset):
    def __init__(self, fpath, tokenizer):
        """
        fpath: [train|valid|test].txt
        """
        ner_types = ['ORG', 'PER', 'LOC', 'MISC']
        self.tokenizer = tokenizer
        self.VOCAB = ['<PAD>','O']+create_biluo_tag_from_ner_types(ner_types)[:-1]
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.VOCAB)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.VOCAB)}


        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents, tags_li = [], [] # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + iob_to_biluo(tags) + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        return x, is_heads, y, seqlen
    
    def get_vocab(self):
        return self.VOCAB, self.idx2tag, self.tag2idx

def pad(batch):
    '''
    Pads the batch to the maximum length within that batch
    Parameters
    ----------
    batch : non-padded batch

    Returns
    -------
    returns padded batch
    '''
    f = lambda x: [sample[x] for sample in batch]
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(0, maxlen)
    y = f(2, maxlen)
    is_heads = f(1, maxlen)


    f = torch.LongTensor

    return f(x), f(is_heads), f(y), seqlens

def train(model, iterator, optimizer, criterion):
    '''
    Function to train the model for one epoch

    Parameters
    ----------
    model : instance of type Net()
    iterator : train iterator
    optimizer : optimizer. Eg: Adam, SGD
    criterion : Loss instance

    Returns
    -------
    model : Model trained for one more epoch
    '''
    model.train()
    for i, batch in enumerate(iterator):
        x, is_heads, y, seqlens = batch
        optimizer.zero_grad()
        logits = model(x) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1).to(device).long()  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i%20==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

    return model
            
def eval(model, iterator):
    '''
    Function to do token level evaluation

    Parameters
    ----------
    model : Instance of class Net
    iterator : Data iterator

    Returns
    -------
    Returns Precision, Recall and F1 score
    '''
    model.eval()
    Is_heads, Y, Y_hat = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, is_heads, y, seqlens = batch

            logits = model(x)  # y_hat: (N, T)
            y_hat = logits.argmax(-1)
            
            Is_heads.extend(is_heads)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    y_temp = []
    y_hat_temp = []
    
    for y, is_heads, y_hat in zip(Y, Is_heads, Y_hat):
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        y_hat_temp+=y_hat[1:-1]
        ys = [hat for head, hat in zip(is_heads, y) if head==1]
        y_temp+= ys[1:-1]
    y_true = np.array(y_temp)
    y_pred = np.array(y_hat_temp)
    
    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.2f"%f1)
    return precision, recall, f1

def count_correct(y, pred):
    '''
    Function to count number of correct prediction for each of entity type

    Parameters
    ----------
    y : Ground truth in the form of list of tuples of type (start_idx, end_idx, entity_type)
    pred : Prediction in the form of list of tuples of type (start_idx, end_idx, entity_type)

    Returns
    -------
    count : Dictionary containing number of correct predictions for each of entity type
    '''
    count = {'overall':0, 'PER':0, 'ORG':0, 'MISC':0, 'LOC':0}
    y_dict = {}
    pred_dict = {}
    for ent in y:
        y_dict[ent[0]] = (ent[1],ent[2])
    for ent in pred:
        pred_dict[ent[0]] = (ent[1],ent[2])
        
    for key in y_dict:
        if key in pred_dict:
            if y_dict[key]==pred_dict[key]:
                count['overall'] +=1
                count[y_dict[key][1]]+=1
                
    return count

def count_all_ent(entity):
    '''
    Function to count number of occurences of each of entity type

    Parameters
    ----------
    entity : list of entities

    Returns
    -------
    count : Dictionary containing count of each of entity types
    '''
    count = {'overall':0, 'PER':0, 'ORG':0, 'MISC':0, 'LOC':0}
    for ent in entity:
        count['overall']+=1
        count[ent[2]]+=1
        
    return count

def calculate_f1(total_correct, total_proposed, total_act):
    '''
    Function to calculate f1 score

    Parameters
    ----------
    total_correct : Total entities that are predicted correct
    total_proposed : Total entities that were predicted as some entity of interest
    total_act : Total entities that were actually the entities of interest

    Returns
    -------
    Precision, Recall and F1 score

    '''
    try:
        precision = total_correct / total_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = total_correct / total_act
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    return precision, recall, f1

def generate_entities(model, iterator, VOCAB):
    '''
    Function to compute entity level F1 score

    Parameters
    ----------
    model : Trained model of instance Net
    iterator : Data iterator
    VOCAB : List of target labels

    Returns
    -------
    Prints entity wise precision, recall and F1 score

    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ner_types = ['ORG', 'PER', 'LOC', 'MISC']
    valid_transition_matrix = create_valid_transition_matrix(ner_types)
    total_proposed = {'overall':0, 'PER':0, 'ORG':0, 'MISC':0, 'LOC':0}
    total_act = {'overall':0, 'PER':0, 'ORG':0, 'MISC':0, 'LOC':0}
    total_correct = {'overall':0, 'PER':0, 'ORG':0, 'MISC':0, 'LOC':0}
    
    for data in iterator:
        x, mask, y, lens = data
        
        with torch.no_grad():
            out = model(x.to(device).long()).cpu().numpy()
            
        out = np.argmax(out, axis = 2)
        
        for i in range(out.shape[0]):
            ex_y = y[i]
            
            ex_pred = out[i]
            ex_mask = mask[i]
            ex_y = [hat for head, hat in zip(ex_mask, ex_y) if head == 1][1:-1]
            ex_pred = [hat for head, hat in zip(ex_mask, ex_pred) if head==1][1:-1]
            
            ex_y_i = np.array(VOCAB)[list(np.array(ex_y).astype('int'))]
            ex_pred_i = np.array(VOCAB)[list(np.array(ex_pred).astype('int'))]
                        
            entity_y = biluo2entity(ex_y_i, valid_transition_matrix)
            entity_pred = biluo2entity(ex_pred_i, valid_transition_matrix) 
            
            temp_act = count_all_ent(entity_y)
            for key in temp_act:
                total_act[key]+=temp_act[key]
            
            temp_proposed = count_all_ent(entity_pred)
            for key in temp_proposed:
                total_proposed[key]+=temp_proposed[key]
            
            temp_correct = count_correct(entity_y,entity_pred)
            for key in temp_correct:
                total_correct[key]+=temp_correct[key]
    
    print("\t    Precision, \t\t Recall, \t\t F1")
    for key in total_act:
        print(key," : ", calculate_f1(total_correct[key],total_proposed[key],total_act[key]))


if __name__=='__main__':
    
    args = parser.parse_args()
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
        
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = NerDataset(args.trainset, tokenizer)
    eval_dataset = NerDataset(args.validset, tokenizer)
    test_dataset = NerDataset("../ner_copy_2/data/raw/CoNLL2003/test.txt", tokenizer)
    VOCAB, idx2tag, tag2idx = train_dataset.get_vocab()

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn = pad)
    
    test_iter = data.DataLoader(dataset = test_dataset,
                               batch_size = args.batch_size,
                               shuffle = False,
                               num_workers = 4,
                               collate_fn = pad)

    if args.load_trained_model!=None:
        model = torch.load(args.load_trained_model).module.to(device)
    else:
        model = Net(len(VOCAB), device).to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_f1 = 0
    best_param = None
    
    if args.do_train:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
            print("Save directory created")
        for epoch in range(1, args.n_epochs+1):
            model = train(model, train_iter, optimizer, criterion)
    
            print(f"=========eval at epoch={epoch}=========")
            precision, recall, f1 = eval(model, eval_iter)
        
            if best_f1<f1:
                best_f1 = f1
                torch.save(model, args.save_path+args.save_name)
                best_param = model.state_dict()
            
        try:
            model.load_state_dict(best_param)
        except:
            pass
    print()
    if args.do_eval:
        print(f"========= Entity level eval ==========")  
        print("Validation set")
        generate_entities(model,eval_iter, VOCAB)
        print("Test set")
        generate_entities(model,test_iter, VOCAB)
    
        
        