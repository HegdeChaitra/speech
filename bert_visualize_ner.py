from spacy.gold import offsets_from_biluo_tags
from spacy.tokens import Span
import spacy
from spacy import displacy
from spacy.tokenizer import Tokenizer
import argparse

import os
import sys
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.evaluation import *
import numpy as np
from IPython.core.display import display, HTML


parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="./saved_models/best_bert_ner2.model", type=str,
                    help="Complete path of the trained model")

parser.add_argument("--viz_save_path",
                   default = "./saved_viz/",
                   type=str,
                   help = "name of the folder where visualization has to be saved. default = './saved_viz/'")

parser.add_argument("--html_file",
                    default="html_out1.html",
                    type=str,
                    help="name of output file. default=html_out1.html")

parser.add_argument("--input_file",
                   default = "./saved_viz/input1.txt",
                   type=str,
                   help="Path where input is located. default = './saved_viz/input1.txt'")

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab)


def get_pred_val(model, tokenizer, ner_types, all_text):
    
    idx2tag = ['<PAD>','O']+create_biluo_tag_from_ner_types(ner_types)[:-1]

    all_x = []
    all_val_pred = []
    nlp = spacy.load('en_core_web_sm')
#     text_spacy = nlp(all_text)
#     all_sents = text_spacy.sents
#     all_sents = all_text.split('.')
    all_sents = [all_text]
    for texts in all_sents:
    
        text = ['[CLS]']+str(texts).split()+['[SEP]']
        x= []
        all_toks = []
        is_heads = []
        for w in text:
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]

            xx = tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0]*(len(tokens) - 1)
        
            if len(is_head)>0:
                all_toks+=[w]
            
            x.extend(xx)
            is_heads.extend(is_head)

        inp = torch.from_numpy(np.array(x)).view(1,-1).cuda().long()
        a = model(inp)
        out = np.argmax(a.detach().cpu().numpy(), axis = 2)[0]
        all_val_pred.append([idx2tag[tok] for i, tok in enumerate(out[1:-1]) if is_heads[1:-1][i]==1])
        all_x.append(" ".join(all_toks[1:-1]))
    return all_x, all_val_pred

def visualize(giv_text_all, pred_ent_all, ner_types):
    valid_transition_matrix = create_valid_transition_matrix(ner_types)
    nlp = spacy.blank('en')
    nlp.tokenizer = custom_tokenizer(nlp)
    all_docs = []
    i=0
    for giv_text, pred_ent in zip(giv_text_all, pred_ent_all):
        all_entities = biluo2entity(pred_ent, valid_transition_matrix)
        doc = nlp(giv_text)
        for ent in all_entities:
            if ent[-1] not in ["ORG","LOC","PER",'MISC']:
                continue
                
            if ent[0]>ent[1]:
                continue
            if ent[-1]=='MISC':
                ent_typei = doc.vocab.strings.add('MISC')
            if ent[-1]=='PER':
                ent_typei = doc.vocab.strings['PERSON']
            else:   
                ent_typei = nlp.vocab.strings[ent[-1]]
            new_ent = Span(doc, ent[0], ent[1],label=ent_typei)
            doc.ents = list(doc.ents) + [new_ent]
        if len(doc.ents)==0:
            if len(doc)==0:
                continue
            ent_typei = doc.vocab.strings.add('NONE')
            new_ent = Span(doc, 0, 0,label=ent_typei)
            doc.ents = list(doc.ents) + [new_ent]
        if i==0:
            doc.user_data["title"] ="\n New Example"
        i+=1
            
        all_docs.append(doc)
    return all_docs

if __name__=="__main__":
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    if not os.path.exists(args.viz_save_path):
        os.makedirs(args.viz_save_path)
        print("saving directory created")
        
        
    
    ner_types=['ORG', 'PER', 'LOC', 'MISC']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    model = torch.load(args.model_name)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model.module)

    entries = open(args.input_file, 'r').read().strip().split("\n\n")
    full_pred= []
    for text in entries:
        giv_text_all, pred_ent_all= get_pred_val(model, tokenizer, ner_types, text)
        all_out_doc = visualize(giv_text_all, pred_ent_all, ner_types)
        full_pred+=all_out_doc
    
    Html_file= open(args.viz_save_path+args.html_file,"w")
    Html_file.write(spacy.displacy.render(full_pred, style="ent", page="true",minify=True))
    Html_file.close()

