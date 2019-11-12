import os
import pickle
from unidecode import unidecode
import numpy as np
import pandas as pd
from fasttext import FastVector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_decomposition import CCA

import argparse
from args import parse_args

import codecs
import sys
import csv

csv.field_size_limit(sys.maxsize)

def read_fr_dataset(path):
    with codecs.open(path, 'r', encoding='utf-8') as csvfile:
        df = pd.DataFrame()
        csv_reader = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(csv_reader):
            df = df.append(pd.DataFrame([row]))
    df.columns = ['label', 'data']
    df = df[1:]
    return df

def read_dataset(infile):
    data = pd.read_csv(infile, sep='\t', encoding='utf-8')
    data.columns = ['label', 'data']
    return data

def process_dataset(df, lang_dict, tfidfvec=None):
    doc_emb, vectorizor = doc_embedding(df, lang_dict, tfidfvec=tfidfvec)
    return df['label'].values, doc_emb, vectorizor


def read_dictionary(infile, max_len=15000):
    with open(infile, "r") as file:
        lines = file.readlines()
    bi_dict = [tuple(line.rstrip('\n').split(' ')) for line in lines]
    bi_dict = bi_dict[:max_len]
    return bi_dict


def make_training_matrices(source_dictionary, target_dictionary, dictionary):
    source_matrix = []
    target_matrix = []
    for (source, target) in dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])
    print("number of CCA training pairs:", len(source_matrix))
    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)


def cca(src_dict, tgt_dict, bi_dict, dim=250):

    #with open('../data/seed_embedding.dat', 'wb') as f:
    #    pickle.dump(x, f)
    #    pickle.dump(y, f)
    cca_model = CCA(n_components=dim)
    src_mat, tgt_mat = make_training_matrices(src_dict, tgt_dict, bi_dict)
    cca_model.fit(src_mat, tgt_mat)
    return cca_model.transform(src_dict.embed, tgt_dict.embed)

def doc_embedding(df, lang_vec, tfidf=True, tfidfvec=None):
    if tfidf:
        if not tfidfvec:
            tfidfvec = TfidfVectorizer(vocabulary=lang_vec.word2id)
            weights = tfidfvec.fit_transform(df['data'])
        else:
            weights = tfidfvec.transform(df['data'])
            
        doc_emb = weights.dot(lang_vec.embed)
    else:
        doc_emb = np.zeros([len(df), 300])
        doc = df['data'].apply(lambda x: x.lower().split())

        for idx, sent in enumerate(doc):
            for word in sent:
                if word in lang_vec.word2id.keys():
                    doc_emb[idx] += lang_vec[word]
    return doc_emb, tfidfvec

    

if __name__ == '__main__':
    args = parse_args()

    print('loading vectors')
    en_dictionary = FastVector(vector_file=args.en_embedding)
    fr_dictionary = FastVector(vector_file=args.fr_embedding)

    #print('transforming vectors')
    #fr_dictionary.apply_transform('alignment_matrices/fr.txt')

    #print('CCA...')
    #en_fr = read_dictionary(args.embedding_path+'en_fr.txt')
    #en_dictionary.embed, fr_dictionary.embed = cca(en_dictionary, fr_dictionary, en_fr, dim=250)

    print("Hello score:", FastVector.cosine_similarity(en_dictionary["hello"], fr_dictionary["bonjour"]))

    print('processing data')
    en_train_file = args.source_path + 'en_train.tsv'
    en_test_file = args.source_path + 'en_test.tsv'
    fr_train_file = args.source_path + 'fr_train.tsv'
    fr_test_file = args.source_path + 'fr_test.tsv'

    print('english train')
    en_train_df = read_dataset(en_train_file)
    en_train_y, en_train_x, en_vectorizor = process_dataset(en_train_df, en_dictionary, None)

    n_classes = len(set(en_train_y))
    label_encoder = dict(zip(list(set(en_train_y)), np.arange(n_classes)))
    en_train_y = np.array([label_encoder[i] for i in en_train_y])

    print('english test')
    en_test_df = read_dataset(en_test_file)
    en_test_y, en_test_x, _ = process_dataset(en_test_df, en_dictionary, en_vectorizor)
    en_test_y = np.array([label_encoder[i] for i in en_test_y])

    print('french train')
    fr_train_df = read_fr_dataset(fr_train_file)
    fr_train_y, fr_train_x, fr_vectorizor = process_dataset(fr_train_df, fr_dictionary, None)
    fr_train_y = np.array([label_encoder[i] for i in fr_train_y])


    print('french test')
    fr_test_df = read_fr_dataset(fr_test_file)
    fr_test_y, fr_test_x, _ = process_dataset(fr_test_df, fr_dictionary, fr_vectorizor)
    fr_test_y = np.array([label_encoder[i] for i in fr_test_y])

    try:
        os.makedirs(args.preprocessed_path)
    except OSError:
        pass


    print('writing to pickle')
    train_x = np.concatenate((en_train_x, fr_train_x), axis=0)
    train_y = np.concatenate((en_train_y, fr_train_y), axis=0)

    with open(args.train_path, "wb") as f:
        pickle.dump(train_y, f)
        pickle.dump(train_x, f)
    with open(args.en_test_path, "wb") as f:
        pickle.dump(en_test_y, f)
        pickle.dump(en_test_x, f)
    with open(args.fr_test_path, "wb") as f:
        pickle.dump(fr_test_y, f)
        pickle.dump(fr_test_x, f)
