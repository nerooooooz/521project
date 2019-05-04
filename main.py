#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:16:29 2019

@author: Cassie
"""
import argparse
import numpy as np
from nltk.translate import nist_score, bleu_score
from nltk.metrics.distance import edit_distance
from nltk import word_tokenize
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression

from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from SimilarityVectorizer import SimilarityVectorizer
from WordnetSimilarity import WordnetSimilarity

from sklearn import metrics
import matplotlib.pyplot as plt


def load_data(msr_data):
    """Read a dataset"""
    texts = []
    labels = []
    with open(msr_data,'r') as file:
        for line in file:
            fields = line.strip().split("\t")
            pair=[fields[3].lower(),fields[4].lower()]
            texts.append(pair)
            labels.append(fields[0])

    return texts, labels


def string_sim(sent_pairs):
    """Create a matrix where every row is a pair of sentences and every column in a feature.
    Feature (column) order is not important to the algorithm."""

    features = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Levenshtein distance"]
    nist_list=[]
    bleu_list=[]
    wer_list=[]
    lcs_list=[]
    dist_list=[]
    for pair in sent_pairs:
        t1 = pair[0]
        t2 = pair[1]
        t1_token = word_tokenize(pair[0])
        t2_token = word_tokenize(pair[1])
        
        # NIST
        try:
            nist1=nist_score.sentence_nist([t2_token,],t1_token)
            nist2=nist_score.sentence_nist([t1_token,],t2_token)
            nist=nist1+nist2
        except ZeroDivisionError:
            nist=0
        nist_list.append(nist)
        
        # BLEU
        bleu1=bleu_score.sentence_bleu([t1_token,],t2_token)
        bleu2=bleu_score.sentence_bleu([t2_token,],t1_token)
        bleu_list.append(bleu1+bleu2)
        
        # Longgest common substring
        s=SequenceMatcher(None,t1,t2)
        lcs=s.find_longest_match(0,len(t1),0,len(t2))
        lcs_list.append(lcs[2])
        
        # Edit distance
        dist = edit_distance(t1, t2)
        dist_list.append(dist)
        
        # Word error rate
        dist_wer = edit_distance(t1_token, t2_token)
        wer=dist_wer/len(t1_token) + dist_wer/len(t2_token)
        wer_list.append(wer)
    
    all_list=[nist_list,bleu_list,wer_list,lcs_list,dist_list]
    X = np.zeros((len(sent_pairs), len(features)))
    for i in range(len(all_list)):
        X[:,i]=np.asarray(all_list[i])
    
    return X


def main(train_file, test_file,w2v_file):
    
    # loading train
    train_texts, train_y = load_data(train_file)
    # loading test
    test_texts, test_y = load_data(test_file)
    
    # load word2vec
    w2v_vecs = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    # prepare tfidf vectorizer using train
    tfidf_vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word",
      token_pattern="\S+", use_idf=True, min_df=10)
    tfidf_vectorizer.fit([i[0]+i[1] for i in train_texts])

    # create a SimilarityVectorizer object
    sim_vectorizer = SimilarityVectorizer(tfidf_vectorizer, w2v_vecs)
    
    # create a WordnetSimilarity object
    sim_wn = WordnetSimilarity()
    
    # load String similarity train and test
    train_str_X = string_sim(train_texts)
    test_str_X = string_sim(test_texts)
    
    # load Vector based train and test
    train_vec_X = sim_vectorizer.load_X(train_texts)
    test_vec_X = sim_vectorizer.load_X(test_texts)
    
    # load Wordnet train and test
    train_wn_X = sim_wn.load_X(train_texts)
    test_wn_X = sim_wn.load_X(test_texts)
    
    
    print(f"Found {len(train_texts)} training pairs")
    print(f"Found {len(test_texts)} dev pairs")
    
    
    print("Fitting and evaluating model")
    logreg=LogisticRegression()
    print("String Similarity model Accuracy:")
    model = logreg.fit(train_str_X, train_y)
    score = model.score(test_str_X,test_y)
#    pred_str = model.predict_proba(test_str_X)[:, 1]
    print(f"Accuracy is {score}")
    
    print("=======================================================")
    
    print("Vector-based Similarity model Accuracy:")
    model = logreg.fit(train_vec_X, train_y)
    score = model.score(test_vec_X,test_y)
#    pred_vec = model.predict_proba(test_vec_X)[:, 1]
    print(f"Accuracy is {score}")
    
    print("=======================================================")
    
    print("Wordnet-based Similarity model Accuracy:")
    model = logreg.fit(train_wn_X, train_y)
    score = model.score(test_wn_X,test_y)
#    pred_wn = model.predict_proba(test_wn_X)[:, 1]
    print(f"Accuracy is {score}")
    
    print("=======================================================")
    
    print("String Similarity + Vector-based Similarity model Accuracy:")
    train_vec_str=np.concatenate((train_str_X,train_vec_X),axis = 1)
    test_vec_str=np.concatenate((test_str_X,test_vec_X),axis = 1)
    model = logreg.fit(train_vec_str, train_y)
    score = model.score(test_vec_str,test_y)
    print(f"Accuracy is {score}")
    
    print("=======================================================")
    
    print("String Similarity + Wordnet-based Similarity model Accuracy:")
    train_str_wn=np.concatenate((train_str_X,train_wn_X),axis = 1)
    test_str_wn=np.concatenate((test_str_X,test_wn_X),axis = 1)
    model = logreg.fit(train_str_wn, train_y)
    score = model.score(test_str_wn,test_y)
    print(f"Accuracy is {score}")
    
    print("=======================================================")
    
    print("Wordnet-based Similarity + Vector-based Similarity model Accuracy:")
    train_vec_wn=np.concatenate((train_wn_X,train_vec_X),axis = 1)
    test_vec_wn=np.concatenate((test_wn_X,test_vec_X),axis = 1)
    model = logreg.fit(train_vec_wn, train_y)
    score = model.score(test_vec_wn,test_y)
    print(f"Accuracy is {score}")
    
    print("=======================================================")
    
    print("String + Vector-based + Wordnet-based Similarity model Accuracy:")
    train_all=np.concatenate((train_str_X,train_wn_X,train_vec_X),axis = 1)
    test_all=np.concatenate((test_str_X,test_wn_X,test_vec_X),axis = 1)
    model = logreg.fit(train_all, train_y)
    score = model.score(test_all,test_y)
    print(f"Accuracy is {score}")
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="msr_paraphrase_test.txt",
                        help="dev file")
    parser.add_argument("--train_file", type=str, default="msr_paraphrase_train.txt",
                        help="train file")
    parser.add_argument("--w2v_file", type=str, default="50K_GoogleNews_vecs.txt",
                        help="file with word2vec vectors as text")
    args = parser.parse_args()

    main(args.train_file, args.test_file,args.w2v_file)