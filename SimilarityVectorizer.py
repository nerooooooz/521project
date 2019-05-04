#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:33:09 2019

@author: Cassie
"""
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class SimilarityVectorizer:
    """Creates a vector of similarities for pairs of sentences"""

    def __init__(self, tfidf_vectorizer, word2vec):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.word2vec = word2vec

    def tfidf_sim(self, t1, t2):
        """Returns a float of cosine similarity between tfidf vectors for two sentences.
        Uses preprocessing including stemming."""
        t1=self.preprocess_text(t1,stem=True)
        t2=self.preprocess_text(t2,stem=True)
        tfidf=self.tfidf_vectorizer.transform([t1,t2])
        m=tfidf.todense()
        m1=np.array(m[0])
        m2=np.array(m[1])
        
        return cosine_similarity(m1,m2)[0,0]

    def w2v_sim(self, t1, t2):
        """Returns a float of cosine similarity between w2v vectors for two sentences.
        w2v vectors are the mean of any in-vocabulary words in the sentence, after lowercasing.
        Cosine similarity is 0 if either sentence is completely out of vocabulary. """
        
        t1_vector = self.w2v_sentence(t1, self.word2vec)
        if t1_vector is None:
            return 0
        t1_vector = t1_vector.reshape((1, -1)) # shape for cosine similarity
        t2_vector = self.w2v_sentence(t2, self.word2vec)
        if t2_vector is None:
            return 0
        t2_vector = t2_vector.reshape((1, -1))
        pair_similarity = cosine_similarity(t1_vector, t2_vector)[0, 0]
        return pair_similarity

    def load_X(self, sent_pairs):
        """Create a matrix where every row is a pair of sentences and every column in a feature.
        """
        X = np.zeros((len(sent_pairs), 2))
        tfidf=[]
        w2v=[]
        for pair in sent_pairs:
            t1 = pair[0]
            t2 = pair[1]
            tfidf.append(self.tfidf_sim(t1,t2))
            w2v.append(self.w2v_sim(t1,t2))
            
        all_list=[tfidf,w2v]
        for i in range(2):
            X[:,i]=np.asarray(all_list[i])
        
        return X
    
    
    def preprocess_text(self,text, stem=False):
        """Preprocess one sentence: tokenizes, lowercases, (optionally) applies the Porter stemmer,
         removes punctuation tokens and stopwords.
         Returns a string of tokens joined by whitespace."""
        toks = word_tokenize(text)
        if stem==True:
            ps = PorterStemmer()
            toks=[ps.stem(tok) for tok in toks]
            
        remove_tokens = set(stopwords.words("english") + list(string.punctuation))
        toks = [tok for tok in toks if tok not in remove_tokens]
        
        toks = ' '.join(toks)
        return toks

    def w2v_sentence(self,sent, word2vec):
        """Creates a sentence representation by taking the mean of all in-vocabulary word vectors.
        Returns None if no words are in vocabulary."""
        toks = self.preprocess_text(sent).split()
        veclist = [word2vec[tok] for tok in toks if tok in word2vec]
        if len(veclist) == 0:
            return None
        #vec_mat = np.vstack(veclist)
        mean_vec = np.mean(veclist, axis=0)
        return mean_vec