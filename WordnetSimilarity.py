#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:04:49 2019

@author: Cassie
"""

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import numpy as np

class WordnetSimilarity:
    
    def __init__(self):
        self.brown_ic = wordnet_ic.ic('ic-brown.dat')
  
    def penn_to_wn(self,tag):
        """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
        if tag.startswith('N'):
            return 'n'
     
        if tag.startswith('V'):
            return 'v'
     
        if tag.startswith('J'):
            return 'a'
     
        if tag.startswith('R'):
            return 'r'
     
        return None
 
    def tagged_to_synset(self,word, tag):
        wn_tag = self.penn_to_wn(tag)
        if wn_tag is None:
            return None
     
        try:
            return wn.synsets(word, wn_tag)[0]
        except:
            return None
 
    def sentence_similarity(self,wnsimilarity,sentence1, sentence2,icneed=False):
        """ compute the sentence similarity using Wordnet """
        # Tokenize and tag
        sentence1 = pos_tag(word_tokenize(sentence1))
        sentence2 = pos_tag(word_tokenize(sentence2))
     
        # Get the synsets for the tagged words
        synsets1 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence1]
        synsets2 = [self.tagged_to_synset(*tagged_word) for tagged_word in sentence2]
     
        # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]
     
        
        score, count = 0.0, 0
        # For each word in the first sentence
        for synset in synsets1:
            
            # Get the similarity value of the most similar word in the other sentence
            score_list=[]
            if icneed == True :
                for ss in synsets2:
                    try:
                        temp=wnsimilarity(synset,ss,self.brown_ic)
                        score_list.append(temp)
                    except:
                        continue
                
            else:
                for ss in synsets2:
                    try:
                        temp=wnsimilarity(synset,ss)
                        score_list.append(temp)
                    except:
                        continue
            
                
            score_list = np.array(score_list, dtype=np.float64)
            score_list = np.nan_to_num(score_list)
#            print(score_list)
            if len(score_list)>0:
                best_score = np.nanmax(score_list)
            else:
                best_score=0.0
#            print(best_score)
#            print(type(best_score))
     
            # Check that the similarity could have been computed
            if best_score is not None:
                score =score + best_score
#                print(score)
                count = count+ 1
            
            
#        print("one sentence over")
        # Average the values
        score /= count
        return score

    def symmetric_sentence_similarity(self,wnsimilarity,sentence1, sentence2,icneed=False):
        """ compute the symmetric sentence similarity using Wordnet """
        if icneed==True:
            return (self.sentence_similarity(wnsimilarity,sentence1, sentence2,icneed=True) + self.sentence_similarity(wnsimilarity,sentence2, sentence1,icneed=True)) / 2 
        else:
            return (self.sentence_similarity(wnsimilarity,sentence1, sentence2) + self.sentence_similarity(wnsimilarity,sentence2, sentence1)) / 2 
    

#print(symmetric_sentence_similarity(wn.path_similarity,"Cats are beautiful animals.", "Dogs are awesome."))


    def load_X(self,sent_pairs):
        """Create a matrix where every row is a pair of sentences and every column in a feature.
        """
        """
        Path Similarity
        Leacock-Chodorow Similarity
        Wu-Palmer Similarity
        Resnik Similarity
        Jiang-Conrath Similarity
        Lin Similarity
        """
        
        X = np.zeros((len(sent_pairs), 2))
        path=[]
        lch=[]
        wup=[]
        res=[]
        jcn=[]
        lin=[]
        for pair in sent_pairs:
            t1 = pair[0]
            t2 = pair[1]
            path.append(self.symmetric_sentence_similarity(wn.path_similarity,t1,t2))
            lch.append(self.symmetric_sentence_similarity(wn.lch_similarity,t1,t2))
            wup.append(self.symmetric_sentence_similarity(wn.wup_similarity,t1,t2))
            res.append(self.symmetric_sentence_similarity(wn.res_similarity,t1,t2,icneed=True))
            jcn.append(self.symmetric_sentence_similarity(wn.jcn_similarity,t1,t2,icneed=True))
            lin.append(self.symmetric_sentence_similarity(wn.lin_similarity,t1,t2,icneed=True))
            
        all_list=[path,lch]
        for i in range(len(all_list)):
            X[:,i]=np.asarray(all_list[i])
        
        return X