# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:11:57 2019

@author: SRIKANT
"""


import random

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos,neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi,'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l)
                lexicon+= list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_count = Counter(lexicon)
    l2 = []
    for w in w_count:
        if 1000>w_count[w]>50:
           l2.append(w)
    print(len(l2))    
    return l2

def sample_handling(sample,lexicon,classification):
    featureset=[]
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l)
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value]+=1
            features = list(features)
            featureset.append([features,classification])
    return featureset
    
    

    
def create_feature_sets_labels(pos,neg,test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features = sample_handling('pos.txt',lexicon,[1,0])
    features = sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    
    features = np.array(features)
    #testing_size = int(test_size*len(features))
    #train_X = list(features[:,0][:-testing_size])
    #train_y = list(features[:,1][:-testing_size])
    #test_X = list(features[:,0][-testing_size:])
    #test_y = list(features[:,1][-testing_size:])
    
    
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=700)

    for train_index,test_index in kfold.split(features):
        print(train_index,test_index)
    train_X,train_y,test_X,test_y = list(features[:,0][train_index]), list(features[:,1][train_index]),list(features[:,0][test_index]),list(features[:,1][test_index])

    
    return train_X,train_y,test_X,test_y

