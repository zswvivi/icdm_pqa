#import argparse
#import math
import os
#import re
#import sys
import gzip
#import json
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
#import statistics
#from collections import Counter
#import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize

PAD_WORD = '[PAD]'
data_path = './data'

# data loading functions:
def parse(path):
    g = gzip.open(path, 'rb')
    #g = open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def generate_QA_pairs(t):
    temp=[]
    for i in range(len(t['questions'])):
        temp.append([t['asin'],t['questions'][i]]) 
                     #,t['questions'][i]['questionText'],t['questions'][i]['answers'][0]['answerText']])    
    return temp

def tokenizer_sentence(t):
        sentences = sent_tokenize(t)
        sentences = sorted(sentences, key=len)
        sentences = list(dict.fromkeys(sentences))
        return sentences
    
categories = ['Tools_and_Home_Improvement','Patio_Lawn_and_Garden','Automotive','Cell_Phones_and_Accessories','Health_and_Personal_Care','Sports_and_Outdoors','Home_and_Kitchen']

all_training = {}

for category in categories:
    reviews=os.path.join(data_path,'reviews_'+category+'.json.gz')
    qa=os.path.join(data_path,'QA_'+category+'.json.gz')
    da = getDF(qa)
    #da = da[:300] # for test purpose
    dr = getDF(reviews)

    # Generate qa pairs
    qa_pairs = da.apply(generate_QA_pairs,axis=1)
    qa_pairs = [item for sublist in qa_pairs for item in sublist]
    qa_pairs =pd.DataFrame(qa_pairs,columns=['asin','QA'])
    asins=da.asin.unique()

    # Only keep reviews whose product has QA pairs
    dr=dr[dr.asin.isin(asins)]
    dr=dr.reset_index(drop=True)
    dr = dr[['asin','reviewText']]
    
    # split review into sentences for each product
    review_agg=dr.groupby('asin')['reviewText'].apply(lambda x: "%s" % ' '.join(x)).reset_index()
    review_agg['reviewText']=review_agg['reviewText'].apply(tokenizer_sentence)
    
    qa_withReivews = pd.merge(qa_pairs, review_agg, how='inner', on=['asin'])
    
    # suffle dataset with a random seed (First 80% for traininig, then 10% for dev and 10% for test)
    qa_withReivews = shuffle(qa_withReivews,random_state=30)
    qa_withReivews = qa_withReivews.reset_index(drop=True)
    if(qa_withReivews.isnull().values.any()):
        qa_withReivews = qa_withReivews.replace(np.nan, PAD_WORD, regex=True)
    qa_withReivews.to_csv(os.path.join(data_path, category+'.txt'), index=None, sep='\t', mode='w')
    all_training[category] = qa_withReivews[:int(len(qa_withReivews)*0.8)]
    
ALL = []
for key in all_training.keys():
    temp = all_training[key]
    temp['category'] = key
    ALL.append(temp)
ALL = pd.concat(ALL, ignore_index=True)
ALL.to_csv(os.path.join(data_path, 'ALL.txt'), index=None, sep='\t', mode='w',encoding='utf-8')
