
# coding: utf-8


import sys
import re
import jieba
import os
import gensim
from gensim.models import KeyedVectors
from gensim.models import word2vec
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance
from scipy import spatial
import math
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import argparse

#clean annotation
def clean(file):
    with open (file,"r",encoding="utf-8")as text:
        #.read().decode('utf8','ignore') 
        cleaned = []
        final = []
        error_reg = '\n|{[a-zA-Z0-9]*\s*[\,\-\。\?\+]*[a-zA-Z0-9]*[\u2E80-\u9FFF]*\s*\}*|[a-zA-Z0-9]+}|\[[a-zA-Z0-9]*\s*[\,\，\-\。\?\“\”\……\〉、#]*\s*[\u2E80-\u9FFF]*\s*\]'
        for txt in text:  
            new = re.sub(error_reg,'',txt,count = 0, flags = 0)
            cleaned.append(new)
        for i in cleaned:
            final_data = re.sub(error_reg,'',i,count = 0, flags = 0)
            final.append(final_data)
    return final

#  load stopword list
def load_stopword():
    f_stop = open('stopwords.txt', encoding='utf-8')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

# lexicalize, remove stopwords and punctuation
file_userDict = 'dict.txt'
jieba.load_userdict(file_userDict)
stopwords = load_stopword()

# word segmentation
def seg_word(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = [word for word in sentence_seged if word not in stopwords]
    outstr = [word for word in outstr if word.strip()]
    outstr = " ".join(outstr)
    return outstr


#store string into the new file
def store(filename,sentences):
    doc = open(filename,'w',encoding="utf-8")
    doc.write(sentences)
    doc.close()
    return doc


# In[5]:


#preprocessing function
def preprocess(file):    
    text = str(clean(file))
    words = seg_word(text)    
    return words

#Preprocess the training data and store them into new files
txts = []
for root, dirs, files in os.walk("data"):
    for file in files:
        path = os.path.join(root, file).replace("\\","/")
        result =preprocess(path)
        path = path.replace("data","newData")
        store(path,result)


#load pretrained word2vec model
model=gensim.models.KeyedVectors.load_word2vec_format('baike_26g_news_13g_novel_229g.bin', binary=True)

#get each word's vector of one document
def get_vec(doc):
    all_vec = {}
    with open (doc) as file: 
        content = file.read()
        content_list = content.split(' ')
        content_list.remove('\n')
        while '' in content_list: 
            content_list.remove('') 
        for i in content_list:
            vector = model[i]
            all_vec.update({i:vector})
    return all_vec

#feaature 1: averaged word embeddings
def get_average(all_vec):
    vec_list = all_vec.values()
    s = pd.Series(all_vec)
    averaged = s.mean(axis=0)
    return averaged

#calculate similarity between word embeddings
def sim(vec1,vec2):
    cosine_similarity = 1 - spatial.distance.cosine(vec1, vec2)
    return cosine_similarity


#get idf score
def get_idf(path):
    wholedict={}
    idfdict = {}
    n=0
    txts = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace("\\","/")
            n +=1
            #split according to blank
            with open(file,'r',encoding="utf-8") as f:
                content = f.read()
                content_list = content.split(' ')
            while '\n' in content_list:
                content_list.remove('\n')
            while '' in content_list:
                content_list.remove('')
            word_count = Counter(content_list)
            wordlist = word_count.keys()
            for i in wordlist:
                if i in idfdict.keys():
                    idfdict[i] += 1     
                else:
                    idfdict.update({i:1})
        for i in idfdict:
            a = n / idfdict[i] 
            idfdict[i] = math.log(a)    
    return idfdict            
idf_dict = get_idf('newData')

#feature 2: get weighted word embeddings
def get_weighted_vec(doc):
    all_vec = []
    weighted_dict={}
    with open (doc) as file: 
        content = file.read()
        content_list = content.split(' ')
        content_list.remove('\n')
        while '' in content_list: 
            content_list.remove('') 
        for i in vec3.keys():
            vector = vec3[i]
            para = idf_dict[i] 
            weighted = vector * para
            weighted_dict.update({i:weighted})
    return weighted_dict


# In[ ]:


#特征3：word mover distance
def get_WMD(doc_vec1, doc_vec2):
    distance = model.wmdistance(doc_vec1, doc_vec2)
    return distance

#construct and visualize the similarity grids
def sim_grid(doc1,doc2):
    vec1 = get_vec(doc1)
    vec2 = get_vec(doc2)
    list1=list(vec1.values())
    list2=list(vec2.values())
    wholelist = []
    i = 0
    while i<len(list1):    
        a = list1[i]
        j = 0  
        simlist = []
        while j<len(list2):     
            b = list2[j]
            similarity= sim(a,b)            
            simlist.append(similarity)  
            j+=1   
        i+=1
        wholelist.append(simlist)
    return wholelist

#visualize the similarity grids
def sim_pic(sim_matrix):
    ar = np.asmatrix(sim_matrix)
    sim_picture = plt.matshow(ar)
    plt.savefig('./my_figure_name.png')
    return sim_picture


#Put the previous functions together
def get_pic(doc1,doc2):
    a = sim_grid(doc1,doc2)
    b = sim_pic(a)
    return b


