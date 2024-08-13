#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import os
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle
import gensim
import seaborn as sns
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# In[21]:


test_path_neg="/Users/sameertele/Machine Learning Project Sem 2/data/test/neg"
test_path_pos="/Users/sameertele/Machine Learning Project Sem 2/data/test/pos"


# In[22]:


# LOADS FILES FROM THE GIVEN DIREFCTORY INTO A DATAFRAME WITH COLUMN AS THE CONTENTS AND CLASS VALUE PRVOIDED
def load_files_with_class(directory,class_name):
    df = pd.DataFrame(columns=['Contents','class'])
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
        #Open and read the file         
            with open(os.path.join(directory, file), 'r') as f:
                content = f.read()
                df.loc[len(df)]=[content,class_name]
    return df


# In[23]:


test_neg_df=load_files_with_class(test_path_neg,0)


# In[24]:


test_pos_df=load_files_with_class(test_path_pos,1)


# In[25]:


test_df = pd.concat([test_neg_df,test_pos_df])
test_df.reset_index(inplace=True,drop=True)


# In[26]:


test_df.tail()


# ## Preprocessing

# In[27]:


stop = set(stopwords.words('english')) #set of stopwords


# In[28]:


def cleanhtml(sentence):
    '''This function removes all the html tags in the given sentence'''
    cleantext = re.sub('<.*?>', ' ', sentence) 
    return cleantext


# In[29]:


def cleanpunc(sentence):
    '''This function cleans all the punctuation or special characters from a given sentence'''
    cleaned = re.sub(r'[?|@|!|^|%|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned


# In[30]:


lemmatizer = WordNetLemmatizer() 


# In[31]:


def preprocessing(series):
    '''The function takes a Pandas Series object containing text in all the cells
       And performs following Preprocessing steps on each cell:
       1. Clean text from html tags
       2. Clean text from punctuations and special characters
       3. Retain only non-numeric Latin characters with length > 2
       4. Remove stopwords from the sentence
       5. Lemmatization
       
       Return values:
       1. final_string - List of cleaned sentences
       2. list_of_sent - List of lists which can be used for vectorization'''
    
    i = 0
    str1=" "
    final_string = []    ## This list will contain cleaned sentences
    list_of_sent = []    ## This is a list of lists containing indivdual words
    
    for sent in series.values:
        filtered_sent = []
        sent = cleanhtml(sent)    ## Clean the HTML tags
        sent = cleanpunc(sent)    ## Clean the punctuations and special characters
        ## Sentences are cleaned and words are handled individually
        for cleaned_words in sent.split():
            ## Only consider non-numeric words with length at least 3
            if((cleaned_words.isalpha()) and (len(cleaned_words) > 2)):
                ## Only consider words which are not stopwords and convert them to lowet case
                if(cleaned_words.lower() not in stop):
                    ## Apply lemmetizer and add them to the filtered_sent list
                    s = (lemmatizer.lemmatize(cleaned_words.lower()))
                    filtered_sent.append(s)    ## This contains all the cleaned words for a sentence
        ## Below list is a list of lists
        list_of_sent.append(filtered_sent)
        ## Join back all the words belonging to the same sentence
        str1 = " ".join(filtered_sent)
        ## Finally add the cleaned sentence in the below list
        final_string.append(str1)
        #print(i)
        i += 1
    return final_string, list_of_sent


# In[32]:


final_string, list_of_sent=preprocessing(test_df["Contents"])


# In[33]:


print(final_string[0])


# In[34]:


test_df.loc[0]["Contents"]


# In[35]:


list_of_sent[0]


# In[36]:


test_df["cleaned_content"] = final_string


# In[37]:


test_df.head


# In[38]:


with open('Cleaned_test_sentences.pickle', 'wb') as file:
    pickle.dump(test_df, file)


# ## Vectorization

# In[19]:


tfidf=TfidfVectorizer()
tfidf_vector=tfidf.fit_transform(test_df["cleaned_content"].values)


# In[20]:


tfidf_feat = tfidf.get_feature_names_out()
(tfidf_feat)


# In[21]:


word2vec_output_file = '/Users/sameertele/Machine Learning Project Sem 2/glove.6B.200d.word2vec'


# In[22]:


w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


# In[23]:


def calc_tfidf_avg_w2v_optimized(list_of_sent, w2v_model, tf_idf, tfidf_feat):
    tfidf_sent_vectors = []
    word_vectors = {} 
    
    for word in tfidf_feat:
        try:
            word_vectors[word] = w2v_model[word]
        except KeyError:
            continue
    
    for row, sent in enumerate(list_of_sent):
        sent_vec = np.zeros(200)
        weighted_sum = 0
        
        for word in sent:
            if word in word_vectors:
                vec = word_vectors[word]
                tfidf = tf_idf[row, np.where(tfidf_feat == word)[0][0]]
                sent_vec += vec * tfidf
                weighted_sum += tfidf
        
        if weighted_sum != 0:
            sent_vec /= weighted_sum
        
        tfidf_sent_vectors.append(sent_vec)
    
    return tfidf_sent_vectors


# In[24]:


tfidf_weighted_w2v_optimized_new=calc_tfidf_avg_w2v_optimized(list_of_sent,w2v_model,tfidf_vector,tfidf_feat)


# In[25]:


tfidf_weighted_w2v_optimized_new[0]


# In[26]:


df_test_vectors = pd.DataFrame(tfidf_weighted_w2v_optimized_new)
df_test_vectors["Class"]=test_df["class"]


# In[28]:


df_test_vectors.tail()


# In[29]:


df_test_vectors.shape


# In[30]:


import pickle

with open('TF-IDF_glove_weights_test.pickle', 'wb') as file:
    pickle.dump(tfidf_weighted_w2v_optimized_new, file)


# In[31]:


with open('TF-IDF_glove_weights_test_dataframe.pickle', 'wb') as file:
    pickle.dump(df_test_vectors, file)

