#!/usr/bin/env python
# coding: utf-8

# # Loading libraries and training data in DF

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


train_path_neg="/Users/sameertele/Machine Learning Project Sem 2/data/train/neg"
train_path_pos="/Users/sameertele/Machine Learning Project Sem 2/data/train/pos"


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


train_neg_df=load_files_with_class(train_path_neg,0)


# In[24]:


train_pos_df=load_files_with_class(train_path_pos,1)


# In[25]:


train_df = pd.concat([train_neg_df,train_pos_df])
train_df.reset_index(inplace=True,drop=True)


# In[26]:


train_df.tail()


# # Pre Processing

# ## Defining individual preprocessing functions

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


#Used lemmatization insetead of stemming
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


final_string, list_of_sent=preprocessing(train_df["Contents"])


# In[33]:


print(final_string[0])


# In[34]:


train_df.loc[0]["Contents"]


# In[35]:


list_of_sent[0]


# In[36]:


train_df["cleaned_content"] = final_string


# In[37]:


train_df.head


# In[38]:


with open('Cleaned_train_sentences.pickle', 'wb') as file:
    pickle.dump(train_df, file)


# ## Vectorization

# In[22]:


tfidf=TfidfVectorizer()
tfidf_vector=tfidf.fit_transform(train_df["cleaned_content"].values)


# In[23]:


tfidf_feat = tfidf.get_feature_names_out()
(tfidf_feat)


# In[24]:


# Convert the GloVe file to the Word2Vec format
glove_file = 'glove.6B.200d.txt'
word2vec_output_file = 'glove.6B.200d.word2vec'
glove2word2vec(glove_file, word2vec_output_file)


# In[25]:


w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)


# In[26]:


def calc_tfidf_avg_w2v(list_of_sent, w2v_model, tf_idf, tfidf_feat):
    '''This function takes in 4 parameters as follows:
       1. list_of_sent - This is the list of sentences/reviews for which sentence vetors are to be constructed
       2. w2v_model - This is the Word2Vec model which is trained on the working corpus - contains the word vectors
       3. tf_idf - This is the TF-IDF model built using the same reviews/sentences - it is the TF-IDF sparse matrix
       4. tfidf_feat - This is the feature vector constructed from the TF-IDF model
       
       Return Value:
       tfidf_sent_vectors - This is a list of sentence/review vectors constructed by using tfidf weighted average on the word vectors
    '''
    
    ## Initialize an empty list
    tfidf_sent_vectors = []
    row = 0
    ## Consider one sentence/review at a time
    for sent in list_of_sent:
        ## Initialize sentence vector to 0
        sent_vec = np.zeros(200)
        ## Initialize weighted sum to 0
        weighted_sum = 0
        ## Consider the words one by one
        for word in sent:
            try:
                ## Calculate the word vector using the W2V model
                vec = w2v_model[word]
                ## Calculate tfidf value of the word in that review using tfidf model
                tfidf = tf_idf[row, np.where(tfidf_feat == word)[0][0]]
                ## Add the product of tfidf*word_vec to the sentence vector (This is the numerator)
                sent_vec += vec*tfidf
                ## Sum all the tfidf values (This is the denominator)
                weighted_sum += tfidf
            except:
                pass
        #print(row, weighted_sum)
        
        ## Divide the numerator by the denominator to get the sentence vector
        sent_vec /= weighted_sum
        ## Add the sentence vector in the final list
        tfidf_sent_vectors.append(sent_vec)
        row += 1
    ## return the list of all the sentence vectors
    return tfidf_sent_vectors


# In[27]:


tfidf_weighted_w2v=calc_tfidf_avg_w2v(list_of_sent,w2v_model,tfidf_vector,tfidf_feat)


# In[39]:


df_train_vectors = pd.DataFrame(tfidf_weighted_w2v)
df_train_vectors["Class"]=train_df["class"]


# In[40]:


df_train_vectors.shape


# In[42]:


df_train_vectors.head()


# In[28]:


import pickle

with open('TF-IDF_glove_weights.pickle', 'wb') as file:
    pickle.dump(tfidf_weighted_w2v, file)


# In[43]:


with open('TF-IDF_glove_weights_train_dataframe.pickle', 'wb') as file:
    pickle.dump(df_train_vectors, file)


# In[29]:


tfidf_weighted_w2v[0]


# In[72]:


len(tfidf_weighted_w2v)


# In[49]:


vec = w2v_model["working"]
vec


# In[65]:


tfidf_vector[0, np.where(tfidf_feat == 'working')[0][0]]


# In[61]:


np.where(tfidf_feat == 'working')[0][0]


# In[64]:


tfidf_vector

