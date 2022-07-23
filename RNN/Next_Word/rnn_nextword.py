# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 05:34:40 2022

@author: ppallapotu
"""

import nltk  
from tensorflow.keras import layers,Sequential
from gensim.models import Word2Vec
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing.text import Tokenizer
dataset = pd.read_csv(open(r'C:\Users\ppallapotu\Downloads\SMSSpamCollection.txt'),sep='\t',names=["label", "message"])


for i in list(dataset['message']):
    sentence.append(nltk.word_tokenize(i))#convert the sentence into words
model=Word2Vec(sentence,min_count=1)#convert english word into numerical representation
words=model.wv.key_to_index # it will get the index of the words
word1=model.wv['1-Hanuman']#it will give the vector of specify of length 100
wordsimilartype=model.wv.most_similar('1-Hanuman')# it will give the list of similar words 

doc = open(r'C:\Users\ppallapotu\Downloads\corpus.txt','r')
data=doc.read()
sentence=[list(str(doc.read()).split(' '))]
tk=Tokenizer()
tk.fit_on_texts([data])
encoded_data=tk.texts_to_sequences([data])[0]
vocb_size=len(tk.word_index)+1
sequence=[]
for i in range(len(encoded_data)):
    sequence.append([encoded_data[i],encoded_data[i+1]])
sequence=np.array(sequence)
x,y=sequence[:,0],sequence[:,1] 
#convert the target variable into categorical type
from keras.utils import np_utils
import pandas as pd
y=pd.DataFrame(y).astype('str')
y=pd.get_dummies(y)
s=np.array(y)
y = np_utils.to_categorical(y, vocb_size)
model=Sequential()
model.add(layers.Embedding(vocb_size,6,input_length=1))
model.add(layers.SimpleRNN(124))
model.add(layers.Dense(vocb_size,activation='softmax'))
model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')
model.summary()
model.fit(x,y,batch_size=3,epochs=5)

def predictnextword(model,word,count):
    index=''
    pred_word=str(word)
    nextword=[word]
    for _ in range(count):
        index=np.argmax(model.predict(np.array(tk.texts_to_sequences([word]))),axis=1)
        pred_word=pred_word+' '+tk.index_word[int(index)]
        word=tk.index_word[int(index)]
    return pred_word
predictnextword(model, 'surprise', count=3)        
        
        












