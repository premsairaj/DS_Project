# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 20:46:16 2022

@author: ppallapotu
"""

from tensorflow.keras import activations, optimizers, losses
import pandas as pd
import numpy as np
dataset = pd.read_csv(open(r'C:\Users\ppallapotu\Downloads\SMSSpamCollection.txt'),sep='\t',names=["label", "message"])
from tensorflow.keras import layers,Sequential
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
from keras.preprocessing.text import one_hot
onehot_repr=[one_hot(words,10000)for words in dataset['message']]
from keras.preprocessing.sequence import pad_sequences

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=100)
inform=pd.get_dummies(dataset['label'])
model=Sequential()
model.add(layers.Embedding(10000,20,input_length=100))
model.add(layers.SimpleRNN(124))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='rmsprop')
model.summary()
history=model.fit(embedded_docs,y,batch_size=8,epochs=10)
output=model.predict(embedded_docs)
#1---->spam
#0----->ham
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(embedded_docs,y,test_size=0.2)
history=model.fit(x_train,y_train,batch_size=8,epochs=10)
model.evaluate(x_test,y_test)
output=model.predict(x_test)

