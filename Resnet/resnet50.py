# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:42:52 2022

@author: ppallapotu
"""

import cv2
import numpy as np
import tensorflow as tf
from keras import layers
import pathlib
import PIL
import pandas as pd
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50
import matplotlib.pyplot as plt
train_ditc={'Object_Image':list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\train\\Object Image").glob("*.*")),'Paragraph_Image':list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\train\\Paragraph Image").glob("*.*")),"Text_Image":list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\train\\Text Image").glob("*.*"))}#Extracting Image Paths
print("Train Samples Counts \n" +" Object_Image = "+str(len(train_ditc['Object_Image']))+" Paragraph_Image = "+str(len(train_ditc['Paragraph_Image']))+" Text_Image = "+str(len(train_ditc['Text_Image'])))
valid_ditc={'Object_Image':list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\valid\\Object Image").glob("*.*")),'Paragraph_Image':list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\valid\\Paragraph Image").glob("*.*")),"Text_Image":list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\valid\\Text Image").glob("*.*"))}#Extracting Image Paths
print("valid Samples Counts \n" +" Object_Image = "+str(len(valid_ditc['Object_Image']))+" Paragraph_Image = "+str(len(valid_ditc['Paragraph_Image']))+" Text_Image = "+str(len(valid_ditc['Text_Image'])))

test_ditc={'Object_Image':list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\test\\Object Image").glob("*.*")),'Paragraph_Image':list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\test\\Paragraph Image").glob("*.*")),"Text_Image":list(pathlib.Path("D:\\Data Science\\Obj_Text_Para\\test\\Text Image").glob("*.*"))}#Extracting Image Paths
print("Test Samples Counts \n" +" Object_Image = "+str(len(test_ditc['Object_Image']))+" Paragraph_Image = "+str(len(test_ditc['Paragraph_Image']))+" Text_Image = "+str(len(test_ditc['Text_Image'])))

def augment(img1):
    imglist=[]
  #rotation
    img = PIL.Image.open(img1)#Reading the Image
    img = np.array(img.convert("RGB"))#converted into RGB Channel
    img1 = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    img1=cv2.resize(img1,(300,300))
    img2 = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img2=cv2.resize(img2,(300,300))
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,-0]])
    for g in [cv2.resize(img,(300,300)),img1,img2]:
        imglist.append(cv2.filter2D(g, -1, kernel))#for Sharpening the Image applying filter2D
        
    return  imglist


x_train=[]#Extracting All train sample 
y_train=[]#Extracting All train sample labels
error=[]
for i,j in train_ditc.items():
  for k in range(len(train_ditc[i])):
     for p in range(len(augment(str(train_ditc[i][k])))):
       error.append(str(train_ditc[i][k]))
       x_train.append(augment(str(train_ditc[i][k]))[p])
       y_train.append(str(i))
       
       
x_test=[]#Exracting Test Samples
y_test=[]#Extracting Test Sample Labels
for i,j in test_ditc.items():
  for k in range(len(test_ditc[i])):
     for p in range(len(augment(str(test_ditc[i][k])))):
       x_test.append(augment(str(test_ditc[i][k]))[p])
       y_test.append(str(i))
       
x_valid=[]#Exracting Test Samples
y_valid=[]#Extracting Test Sample Labels
for i,j in valid_ditc.items():
  for k in range(len(valid_ditc[i])):
     for p in range(len(augment(str(valid_ditc[i][k])))):
       x_valid.append(augment(str(valid_ditc[i][k]))[p])
       y_valid.append(str(i))
       
       
       
       
x_train=np.array(x_train).astype('float32')/255.0
y_train=pd.DataFrame(y_train)
y_train.loc[y_train[0] == "Object_Image",0] =0#Replacing label with number
y_train.loc[y_train[0] == "Paragraph_Image",0] =1#Replacing label with number
y_train.loc[y_train[0] == "Text_Image",0] =2#Replacing label with number
x_valid=np.array(x_valid).astype('float32')/255.0
y_valid=pd.DataFrame(y_valid)
y_valid.loc[y_valid[0] == "Object_Image",0] =0#Replacing label with number
y_valid.loc[y_valid[0] == "Paragraph_Image",0] =1#Replacing label with number
y_valid.loc[y_valid[0] == "Text_Image",0] =2#Replacing label with number

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 3)
y_valid= np_utils.to_categorical(y_valid, 3)

ResNet50=ResNet50(include_top=False,classes=3,input_shape=(300,300,3))#clone the VGG16 Model
for layers in ResNet50.layers:#trainable parameters
    layers.trainable=False
ResNet50.summary()

x=Flatten()(ResNet50.output)
Preduction=Dense(3,activation='softmax',)(x)
model=Model(inputs=ResNet50.input,outputs=Preduction)
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt,metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()
history=model.fit(x_train,y_train,validation_data=(x_valid,y_valid),batch_size=16,epochs=5)
model.save('Resnet50.h5')
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'y',label='training loss')
plt.plot(epochs,val_loss,'r',label='val_loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'y',label='training accuracy')
plt.plot(epochs,val_acc,'r',label='val_accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

accuracy=[]
loss=[]
train_loss,train_acc=model.evaluate(x_train,y_train,batch_size=16)

accuracy.append(train_acc)
loss.append(train_loss)
valid_loss,valid_acc=model.evaluate(x_valid,y_valid,batch_size=16)
accuracy.append(valid_acc)
loss.append(valid_loss)
x_test=np.array(x_test).astype('float32')/255.0
y_test=pd.DataFrame(y_test)
y_test.loc[y_test[0] == "Object_Image",0] =0#Replacing label with number
y_test.loc[y_test[0] == "Paragraph_Image",0] =1#Replacing label with number
y_test.loc[y_test[0] == "Text_Image",0] =2#Replacing label with number
from keras.utils import np_utils
y_test = np_utils.to_categorical(y_test, 3)
test_loss,test_acc=model.evaluate(x_test,y_test,batch_size=16)
accuracy.append(test_acc)
loss.append(test_loss)
plt.bar(np.array(['train_Acc','Valid_acc','Test_acc']),accuracy,width=0.3)
plt.ylabel('accuracy')
plt.show()
plt.bar(np.array(['train_loss','Valid_loss','Test_loss']),loss,width=0.3)
plt.ylabel('loss')
plt.show()