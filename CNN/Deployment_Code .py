# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:45:56 2022

@author: ppallapotu
"""

from keras.models import load_model
import pathlib
import numpy as np
import cv2
import PIL
prem=load_model("C:\\Users\\ppallapotu\Desktop\Project\\VGG16\\VGG16model.h5")
test_ditc={'data':list(pathlib.Path("C:\\Users\\ppallapotu\\Desktop\\test").glob("*.*"))}
x_test=[]
for i,j in test_ditc.items():
  for k in test_ditc[i]:
      x_test.append(cv2.resize(np.array(PIL.Image.open(str(k)).convert("RGB")),(300,300)))
          
x_test=np.array(x_test).astype('float32')/255
y_test_pred=prem.predict(x_test)
premresnet=load_model("C:\\Users\\ppallapotu\\Desktop\\Project\\Resnet\\Resnet50.h5")
y_test_pred_res=premresnet.predict(x_test)
