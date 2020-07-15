#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
from numpy import random
import numpy as np
import os


# In[4]:


# Channel Shift ảnh 
def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img
# Xoay ảnh 
def rotation(img, angle):
    #angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img


# In[16]:


# folder = input("Nhập tên folder muốn tạo đi bro: ")
# createf = folder.replace(".com",'')

mode = input("Nhập vào loại folder muốn aug ")
name_file = input("Nhập vào tên file cần có ")
file_name = input("Nhập tên folder cần thay đổi nào: ")
sampleNum = 1
createf = file_name + '-' + mode.capitalize()

if mode.lower() == "test":
    path = '/home/lahai/Desktop/Test/'
else:
    path = '/home/lahai/Desktop/Train/'
pathFolder = path + file_name
for imgs in os.listdir(pathFolder):
    file_name = pathFolder + '/'+imgs
    image = cv2.imread(file_name)
    if not os.path.exists(path +'/'+createf): os.mkdir(path +'/'+createf)
    # Thểm ảnh xoay
    for i in (-90,90):
        cv2.imwrite(path+'/'+createf+"/"+name_file+'-r'+str(i)+ str(sampleNum) + ".jpg", cv2.resize(rotation(image,i),(216,216)))
    # Thêm ảnh đã cắt
    cv2.imwrite(path+'/'+createf+"/"+name_file+'-'+ str(sampleNum) + ".jpg", cv2.resize(image,(216,216)))
    # Thêm ảnh đã 'channel shift'
    #cv2.imwrite(path+'/'+createf+"/"+name_file+'-sh'+ str(sampleNum) + ".jpg", cv2.resize(channel_shift(image,255),(216,216)))
    # Thêm ảnh đã 'Horizal Flip'o
    cv2.imwrite(path+'/'+createf+"/"+name_file+'-hf-'+ str(sampleNum) + ".jpg", cv2.resize(cv2.flip(image, 1),(216,216)))
    # Thêm ảnh đã 'Vertical Flip'
    cv2.imwrite(path+'/'+createf+"/"+name_file+'-vf-'+ str(sampleNum) + ".jpg", cv2.resize(cv2.flip(image, 0),(216,216)))
    sampleNum+=1


# In[ ]:




