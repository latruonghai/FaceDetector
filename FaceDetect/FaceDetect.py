#!/usr/bin/env python
# coding: utf-8

# In[1]:


#La Truong Hai - 18520698
# Import thư viện cv2 để detect face
import numpy as np
from numpy import random
import cv2
# Import thư viện os để dẫn đường dẫn đến thư mục
import os
import dlib


# In[2]:

# Sử dụng bộ lọc harcascade
faceCasCade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)


# In[3]:


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


# In[14]:
 def detectFace():
    # Doc hinh
    #hai = ['./MTPs/MTP'+str(i)+'.jpg' for i in range(1,188) if i not in [89,152]]
    files = input("Nhập tên folder chứa ảnh mặt đã cắt : ")
    createf = files.replace(".com","")
    name_file = input("Nhập tên file đi nào: ") # Nhập tên files muốn lưu
    name_folder = input("Nhập tên folder chứa ảnh cần cắt: ") # Nhập tên folder dẫn đến thư mục chứa ảnh cần cắt mặt
    sampleNum = 0
    path = '/home/lahai/Downloads/MyTam/' # Đường dẫn dẫn đến thư mục muốn cắt ảnh mặt
    path_dir = "paths" # Đường dẫn đến thư mục muốn lưu ảnh mặt
    pathFolder = path + name_folder # Đường dẫn đến thư mục muốn cắt ảnh
    for img1 in os.listdir(pathFolder):
        file_name = pathFolder + '/'+img1
        img = cv2.imread(file_name)
        #print(img.shape)
        #img = cv2.equalizeHist(img)
        #img = cv2.resize(img,(400,400))
        img = cv2.resize(img,(400,int(img.shape[0]*400/img.shape[1])))
        # Covert ảnh từ BGR sang YUV
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # Trích xuất phần đặc trưng trong ảnh (các, phần tử quan tronjg)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        #grayImag = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face = faceCasCade.detectMultiScale(
                img_output,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30,30))
        #Vẽ các đường màu xanh lá quanh khuôn mặt
        if len(face) <=1:
            for (x,y,w,h) in face:
                sampleNum=sampleNum+1
                image = img[y:y+h,x:x+w]
                #Lưu ảnh khuôn mặt vào thư mục có tên(creatình
                if not os.path.exists(path_dir +'/'+createf): os.mkdir(path_dir +'/'+createf)
                # Thểm ảnh xoay
                # Thêm ảnh ã cắt
                try:
                    cv2.imwrite(path_dir+'/'+createf+"/"+name_file+'-'+ str(sampleNum) + ".jpg", cv2.resize(image,(216,216)))
                except:
                    continue
            # Hiển thị ra màn hình
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)
        #if sampleNum>60:
         #   break
            cv2.imshow("Face Detection ", img)
            cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




