#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:27:50 2020

@author: lg
"""
from PIL import Image
import matplotlib.pyplot as plt
#pil_im = Image.open('1.jpg').convert('L') #灰度操作
 


import os
import cv2
array_of_img = [] # this if for store all of the image data

name_list=[]
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        pname=os.listdir(directory_name+"/"+filename)
        img = cv2.imread(directory_name + "/" + filename+"/"+pname[0])
        array_of_img.append(img)
        #print(img)
        """
        plt.figure("love")
        plt.imshow(img)
        plt.show()
        """
        print(filename)
        name_list.append(filename)
#        print(array_of_img)
        
        
        
read_directory("./data/test_images")


import insightface
import urllib
import urllib.request
import cv2
import numpy as np



#

model = insightface.app.FaceAnalysis()

################################################################
# Use CPU to do all the job. Please change ctx-id to a positive number if you have GPUs
#

ctx_id = -1


################################################################
# Prepare the enviorment
# The nms threshold is set to 0.4 in this example.
#

model.prepare(ctx_id = ctx_id, nms=0.4)

################################################################
# Analysis faces in this image


face_embed=[]
for n in array_of_img:
#faces = model.get(array_of_img[0])fav
    faces = model.get(n)
#faces = model.get(img)
    for idx, face in enumerate(faces):
      print("Face [%d]:"%idx)
      print("\tage:%d"%(face.age))
      gender = 'Male'
      if face.gender==0:
        gender = 'Female'
      print("\tgender:%s"%(gender))
      print("\tembedding shape:%s"%face.embedding.shape)
      face_embed.append(face.embedding)
      print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
      print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
      print("")

face_dict=dict(zip(name_list,face_embed))


import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
#transform2=transforms.Compose([transforms.ToTensor()])
#tensor2=transform2(frame)

def insight_frame(frame):
    
    name_list=[]
    temp = model.get(frame)
    vect=0
    disv1=0
    if len(temp)>0:
#        print('face number is ',len(temp))
        for k in temp:

            vect=k.embedding
            distt=[]
            
            for j in face_dict.values():
                disv=vect-j
                disv1=disv*disv
                distt.append(sum(disv1))
            dd=pd.Series(distt,face_dict.keys()) 
            
            dd1=dd.sort_values()
            if dd1.values[0]<500:

                print('insight dist',dd1.values[0])
                name_list.append(dd1.index[0])
#            print(dd1.index[0])
            return name_list,vect,disv1
    else:
        return 0 ,0,0        




