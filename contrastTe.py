# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:49:02 2022

@author: Lander
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm 

def CLAHEOneChannel(imagenIn):
    # Converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((img, enhanced_img))
    
    #Resize
    down_width = 250
    down_height = 250
    down_points = (down_width, down_height)
    resized_down = cv2.resize(enhanced_img, down_points, interpolation= cv2.INTER_LINEAR)

    return resized_down

data_path_trainImg = "MRI_Ent/" 
data_path_NewImg = "newMRI/" 
# obtenemos una lista con los archivos dentro de cada carpeta
data_list_trainImg = os.listdir(data_path_trainImg)

i=1

for file in tqdm(data_list_trainImg):
    img = cv2.imread(data_path_trainImg + file, 1)
    
    
    
    final = CLAHEOneChannel(img)
    
    #print("\nImagenes: " + data_path_trainImg + file)
    
    #plt.figure(i)
    #plt.imshow(final, cmap='gray')
    cv2.imwrite(data_path_NewImg + "/tumor" + str(i) + ".jpg", final)
    i = i+1
    
    














