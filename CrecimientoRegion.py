# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 07:21:52 2022

@author: Sala8
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, img_as_float, color, morphology
from skimage.morphology import square
import cv2
import os
from tqdm import tqdm 

from PIL import Image 
import glob
import statistics

#C:\Users\luisa\Desktop\newMRI
gris=color.rgb2gray(io.imread("C:/Users/luisa/Desktop/newMRI/tumor466.jpg"))
#gris= io.imread('Y3.jgp'5
plt.imshow(gris, cmap='gray')
plt.ion()

tolerancia=0.12
fil, col = gris.shape
semilla = np.int32(plt.ginput(0,0)) #valores enteros de 8 bits

phi1 = np.zeros((fil, col), dtype=np.byte) #variable llena de ceros 0 negro 1 blanco, imagen nueva
phi2 = np.zeros((fil, col), dtype=np.byte) #imagen anterior

phi1[semilla[:,1], semilla[:,0]] = 1

pixeles = gris[semilla[:,1], semilla[:,0]]
promedio = np.mean(pixeles)

while (np.sum(phi2)!=np.sum(phi1)):
    plt.cla()
    phi2 =np.copy(phi1) 
    bordes = morphology.binary_dilation(phi1)- phi1
    newpos = np.argwhere(bordes) #busca lo que sea diferente de cero, regresa el numero d filas y columnas
    newpix = gris[newpos[:,0], newpos[:,1]]
    compara = list(np.logical_and([newpix>promedio-tolerancia], [newpix<promedio+tolerancia]))
    datos = newpos[compara]
    phi1[datos[:,0], datos[:,1]]=1
    plt.imshow(phi1, cmap='gray')
    plt.pause(0.02)
    

plt.figure(2)
plt.imshow(gris, cmap='gray')



binario = morphology.closing(phi1,morphology.diamond(5))
plt.figure(3)
plt.axis('off')
plt.imshow(binario, cmap='gray')

#Se tiene que poner el nombre de la carpeta y el nombre de la imagen 
plt.savefig("C:/Users/luisa/Desktop/mascaras_crecimiento/mascara466.jpg",bbox_inches='tight', transparent=True, pad_inches=0)


