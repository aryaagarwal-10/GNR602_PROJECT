import tifffile as tiff
imgo=tiff.imread('/Users/sdl/Desktop/GNR_project/data.tif')
print(imgo.shape)
import cv2
from skimage.feature import graycomatrix
import math
import numpy as np
from matplotlib import pyplot as plt
max=255
k=10
w=[3]
g=32
init1=0
init2=0
size1=100
size2=100
img= imgo[init1:init1+size1, init2:init2+size2, :]





def compute_glcm(img, window_size,g):
    glcm = np.zeros((g,g))
    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1] - 1):
            p = img[i, j]
            q = img[i+1, j+1]
            glcm[p,q] += 1
            glcm[q,p] += 1
    #glcm = glcm / np.sum(glcm)
    #glcm = graycomatrix(img, distances=[1], angles=[90], levels=256,
                       # symmetric=True, normed=False)
    #print(glcm.shape,np.count_nonzero(glcm))
    #print(glcm)
    sum1 = 0
    energy = 0
    homo = 0
    asm = 0
    ent = 0
    
 #glcm = np.array([[1,2],[4,5]])
    #np.seterr(divide='ignore', invalid='ignore')
    for i in range(0, glcm.shape[0]):
         for j in range(0, glcm.shape[1]):
            sum1 = sum1 + ((i-j)*(i-j)*glcm[i,j])
            energy = energy + glcm[i,j]*glcm[i,j]
            homo = homo + (glcm[i,j]/(1+abs(i-j)))
            asm = asm + glcm[i,j]*glcm[i,j]
            #ent = ent + glcm[i,j]*(math.log(1/glcm[i,j]))
            ent=1
    ##features_glcm = [sum1, energy, homo, asm, ent]

    return sum1, energy, homo, asm, ent
    
    
    
    
    
    # Calculate the GLCM properties
    #contrast = np.sum((np.arange(256)[:,np.newaxis,np.newaxis] - np.arange(256)[np.newaxis,:,np.newaxis])**2 * glcm)
    #homogeneity = np.sum(glcm / (1 + (np.arange(256)[:,np.newaxis,np.newaxis] - np.arange(256)[np.newaxis,:,np.newaxis])**2))
    #energy = np.sum(glcm**2)
    #x,y = np.meshgrid(np.arange(256), np.arange(256))
    #correlation = np.sum(((x - np.mean(glcm)) * (y - np.mean(glcm))) * glcm) / (np.std(glcm) ** 2)
    #return np.array([contrast, homogeneity, energy])