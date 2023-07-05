import tifffile as tiff
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
img=tiff.imread('/Users/sdl/Desktop/GNR_project/data.tif')
print(img.shape)
import cv2

def showimg(imgi):
    plt.rcParams["figure.figsize"] = [7.3,5]
    plt.rcParams["figure.autolayout"] = True
    plt.imshow(imgi)
    plt.show()

def calculate_glcm_features(glcm):
    p = glcm / np.sum(glcm) 
    contrast = np.sum((np.arange(256) - np.arange(256)[:, np.newaxis])**2 * p)
    homogeneity = np.sum(p / (1 + np.abs(np.arange(256) - np.arange(256)[:, np.newaxis])))
    entropy = -np.sum(p * np.log2(p + 1e-12))
    energy = np.sum(p**2)
    #mean1 = np.sum(np.arange(256) * np.sum(p, axis=1))
    #mean2 = np.sum(np.arange(256) * np.sum(p, axis=0))
    #std1 = np.sqrt(np.sum(np.sum(p, axis=1) * (np.arange(256) - mean1)**2))
    #std2 = np.sqrt(np.sum(np.sum(p, axis=0) * (np.arange(256) - mean2)**2))
    #correlation = np.sum(np.outer(np.arange(256) - mean1, np.arange(256) - mean2) * p) / (std1 * std2)
    
    return np.array([contrast, homogeneity, entropy, energy])


def glcm_features(glcm):
    # calculate various GLCM features
    contrast = np.sum(glcm * np.square(np.arange(glcm.shape[0])[:, np.newaxis] - np.arange(glcm.shape[1])))
    energy = np.sum(glcm ** 2)
    asm = np.sum(glcm ** 2)
    eps = np.finfo(float).eps
    entropy = -np.sum(glcm * np.log2(glcm + eps))
    
    return contrast, energy, asm, entropy

def smi(array):

    fig, axs = plt.subplots(nrows=1, ncols=2)
    for i, ax in enumerate(axs.flat):
        ax.imshow(array[i])
    plt.show()



def glcm_matrix(img_array):


    window_sizes = [3]
    features = []

    for w in window_sizes:
        glcm = np.zeros((256, 256), dtype=np.uint32)
        for i in range(img_array.shape[0] - w + 1):
            for j in range(img_array.shape[1] - w + 1):
                window = img_array[i:i+w, j:j+w]
                for x in range(window.shape[0]):
                    for y in range(window.shape[1]):
                        if x + 1 < window.shape[0] and y + 1 < window.shape[1]:
                            glcm[int(window[x][y])][int(window[x+1][y+1])] += 1
                            glcm[int(window[x+1][y+1])][int(window[x][y])] += 1

        window_features = calculate_glcm_features(glcm)

        features.append(window_features)
        print(features)
    features = np.concatenate(features)
    return features


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    

    arr_norm = (arr - min_val) / (max_val - min_val)
    
    arr_scaled = (arr_norm * 255).astype(np.uint8)
    
    return arr_scaled


def compute_glcm_fep(img, win_size, dist, angles):
    
    rows, cols = img.shape
    features = np.zeros((rows, cols, len(angles)*5)) 
    
    for i in range(rows):
        for j in range(cols):
    
            win = img[max(i-win_size//2, 0):min(i+win_size//2+1, rows),
                      max(j-win_size//2, 0):min(j+win_size//2+1, cols)]
            
            
            glcm = np.zeros((256, 256))
            for k, (dr, dc) in enumerate(angles):
                for r in range(win_size//2, win.shape[0]-win_size//2):
                    for c in range(win_size//2, win.shape[1]-win_size//2):
                        p1 = win[r, c]
                        p2 = win[r+dr*dist, c+dc*dist]
                        glcm[p1, p2] += 1
            contrast, energy, asm, entropy=glcm_features(glcm)
            features[i, j] = asm
    
    return features































img = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
print(img.shape)

mean = np.mean(img, axis=0)
img_centered = img - mean

covariance_matrix = np.cov(img_centered.T)

print(covariance_matrix.shape)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

print(sorted_indices)
print(sorted_eigenvectors)
print(eigenvalues)

principal_components = np.dot(img_centered, sorted_eigenvectors)
#print(principal_components.shape)

pca1 = principal_components[:, 0]
pca1 = normalize_array(pca1.reshape((863, 876)))


pca2 = principal_components[:, 1]
pca2 = normalize_array(pca2.reshape((863, 876)))

#showimg(pca2)
showimg(pca1)
#print(pca1)
#angles = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
#features=compute_glcm_fep(pca1,3,1,angles)
#print(features.shape)

print(pca1)