import scipy.ndimage as ndi
import skimage.morphology as morph
import skimage.filters as skifilter
import skimage.color as skicolor
import skimage.draw as skidraw
import numpy as np
import matplotlib.pyplot as plt
from frangi_2d import frangi_2d
import matplotlib
import pickle
matplotlib.rcParams["backend"] = "Agg"
from stentiford import stentiford

'''Implementação do método sugerido por Calvo em Automatic detection and characterisation of retinal vessel
tree bifurcations and crossovers in eye fundus images'''
def Calvo_implementation(Image,Th,Tw):
    x_add = [-1,0,1,1,1,0,-1,-1,-1]# Começa com a ultima posição pois se faz atual-anterior 
    y_add = [-1,-1,-1,0,1,1,1,0,-1]

    elem = np.ones((15,15))
    Image = Image[:,:,1]
    Image = morph.black_tophat(Image,elem)
    arr_sigma = np.linspace(0.1, 3, 6)
    Image = frangi_2d(Image,arr_sigma)
    Image_array= Image.flatten()
    Image_array = np.sort(Image_array,axis=None)
    Tw = np.percentile(Image_array,Tw)
    Th = np.percentile(Image_array,Th)
    #  Tw_zip = zip(percentile_array,Tw)
    #  print(list(Tw_zip))
    #  print(Th)
    Image = skifilter.apply_hysteresis_threshold(Image,Tw,Th)

    L,N = ndi.label(Image)
    T = ndi.sum(Image,L, range(1,N+1))
    threshold = 200
    comp2remove = np.nonzero(T<threshold)[0] + 1

    num_rows,num_cols = L.shape

    Image_copy = np.copy(Image)

    for row in range(num_rows):
        for col in range(num_cols):
            label =  L[row,col]
            if label in comp2remove:
                Image_copy[row,col] = 0


    # Image_copy = ndi.median_filter(Image_copy,3)
    for i in range(4):
        Image_copy = morph.binary_dilation(Image_copy)

    for i in range(4):
        Image_copy = morph.binary_erosion(Image_copy)
    

    Image_copy = Image_copy.astype(np.int16)
    Image_copy = stentiford(Image_copy)
    

    #Percorre os vizinhos fazendo a somatória ds seu valores
    Image_labels = Image_copy * 255
    Image_labels = skicolor.gray2rgb(Image_labels)
    row_max, col_max = Image_copy.shape
    for row in range(row_max):
        for col in range(col_max): 
            if Image_copy[row,col] == 1:
                sum = 0
                x_ant = x_add[-1]
                y_ant = y_add[-1]
                for x,y in zip(x_add,y_add):
                    sum += abs(Image_copy[row + x_ant, col+y_ant] - Image_copy[row+x,col+y])
                    x_ant = x
                    y_ant = y

                sum *= 0.5

                if sum > 2 :
                    rr,cc = skidraw.circle_perimeter(row,col,5)
                    Image_labels[rr,cc] = (255,0,0)
                   


    plt.imshow(Image_labels)
    plt.show()
    return Image_copy


Image_original = plt.imread("/home/isaacw/Documentos/repos/DRIVE/training/images/21_training.tif") 
mask = plt.imread("/home/isaacw/Documentos/repos/DRIVE/training/mask/21_training_mask.gif")
mask = morph.binary_erosion(mask)

Image_obtida = Calvo_implementation(Image_original,95,90)
# Image_obtida = np.bitwise_and(Image_obtida,mask)

plt.imshow(Image_obtida, 'gray')
plt.show()