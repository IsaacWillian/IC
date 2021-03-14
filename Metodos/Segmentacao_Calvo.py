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
from stentiford import stentiford
from Ordenacao_Perimetro import ordenar_perimetro
import tifffile

def plot_image(img,title="1"):
    fig  = plt.figure(figsize=(20,20))
    print(type(img))
    plt.imshow(img,"gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    tifffile.imwrite('Calvo_results/09_manual1.tif',img.astype(np.uint8)*255)


def Calvo_segmentation(img,Th,Tw,elem,threshold,dilation):
    elem_estru = np.ones((elem,elem))
    Image = img[:,:,1]
    
    Image = morph.black_tophat(Image,elem_estru)
    
    #Image = ndi.median_filter(Image,3)

    arr_sigma = np.linspace(0.1, 3, 6)
    Image = frangi_2d(Image,arr_sigma)
    
    
    Image_array= Image.flatten()
    Image_array = np.sort(Image_array,axis=None)
    Tw = np.percentile(Image_array,Tw)
    Th = np.percentile(Image_array,Th)
    #  Tw_zip = zip(percentile_array,Tw)
    #  print(list(Tw_zip))

    Image = skifilter.apply_hysteresis_threshold(Image,Tw,Th)
    
    

    L,N = ndi.label(Image)
    T = ndi.sum(Image,L, range(1,N+1))
    comp2remove = np.nonzero(T<threshold)[0] + 1

    num_rows,num_cols = L.shape

    Image = np.copy(Image)

    for row in range(num_rows):
        for col in range(num_cols):
            label =  L[row,col]
            if label in comp2remove:
                Image[row,col] = 0

    
    
    
    for i in range(dilation):
        Image = morph.binary_dilation(Image)

    for i in range(dilation):
        Image = morph.binary_erosion(Image)

    #plot_image(Image,"Result4")
    

    return Image

Image_original = plt.imread("Imagens_originais/09_test.tif") 
#mask = plt.imread("../Imagens/21_training_mask.gif")
#mask = morph.binary_erosion(mask)


th = 95
tw = 90
elem = 10



Image_obtida = Calvo_segmentation(Image_original,th,tw,elem,220,2)
#Image_obtida = np.bitwise_and(Image_obtida,mask)
plot_image(Image_obtida,"01_test.tif")
