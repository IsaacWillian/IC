import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
from stentiford import stentiford
import os

def Maximum_filter(img):
    rows,cols = img.shape
    img_maximum = np.zeros((rows//2,cols//2), dtype=np.uint8)
    for row in range(0,rows,2):
        for col in range(0,cols,2):
            pixel = np.amax([img[row+1,col],img[row,col+1],img[row,col],img[row+1,col+1]])
            img_maximum[row//2,col//2] = pixel

  
    return img_maximum


save_path = "Imagens_gold/"
path = "Imagens_gold/"
dir_results = os.listdir(path)

for image in dir_results:
    print("Opening",image)
    img = plt.imread(path + image)
    img = img//255
    #img = tiff.imread(path + image)
    img_maximum = stentiford(img)
    plt.imsave((save_path + "Esq" + image),img_maximum,cmap="gray")
    #tiff.imwrite((save_path + "Esq" + image),img_maximum)

