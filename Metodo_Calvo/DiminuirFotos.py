import tifffile as tiff
import numpy as np
import os

def Maximum_filter(img):
    rows,cols = img.shape
    img_maximum = np.zeros((rows//2,cols//2))
    for row in range(0,rows,2):
        for col in range(0,cols,2):
            pixel = np.amax([img[row+1,col],img[row,col+1],img[row,col],img[row+1,col+1]])
            img_maximum[row//2,col//2] = pixel

    return img_maximum


save_path = "Diminuidas/"
path = "UNET_Results/"
dir_results = os.listdir(path)

for image in dir_results:
    print("Opening",image)
    img = tiff.imread(path + image)
    img_maximum = Maximum_filter(img)
    tiff.imwrite((save_path + image),img_maximum)

