#Algoritmo retirado do github https://github.com/ShreyaPrabhu/Thinning-Algorithms.
#Modificado para permanecer apenas os passos desejados para a implementação do método. 

from skimage import color
import numpy as np
from PIL import Image
from skimage.util import invert
import matplotlib.pyplot as plt
import time
import os
import glob

def zeroToOne(thin_image,i,j):
    p2 = thin_image[i-1][j-1]
    p3 = thin_image[i-1][j]
    p4 = thin_image[i-1][j+1]
    p5 = thin_image[i][j+1]
    p6 = thin_image[i+1][j+1]
    p7 = thin_image[i+1][j]
    p8 = thin_image[i+1][j-1]
    p9 = thin_image[i][j-1]
    count = 0;
    endpoint = 0;
    if(p2+p3+p4+p5+p6+p7+p8+p9 == 1):
        endpoint = 1
    if(p2==0 and p3==1):
        count = count + 1
    if(p3==0 and p4==1):
        count = count + 1
    if(p4==0 and p5==1):
        count = count + 1
    if(p5==0 and p6==1):
        count = count + 1
    if(p6==0 and p7==1):
        count = count + 1
    if(p7==0 and p8==1):
        count = count + 1
    if(p8==0 and p9==1):
        count = count + 1
    if(p9==0 and p2==1):
        count = count + 1
    return count,endpoint


def stentiford(image):
        # Make copy of the image so that original image is not lost
    thin_image = image.copy()
    row, col = thin_image.shape
    check = 2
    template = 1
    outImage = 1
    # Perform iterations as long as there are pixels marked for deletion
    iteration = 0
    total_changes = 0
    while(outImage):
        # Make outImage empty
        outImage = []
        changes = 0
        iteration = iteration + 1 
        # Loop through the pixels of the thin_image
        for i in range(2, row-1):
            for j in range(2, col-1):
                # Proceed only if pixel in consideration is Black(1)
                if(thin_image[i][j]==1):
                    p0 = thin_image[i][j]
                    p1 = thin_image[i-1][j]
                    p2 = thin_image[i][j+1]
                    p3 = thin_image[i+1][j]
                    p4 = thin_image[i][j-1]
                    if(template==1):
                        template_match = (p1==0 and p3==1)
                    if(template==2):
                        template_match = (p2==1 and p4==0)
                    if(template==3):
                        template_match = (p1==1 and p3==0)
                    if(template==4):
                        template_match = (p2==0 and p4==1)
                    connectivity, isEndpoint = zeroToOne(thin_image,i,j)
                    if(template_match==1):
                        if(connectivity == 1):
                            if(isEndpoint== 0):
                                outImage.append((i,j))
        

        # Delete the pixels marked for deletion
        for i,j in outImage:
            thin_image[i][j] = 0
            changes = changes + 1
        template = template+1
        if(template==5):
            template = 1
        total_changes = total_changes + changes

    print("total_changes: ", total_changes)
    return thin_image

