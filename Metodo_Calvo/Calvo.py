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

class Branch:
    def __init__(self,initial_point,path,path_len):
        self.initial_point = initial_point
        self.path = path
        self.path_len = path_len

def SearchNeighbor(img,point,previous_point):
    x_add = [0,1,0,-1]
    y_add = [-1,0,1,0]
    for x,y in zip(x_add,y_add):
        neighbor = (point[0]+x,point[1]+y)
        if img[neighbor] != 0 and ((neighbor) != previous_point):
            return (point[0]+x,point[1]+y)

def ShowBranchs(img,path):
    img_rgb = skicolor.gray2rgb(img)
    for x,y in path:
        img_rgb[x,y] = [255,0,0]    

    plt.imshow(img_rgb)
    plt.show()

# Separar dois grupos (terminações e bifurcaçoes) Checked
# Separar caminhos que terminam TermTerm, BiTerm, BiBi 
def tracking_and_remove_branch(img):
    bifurcation_points = []
    end_points = []
    row_max,col_max = img.shape
    for row in range(row_max):
        for col in range(col_max):
            if img[row,col] >= 150:
                bifurcation_points.append((row,col))
            elif img[row,col] == 50:
                end_points.append((row,col))
    
    for end_point in end_points:
        path = [end_point]
        branchs = []
        path_len = 1
        fim = 0
        previous_point = (0,0)
        while fim != 1:
            neighbor = SearchNeighbor(img,end_point,previous_point)
            path_len +=1
            path.append((neighbor))
            if neighbor in end_points:
                fim = 1
            elif neighbor in bifurcation_points:
                branch = Branch(end_point,path,path_len)
                branchs.append(branch)
                fim = 1
                if path_len < 10: ## Threshold
                    bifurcation = path.pop(-1)
                    img[bifurcation] = img[bifurcation] - 50
                    for x,y in path:
                        img[x,y] = 0
            else: 
                previous_point = end_point
                end_point = neighbor

        
    return img

def local_analysis_classification(img,radius,p):
    R1 = radius - p
    R2 = radius
    R3 = radius + p

    # Aplicar formulas passando por todos os pixeis acima de 150, verificando se são B ou F
    # for
        # for
            #funçao 

    pass

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

    Image = np.copy(Image)

    for row in range(num_rows):
        for col in range(num_cols):
            label =  L[row,col]
            if label in comp2remove:
                Image[row,col] = 0


    # Image = ndi.median_filter(Image,3)
    for i in range(4):
        Image = morph.binary_dilation(Image)

    for i in range(4):
        Image = morph.binary_erosion(Image)
    

    Image = Image.astype(np.int16)
    Image = stentiford(Image)

    

    #Percorre os vizinhos fazendo a somatória ds seu valores
    Image_sum = Image.copy()
    row_max, col_max = Image.shape
    for row in range(row_max):
        for col in range(col_max): 
            if Image[row,col] == 1:
                sum = 0
                x_ant = x_add[-1]
                y_ant = y_add[-1]
                for x,y in zip(x_add,y_add):
                    sum += abs(Image[row + x_ant, col+y_ant] - Image[row+x,col+y])
                    x_ant = x
                    y_ant = y

                sum *= 0.5
                Image_sum[row,col] = int(sum*50)
                
                   

    tracking_and_remove_branch(Image_sum)

    plt.imshow(Image_sum,"gray")
    plt.show()
    return Image


Image_original = plt.imread("../Imagens/21_training.tif") 
mask = plt.imread("../Imagens/21_training_mask.gif")
mask = morph.binary_erosion(mask)


Image_obtida = Calvo_implementation(Image_original,95,90)
# Image_obtida = np.bitwise_and(Image_obtida,mask)

plt.imshow(Image_obtida, 'gray')
plt.show()