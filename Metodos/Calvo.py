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
import counting_points as count
import os
from PIL import Image  

def plot_image(img,title="1"):
    plt.imshow(img,"gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    #plt.imsave(title,img.astype(np.uint8),cmap="gray",format='pdf')
    
def FillHolesinGold(Image,threshold):
    Image = abs((Image/255)-1)
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
                
    
    Image = abs((Image-1)*255)
    return Image


class Branch:
    def __init__(self,initial_point,path,path_len):
        self.initial_point = initial_point
        self.path = path
        self.path_len = path_len

def SearchNeighbor(img,point,path):
    x_add = [ 0,1,0,-1] 
    y_add = [-1,0,1, 0] 
    x_add_diag = [1,1,-1,-1]
    y_add_diag = [-1,1,1,-1]
    neighbors = []
    for x,y in zip(x_add,y_add):
        _x = point[0]+x
        _y = point[1]+y
        neighbor = (_x,_y) 
        if img[neighbor] != 0 and (neighbor not in path):
            neighbors.append(neighbor) 
    
    if neighbors == []:
        for x,y in zip(x_add_diag,y_add_diag):
            _x = point[0]+x
            _y = point[1]+y
            neighbor = (_x,_y) 
            if img[neighbor] != 0 and (neighbor not in path):
                neighbors.append(neighbor) 

    
    return neighbors

def ShowBranchs(img,path):
    img_rgb = skicolor.gray2rgb(img)
    for x,y in path:
        img_rgb[x,y] = [255,0,0]    

    #plot_image(img_rgb,"branch")

def get_features_points(img):
    bifurcation_points = []
    end_points = []
    row_max,col_max = img.shape
    for row in range(row_max):
        for col in range(col_max):
            if img[row,col] >= 150:
                bifurcation_points.append((row,col))
            if img[row,col] == 50:
                end_points.append((row,col))

    
    return bifurcation_points,end_points

def tracking_and_remove_branchs(img):
    bifurcations,ends = get_features_points(img)
    
    for end_point in ends:
        point=end_point
        fim=0
        path = [point]
        path_len = 1
        while fim != 1:
                neighbor = SearchNeighbor(img,point,path)[0]
                path_len +=1
                path.append(neighbor)
                if (neighbor in bifurcations) or (neighbor in ends):
                    fim = 1
                    branch = Branch(end_point,path,path_len)
                    img= remove_branch(img,branch)                                            
                        
                else: 
                    previous_point = point
                    point = neighbor
    
    return img

def Add_graph(graph,map_pos_to_index,feature_point,neighbor):
    index_node_1 = map_pos_to_index[feature_point]
    index_node_2 = map_pos_to_index[neighbor]
    if index_node_2 not in graph[index_node_1]: 
        graph[index_node_1].append(index_node_2)
        graph[index_node_2].append(index_node_1)

def create_graph(img):
    bifurcations,ends = get_features_points(img)
    features_points = bifurcations + ends
    node_index = 0
    map_pos_to_index = {}
    for feature_point in features_points:
        map_pos_to_index[feature_point] = node_index
        node_index += 1 

    
    graph = [ [] for i in range (len(map_pos_to_index))]
    for feature_point in features_points:
        fim=0
        path=[]
        neighbors = SearchNeighbor(img,feature_point,path)
        for point in neighbors:
            previous_point = feature_point
            init_point = point
            path=[feature_point,init_point]
            fim = 0
            if point in features_points:
                fim = 1
                Add_graph(graph,map_pos_to_index,feature_point,init_point)
            while fim != 1:
                neighbor = SearchNeighbor(img,point,path)[0]
                path.append(neighbor)
                if neighbor in features_points:
                    fim = 1
                    Add_graph(graph,map_pos_to_index,feature_point,neighbor)                        
                else: 
                    previous_point = point
                    point = neighbor
    
    
    return img,graph,map_pos_to_index


def remove_branch(img,branch):
    if branch.path_len < 10: ## Threshold
        bifurcation = branch.path.pop(-1)
        end_point = branch.path[0] 
        #Apenas é removido se o track começa no ponto de terminação e chega em bifurcação
        if img[bifurcation]>=150 and img[end_point]==50: 
            img[bifurcation] = img[bifurcation] - 50
            for x,y in branch.path:
                img[x,y] = 0
              
    return img

def local_analysis_classification(img,radius,p):
    crossover_points = []
    bifurcation_points = []
    img_bin = img >= 50 #Para binarizar a imagem
    img_bin = img_bin.astype(int)
    R1 = radius - p
    R2 = radius
    R3 = radius + p

    img_color = skicolor.gray2rgb(img)

    row_max,col_max = img.shape
    for row in range(row_max):
        for col in range(col_max):
            if img[row,col] >= 150: 
                point = (row,col)  #bifurcações ou crossover 

                C = 2 * Cros(img_bin,point,R1) + Cros(img_bin,point,R2) + Cros(img_bin,point,R3)
                B = Bifur(img_bin,point,R1) + Bifur(img_bin,point,R2) + Bifur(img_bin,point,R3) * 2

                if C > B:
                    img[row,col] = 200
                    crossover_points.append((row,col))
                else:
                    bifurcation_points.append((row,col))
                    img[row,col] = 150   
    
                img_color = DrawRadius(img_color,point,[R1,R2,R3])

    #ShowPoints(img)
    #plot_image(img_color,"radius")
    
    return img,crossover_points,bifurcation_points

def DrawRadius(img,point,radius):
    for r in radius:
        rr,cc = skidraw.circle_perimeter(point[0],point[1],r)
        img[rr,cc] = (255,0,255)
    
    return img

def count_transitions(point,raio,img):
    img_color = skicolor.gray2rgb(img)
    img_color = img_color * 255
    rr,cc = skidraw.circle_perimeter(point[0],point[1],raio)
    rr,cc,_ = ordenar_perimetro(rr,cc,point[0],point[1])
    soma = 0
    rr_ant = rr[-1]
    cc_ant = cc[-1]
    
       
    for row,col in zip(rr,cc):
        result = abs(img[rr_ant,cc_ant] - img[row,col])
        if result == 1:
            img_color[row,col] = (255,255,0)
        soma += result
        rr_ant = row
        cc_ant = col

    if point == (445,361):
       
        '''plt.imshow(img_color)
        plt.show()
        plt.imshow(img,'gray')
        plt.show()'''
    
    
    return soma//2

def Cros(img,point,R):
    result = count_transitions(point,R,img)
    if result >= 4: #VERIFICAR PORQUE != e NÂO return 1 se >3
        return 1
    else: 
        return 0
    

def Bifur(img,point,R):
    result = count_transitions(point,R,img)
    if result == 3 or result == 2: 
        return 1
    else:
        return 0   
   

def local_analysis(img,point,R):
    row,col = point
    rr,cc = skidraw.circle_perimeter(row,col,R)
    result = np.sum(img[rr,cc])
    if result == 3: 
        return 1
    elif result == 4:
        return 0

def topological_analysis(img,NotClassified_points,crossover_points,radius,graph,map_pos_to_index):
    crossover_in_pairs,NotClassified_points = find_pairs(NotClassified_points,crossover_points,radius)
    #ShowPairs(img,crossover_in_pairs)
    crossover_in_pairs,NotClassified_points = pairs_is_connected(NotClassified_points,crossover_in_pairs,graph,map_pos_to_index)
    #ShowPairs(img,crossover_in_pairs)

    return img,NotClassified_points,crossover_in_pairs

def pairs_is_connected(NotClassified_points,pairs,graph,map_pos_to_index):
    for pair in pairs:
        index_node_1 = map_pos_to_index[pair[0]]
        index_node_2 = map_pos_to_index[pair[1]]

        if index_node_1 not in graph[index_node_2] or index_node_2 not in graph[index_node_1]:
            pairs.remove(pair)
            add = [(pair[0]),(pair[1])]
            NotClassified_points = NotClassified_points + add  #Se nao estão conectador são duas bifurcações
        
    return pairs,NotClassified_points

def ShowPairs(img,pairs):
    img_rgb = img.copy()
    img_rgb = skicolor.gray2rgb(img)
    for pair in pairs:
        m = ((pair[1][0]+pair[0][0])//2, (pair[1][1]+pair[0][1])//2) 
        rr,cc = skidraw.circle_perimeter(m[0],m[1],10)
        img_rgb[rr,cc] = [0,255,0]
        rr,cc = skidraw.line(m[0],m[1],pair[0][0],pair[0][1])
        img_rgb[rr,cc] = [0,0,255]
        rr,cc = skidraw.line(m[0],m[1],pair[1][0],pair[1][1])
        img_rgb[rr,cc] = [255,0,0]
    
    #plot_image(img_rgb,"Pares de crossovers")

def find_pairs(NotClassified_points,crossover_points,radius):
    pairs = []
    no_more_pairs = False
    while (no_more_pairs == False):
        d_smaller = radius * 2
        no_more_pairs = True
        for point in crossover_points:
            for point_aux in crossover_points:
                if point != point_aux :
                    d = np.sqrt((point_aux[0]-point[0])**2 + (point_aux[1]-point[1])**2 )
                    if d < d_smaller:
                        no_more_pairs = False
                        d_smaller = d
                        pair = ((point,point_aux))
        if no_more_pairs == False:
            crossover_points.remove(pair[0])
            crossover_points.remove(pair[1])
            pairs.append(pair)

    NotClassified_points = NotClassified_points + crossover_points #Pontos que não entram na classificação de crossover vão para bifurcações
    crossover_points = pairs

    return crossover_points,NotClassified_points



def ShowPoints(img):  # e perimetros  print
    img_rgb = img.copy()
    img_rgb = skicolor.gray2rgb(img)
    row_max,col_max = img.shape
    for row in range(row_max):
        for col in range(col_max):
            if img[row,col] == 150:
                rr,cc = skidraw.circle_perimeter(row,col,5)
                img_rgb[rr,cc] = [255,0,0]
            elif img[row,col] == 200:
                rr,cc = skidraw.circle_perimeter(row,col,5)
                img_rgb[rr,cc] = [0,255,0]

    #plot_image(img_rgb,"Bifurcações(Vermelhos) e Crossover(Verdes)")


def PointsNotClassified(img,NotClassified_points,bifurcation_points,Rb):
    
    pairs = []
    no_more_pairs = False
    while (no_more_pairs == False):
        d_smaller = Rb * 2
        no_more_pairs = True
        for point in NotClassified_points:
            for point_aux in NotClassified_points:
                if point != point_aux :
                    d = np.sqrt((point_aux[0]-point[0])**2 + (point_aux[1]-point[1])**2 )
                    if d < d_smaller:
                        no_more_pairs = False
                        d_smaller = d
                        pair = ((point,point_aux))
        if no_more_pairs == False:
            NotClassified_points.remove(pair[0])
            NotClassified_points.remove(pair[1])
            add = [pair[0],pair[1]]
            pairs = pairs + add

    bifurcation_points = bifurcation_points + NotClassified_points #Pontos que não entram na classificação de crossover vão para bifurcações
    NotClassified_points = pairs
    

    return bifurcation_points,NotClassified_points




def ShowResult(img,bifurcations,crossovers_in_pairs,NotClassified_points,Rc):
    img = img.copy()
    img = skicolor.gray2rgb(img)
    for bifur in bifurcations:
        rr,cc = skidraw.circle_perimeter(bifur[0],bifur[1],5)
        img[rr,cc] = (255,0,0)

    for cross in crossovers_in_pairs:
        rr,cc = skidraw.circle_perimeter(((cross[0][0]+cross[1][0])//2),((cross[0][1]+cross[1][1])//2),Rc)
        img[rr,cc] = (0,255,0)

    for NotClassified in NotClassified_points :
        rr,cc = skidraw.circle_perimeter(NotClassified[0],NotClassified[1],5)
        img[rr,cc] = (0,0,255)

    #plot_image(img,"Result20")   

def remove_pad(padding,bifurcation_points,crossover_in_pairs):
    bifurcation_points = [(bifur[0]-padding,bifur[1]-padding) for bifur in bifurcation_points]
    crossover_in_pairs = [((cross[0][0]-padding,cross[0][1]-padding),(cross[1][0]-padding,cross[1][1]-padding)) 
                            for cross in crossover_in_pairs]

    return bifurcation_points,crossover_in_pairs

'''Implementação do método sugerido por Calvo em Automatic detection and characterisation of retinal vessel
tree bifurcations and crossovers in eye fundus images'''
def Calvo_implementation(img,Rc,p,Rb):
   
    padding = Rb//2
    x_add = [-1,0,1,1,1,0,-1,-1,-1]# Começa com a ultima posição pois se faz atual-anterior 
    y_add = [-1,-1,-1,0,1,1,1,0,-1]
    NotClassified_points = []
    
    img = np.pad(img,padding,mode="edge")

    img = img.astype(np.int16)
    img = img//255
    #plot_image(img,"1")
    img = stentiford(img)
   
    
    #plot_image(img,"2")



    #Percorre os vizinhos fazendo a somatória ds seu valores
    Image_sum = img.copy()
    row_max, col_max = img.shape
    for row in range(row_max):
        for col in range(col_max): 
            if img[row,col] == 1:                  
                sum = 0
                x_ant = x_add[-1]
                y_ant = y_add[-1]
                for x,y in zip(x_add,y_add):
                    sum += abs(img[row + x_ant, col+y_ant] - img[row+x,col+y])
                    x_ant = x
                    y_ant = y

                sum *= 0.5
                Image_sum[row,col] = int(sum*50)
                
    

    
    img = tracking_and_remove_branchs(Image_sum)
    img,graph,map_pos_to_index = create_graph(img)
    img,crossover_points,bifurcation_points = local_analysis_classification(img,Rc,p)
    img,NotClassified_points,crossover_in_pairs = topological_analysis(img,NotClassified_points,crossover_points,Rc,graph,map_pos_to_index)
    #ShowResult(img,bifurcation_points,crossover_in_pairs,NotClassified_points,Rc)
    bifurcation_points,NotClassified_points = PointsNotClassified(img,NotClassified_points,bifurcation_points,Rb)
    ShowResult(img,bifurcation_points,crossover_in_pairs,NotClassified_points,Rc)

    bifurcation_points,crossover_in_pairs = remove_pad(padding,bifurcation_points,crossover_in_pairs)

    return bifurcation_points,crossover_in_pairs

def middle_point_crossover(crossovers_in_pairs):
    crossover = []
    for cross in crossovers_in_pairs:
        center_x = (cross[0][0] + cross[1][0])//2
        center_y = (cross[0][1] + cross[1][1])//2
        crossover.append((center_x,center_y))

   
    return crossover    
'''
Image_original = plt.imread("../Imagens/01_manual1.gif") 
bifurcation_points,crossovers_in_pairs = Calvo_implementation(Image_original)
'''

# path_results= 'Results/'
# path = 'Imagens_gold/'
# dir_pontos = os.listdir(path)
# dir_pontos.sort()

# Rc = 8
# p = 3
# Rb = 16


# for image_name in dir_pontos:
#     print(image_name)
#     Image_original = plt.imread(path+image_name)
#     Image_original = FillHolesinGold(Image_original,3)
#     bifurcation_points,crossovers_in_pairs = Calvo_implementation(Image_original,Rc,p,Rb)
#     crossover = middle_point_crossover(crossovers_in_pairs)

#     Result_dict = {'Bifurcations':bifurcation_points,'Crossover':crossover}
#     file = open(path_results+image_name.split(".")[0],'wb')
#     pickle.dump(Result_dict,file)
#     file.close()
#     print("Dump - ",image_name)


