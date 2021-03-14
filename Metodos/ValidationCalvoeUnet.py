import counting_points as count
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import skimage.color as skicolor

def atualizar_points(img,img_menores,points):
    img = np.logical_xor(img,img_menores) 
    img = img.astype(np.uint8)
    points_removed = points.copy()

    plt.imsave("ResultXOR",img,format="pdf")

    for point in points:  
        if img[point] == 1:
            points_removed.remove(point)
                
    return points_removed


def drawpoints(img,img_name,bifurcations,cross_points):
    img_color = img[:,:,0:3].copy()
    img_color[:,:,:] = (0,0,0)
    img_color[bifurcations[:,0],bifurcations[:,1]] = (255,0,0)
    img_color[cross_points[:,0],cross_points[:,1]]=(0,255,0)
    '''
    plt.imsave("Result"+img_name,img_color,format="pdf")
    plt.imshow(img_color)
    plt.show()
'''

def Validation(img_name,bifurcations,crossover):

    path = 'Imagens_pontos/'
    #img = plt.imread(path + img_name.split('.')[0] + '.tiff')  
    img_menores = plt.imread(path + img_name.split('.')[0] + "pontosMenores" + ".tiff" )
    

    #img_bifur = (img[:,:,0] == 255).astype(np.uint8)
    img_menores_bifur = (img_menores[:,:,0] == 255).astype(np.uint8)
    #img_cross = ((img[:,:,0] > 120) & (img[:,:,0] < 132)).astype(np.uint8)
    img_menores_cross = ((img_menores[:,:,0] > 110) & (img_menores[:,:,0] < 140)).astype(np.uint8)

    
    #bifurcations = atualizar_points(img_bifur,img_menores_bifur,bifurcations)
    bifurcations = np.asarray(bifurcations)

    tp,fp,fn = count.count_agreements(img_menores_bifur,bifurcations)
    print(image_name, " tp:",tp," fp:",fp," fn:",fn," - Bifurcation")
    
    precision_bifur = tp/(tp+fp)
    recall_bifur = tp/(tp+fn)

    #crossover = atualizar_points(img_cross,img_menores_cross,crossover)
    crossover = np.asarray(crossover)

    tp,fp,fn = count.count_agreements(img_menores_cross,crossover)

    print(image_name, " tp:",tp," fp:",fp," fn:",fn," - Crossover")
    precision_cross = tp/(tp+fp)
    recall_cross = tp/(tp+fn)

    #drawpoints(img,"",bifurcations,crossover)
    return precision_bifur,recall_bifur,precision_cross,recall_cross

file = open('results_CalvoeUnet','rb')
results_dict = pickle.load(file)

Rc_list = results_dict['params']['Rc_list']
p_list = results_dict['params']['p_list']
Rb_list = results_dict['params']['Rb_list']
paths = results_dict['folder']
list_dir = results_dict['images']

metrics_dict = {}
metrics = {}
metrics['paths'] = paths
metrics['images'] = list_dir[0]  #Apenas para pegar o nome das imagens
#print(metrics)
for path,Images in zip(paths,list_dir):
    precisions_bifur = []
    precisions_cross = []
    recalls_bifur = []
    recalls_cross = []
    metrics[path] = {}
    for Rc in Rc_list:
        for p in p_list:
            for Rb in Rb_list:
                for image_name in Images:
                    print(path,"----",image_name)
                    bifurcations = results_dict['results'][path][(Rc,p,Rb)][image_name]['Bifurcations']
                    crossover = results_dict['results'][path][(Rc,p,Rb)][image_name]['crossover']
                    precision_bifur,recall_bifur,precision_cross,recall_cross = Validation(image_name,bifurcations,crossover)
                    precision_bifur = round(precision_bifur,4)
                    recall_bifur = round(recall_bifur,4)
                    precision_cross = round(precision_cross,4)
                    recall_cross = round(recall_cross,4)
                    metrics_dict[(Rc,p,Rb,image_name,path)] = {'precision_bifur':precision_bifur,'recall_bifur':recall_bifur,'precision_cross':precision_cross,'recall_cross':recall_cross}
                    print("pb:",precision_bifur," rb:",recall_bifur," pc:",precision_cross," rc:",recall_cross)
                    metrics[path][image_name] = [precision_bifur,recall_bifur,precision_cross,recall_cross]
                    '''
                    precisions_bifur.append(precision_bifur)
                    precisions_cross.append(precision_cross)
                    recalls_bifur.append(recall_bifur)
                    recalls_cross.append(recall_cross)
                    ''' 


    

file = open('metrics_textCalvoeUnet','wb')
pickle.dump(metrics_dict,file)
file.close()
print("Gravado metrics_dict")


file = open('metrics_CalvoCalvoeUnet','wb')
pickle.dump(metrics,file)
file.close()
print("Gravado metrics")
