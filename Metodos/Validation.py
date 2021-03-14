import counting_points as count
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import skimage.color as skicolor

def drawpoints(img,img_name,bifurcations,cross_points):
    img_color = img[:,:,0:3].copy()
    img_color[bifurcations[:,0],bifurcations[:,1]] = (255,0,0)
    img_color[cross_points[:,0],cross_points[:,1]]=(0,255,0)
    #plt.imsave("Result"+img_name,img_color,format="pdf")
    plt.imshow(img_color)
    plt.show()

def Validation(img_name,bifurcations,crossover):

    path = 'Imagens_pontos/'
    img = plt.imread(path + img_name.split('.')[0] + '.tiff')  
    img_bifur = (img[:,:,0] == 255).astype(np.uint8)
    img_cross = ((img[:,:,0] > 120) & (img[:,:,0] < 132)).astype(np.uint8)

    bifurcations = np.asarray(bifurcations)
    crossover = np.asarray(crossover)
    #drawpoints(img,img_name,bifurcations,crossover)

    tp,fp,fn = count.count_agreements(img_bifur,bifurcations)
    
    precision_bifur = tp/(tp+fp)
    recall_bifur = tp/(tp+fn)

    tp,fp,fn = count.count_agreements(img_cross,crossover)

    precision_cross = tp/(tp+fp)
    recall_cross = tp/(tp+fn)

    return precision_bifur,recall_bifur,precision_cross,recall_cross

file = open('results_CalvoTeste','rb')
results_dict = pickle.load(file)

Rc_list = results_dict['params']['Rc_list']
p_list = results_dict['params']['p_list']
Rb_list = results_dict['params']['Rb_list']

Images = results_dict['images']

precisions_bifur = []
precisions_cross = []
recalls_bifur = []
recalls_cross = []
metrics_dict = {}

for Rc in Rc_list:
    for p in p_list:
        for Rb in Rb_list:
            for image_name in Images:
                bifurcations = results_dict['results'][(Rc,p,Rb)][image_name]['Bifurcations']
                crossover = results_dict['results'][(Rc,p,Rb)][image_name]['crossover']
                precision_bifur,recall_bifur,precision_cross,recall_cross = Validation(image_name,bifurcations,crossover)
                precision_bifur = round(precision_bifur,4)
                recall_bifur = round(recall_bifur,4)
                precision_cross = round(precision_cross,4)
                recall_cross = round(recall_cross,4)
                metrics_dict[(Rc,p,Rb,image_name)] = {'precision_bifur':precision_bifur,'recall_bifur':recall_bifur,'precision_cross':precision_cross,'recall_cross':recall_cross}
                print("(",Rc,p,Rb,") -- pb:",precision_bifur," rb:",recall_bifur," pc:",precision_cross," rc:",recall_cross)
                precisions_bifur.append(precision_bifur)
                precisions_cross.append(precision_cross)
                recalls_bifur.append(recall_bifur)
                recalls_cross.append(recall_cross)



metrics = [precisions_bifur,recalls_bifur,precisions_cross,recalls_cross]

file = open('metrics_text2','wb')
pickle.dump(metrics_dict,file)
file.close()
print("Gravado metrics_dict")


file = open('metrics_Calvo2','wb')
pickle.dump(metrics,file)
file.close()
print("Gravado metrics")
