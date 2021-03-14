import os
import matplotlib.pyplot as plt
from Calvo import Calvo_implementation,FillHolesinGold,middle_point_crossover
import pickle
import numpy as np

def Metodo_Calvo(Image,Rc,p,Rb):
    Image = FillHolesinGold(Image,3)
    bifurcation_points,crossovers_in_pairs = Calvo_implementation(Image,Rc,p,Rb)
    crossover = middle_point_crossover(crossovers_in_pairs)

    return bifurcation_points,crossover




pathGold = 'Imagens_gold/'
pathUnet = 'Diminuidas/'
pathCalvo = 'Calvo_results/'
dirGold = os.listdir(pathGold)
dirUnet = os.listdir(pathUnet)
dirCalvo = os.listdir(pathCalvo)
dirGold.sort()
dirUnet.sort()
dirCalvo.sort()
#list_dir = [dirGold,dirUnet,dirCalvo]
list_dir = [dirUnet]

#paths = [pathGold[:-1],pathUnet[:-1],pathCalvo[:-1]]
paths = [pathUnet[:-1]]

#Best combination [8,5,16]

Rc_list = [8] #[5,6,7,8,10,12]
p_list = [5] #[1,2,3,4,5]
Rb_list = [16] #[8,10,12,14,16]

results_dict = {'params':{'Rc_list':Rc_list,
                          'p_list':p_list,
                          'Rb_list':Rb_list},
                'results':{},
                'images':list_dir,
                'folder':paths
               }

for path,dirs in zip(paths,list_dir):
    results_dict['results'][path] = {}
    for Rc in Rc_list:
        for p in p_list:
            for Rb in Rb_list:
                results_dict['results'][path][(Rc,p,Rb)] = {}   
                for image_name in dirs:
                        Image_original = plt.imread(path+'/'+image_name)
                        bifurcation_points,crossover = Metodo_Calvo(Image_original,Rc,p,Rb)                               
                        results_dict['results'][path][(Rc,p,Rb)][image_name] = {'Bifurcations':bifurcation_points,'crossover':crossover}
                        print("Terminado -- ","Rc:",Rc," p:",p," Rb:",Rb, "Image:",image_name,"Path:",path )

file = open('results_CalvoeUnet','wb')
pickle.dump(results_dict,file)
file.close()
print("Dump - ",paths)