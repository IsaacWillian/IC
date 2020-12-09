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




path = 'Imagens_gold/'
dir_pontos = os.listdir(path)
dir_pontos.sort()

Rc_list = [5,6,7,8] #[8,10,12,14,16]
p_list = [1,2,3,4,5] #[2,3,4,5,6]
Rb_list = [8,10,12,14,16] #[12,14,16,18,20,22]

results_dict = {'params':{'Rc_list':Rc_list,
                          'p_list':p_list,
                          'Rb_list':Rb_list},
                'results':{},
                'images':dir_pontos
               }
for Rc in Rc_list:
    for p in p_list:
        for Rb in Rb_list:
            for image_name in dir_pontos:
                    Image_original = plt.imread(path+image_name)
                    bifurcation_points,crossover = Metodo_Calvo(Image_original,Rc,p,Rb)
                    results_dict['results'][(Rc,p,Rb)] = {}                
                    results_dict['results'][(Rc,p,Rb)][image_name] = {'Bifurcations':bifurcation_points,'crossover':crossover}
                    print("Terminado -- ","Rc:",Rc," p:",p," Rb:",Rb, "Image:",image_name )

file = open('results_Calvo2','wb')
pickle.dump(results_dict,file)
file.close()
print("Dump - ",image_name)