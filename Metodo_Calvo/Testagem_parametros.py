#True Negative == And(obtida_invert,Image_gold_invert)  Background e foi detectado como background
#False negative == And(obtida_invert,Image_gold)  É vaso e foi detectado como background
#Salvar sum(Image_gold)
#Salva sum(Image_obtida)
#True Positive == and_sum È vaso +e foi detectado como vaso
#False Positive == Sum(Image_fp) Background e foi detectado como vaso      
#  

#ERRO CALVO, COMEÇA EM TERMINAÇÂO E VAI ATÈ OUTRA TERMINAÇÂO, NÂO ACHA NEIGHBOR

import os
import matplotlib.pyplot as plt
from Calvo import Calvo_implementation
import pickle
import numpy as np


dilation_params = [2, 3, 4]
threshold_params = range(70, 280,30)
elem_params = [6,8,10, 12, 14, 16]


caminho_original = 'Imagens_originais/'
caminho_gold= 'Imagens_gold/'
dir_original = os.listdir(caminho_original)
dir_gold = os.listdir(caminho_gold)
tw = 90
th = 95
dir_original.sort()
dir_gold.sort()

results_dict = {'params':{'dilation':dilation_params,
                          'threshold':threshold_params,
                          'elem':elem_params},
                'results':{},
                'images':dir_original

               }


for index_image,(filename_original,filename_gold) in enumerate(zip(dir_original,dir_gold)):
    Image_original = plt.imread(caminho_original + filename_original)
    Image_gold = plt.imread(caminho_gold + filename_gold)
    print("Obtendo parâmetros de ",filename_original)
    results_dict['results'][index_image] = {}
    for dilation in (dilation_params):
        for threshold in (threshold_params):
            for elem in (elem_params):        
                    Image_obtida = Calvo_implementation(Image_original,tw,th,elem,threshold,dilation)
                    Image_tp = np.bitwise_and(Image_obtida,Image_gold)
                    Image_or = np.bitwise_or(Image_obtida,Image_gold)
                    Image_or = np.bitwise_and(Image_or,Image_or)
                    Image_gold_invert = np.logical_not(Image_gold)
                    Image_obtida_invert = np.logical_not(Image_obtida)
                    Image_fp = np.logical_and(Image_gold_invert,Image_obtida)
                    Image_tn = np.logical_and(Image_obtida_invert,Image_gold_invert)
                    Image_fn = np.logical_and(Image_gold,Image_obtida_invert)

                    tp = np.sum(Image_tp) 
                    tn = np.sum(Image_tn) 
                    fp = np.sum(Image_fp)
                    fn = np.sum(Image_fn)
                    sum_obtida = np.sum(Image_obtida)
                    sum_gold = np.sum(Image_gold)
                    results_dict['results'][index_image][(dilation, threshold, elem)] = {
                    'tp':tp,
                    'tn':tn,
                    'fp':fp,
                    'fn':fn,
                    'sum_gold':sum_gold,
                    'sum_obtida':sum_obtida
                }
                    print("d:",dilation,"t:",threshold,"e:",elem," : ","tp: ",tp,"tn:",tn,"fp:",fp,"fn:",fn,"sum_gold:",sum_gold,"sum_obtida:",sum_obtida )


file = open('parameters2','wb')
pickle.dump(results_dict,file)
file.close()

'''[index_image]
for dilation in results_dict['params']['dilation']:
  for threshold in results_dict['params']['threshold']:
      '''