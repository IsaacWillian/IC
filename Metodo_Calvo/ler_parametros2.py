'''
1. Para cada imagem, varre todos os parâmetros e calcula o "precision" e o "recall" de cada 
combinação de parâmetros. Tendo os valores você pode plotar um gráfico de recall x precision para cada imagem.
2. Para cada combinação de parâmetros, calcula o "precision" e o "recall" médio para todas as imagens. 
Portanto, você pode também plotar um gráfico de recall x precision.  
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt

file = open('parameters','rb')
results_dict = pickle.load(file)
file.close()

name_images = results_dict['images']
n_images = len(name_images)
dilation_params = results_dict['params']['dilation']
threshold_params = results_dict['params']['threshold']
elem_params = results_dict['params']['elem']

medias_precision = []
medias_recall = []
    
for dilation in (dilation_params):
    for threshold in (threshold_params):
        for elem in (elem_params): 
            Precisions = []
            Recalls = []     
            for index_image in range(n_images):  
                dict_aux = results_dict['results'][index_image][(dilation, threshold, elem)] 
                precision = dict_aux['tp']/(dict_aux['tp'] + dict_aux['fp'])
                recall = dict_aux['tp']/(dict_aux['tp'] + dict_aux['fn'])
                Precisions.append(precision)
                Recalls.append(recall)
            media_precision = sum(Precisions)/len(Precisions)
            media_recall = sum(Recalls)/len(Recalls)
            medias_precision.append(media_precision)
            medias_recall.append(media_recall)

            print("d:",dilation," e:",elem," t:",threshold,"Precision:",media_precision, " | Recall:",media_recall)

print(np.max(medias_precision),np.max(medias_recall))
plt.plot(medias_precision,medias_recall,'o')
plt.ylabel("Recall")
plt.xlabel("Precision")
plt.xlim((0.66,0.82))
plt.ylim((0.66,0.82))
plt.title("Precision x Recall médio de cada imagem")
plt.savefig("Precision.x.Recall",bbox_inches='tight',format='pdf')
plt.show()
