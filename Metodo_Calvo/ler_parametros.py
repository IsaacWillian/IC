'''
1. Para cada imagem, varre todos os parâmetros e calcula o "precision" e o "recall" de cada 
combinação de parâmetros. Tendo os valores você pode plotar um gráfico de recall x precision para cada imagem.
2. Para cada combinação de parâmetros, calcula o "precision" e o "recall" médio para todas as imagens. 
Portanto, você pode também plotar um gráfico de recall x precision  
'''
import pickle
import matplotlib.pyplot as plt

file = open('parameters','rb')
results_dict = pickle.load(file)
file.close()

name_images = results_dict['images']
n_images = len(name_images)
dilation_params = results_dict['params']['dilation']
threshold_params = results_dict['params']['threshold']
elem_params = results_dict['params']['elem']

for index_image in range(n_images):
    Precisions = []
    Recalls = []
    for dilation in (dilation_params):
        for threshold in (threshold_params):
            for elem in (elem_params):        
                    dict_aux = results_dict['results'][index_image][(dilation, threshold, elem)] 
                    precision = dict_aux['tp']/(dict_aux['tp'] + dict_aux['fp'])
                    recall = dict_aux['tp']/(dict_aux['tp'] + dict_aux['fn'])
                    print("d:",dilation," e:",elem," t:",threshold,"Precision:",precision, " | Recall:",recall)
                    
                    Precisions.append(precision)
                    Recalls.append(recall)
    plt.plot(Precisions,Recalls,'o')
    plt.ylabel("Recall")
    plt.xlabel("Precision")
    plt.xlim((0,1))
    plt.title(name_images[index_image])
    plt.show()

    #2 10 220
    #2 10 220￼  
    #2 16 220
    #2 16 220  
    #2 10 160
    #2 10 220
    #2 12 190
    #2 10 220
    #2 14 220
    #2 12 220


