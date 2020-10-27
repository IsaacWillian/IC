import counting_points as count
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import skimage.color as skicolor

def drawpoints(img,img_name,bifur_points,cross_points):
    img_color = img[:,:,0:3].copy()
    img_color[bifur_points[:,0],bifur_points[:,1]] = (255,0,0)
    img_color[cross_points[:,0],cross_points[:,1]]=(0,255,0)
    plt.imsave("Result"+img_name,img_color,format="pdf")
    #plt.imshow(img_color)
    #plt.show()

precision_bifur = []
recall_bifur = []
precision_cross = []
recall_cross = []

path = 'Imagens_pontos/'
path_results= 'Results/'
dir_results = os.listdir(path_results)
dir_pontos = os.listdir(path)
dir_results.sort()
dir_pontos.sort()

for result,img_pontos in zip(dir_results,dir_pontos):
    file = open(path_results + result,'rb')
    results_dict = pickle.load(file)
    file.close()
    img = plt.imread(path + img_pontos)

    img_bifur = (img[:,:,0] == 255).astype(np.uint8)
    img_cross = ((img[:,:,0] > 120) & (img[:,:,0] < 132)).astype(np.uint8)
    bifur_points = results_dict['Bifurcations']
    cross_points = results_dict['Crossover']

    bifur_points = np.asarray(bifur_points)
    cross_points = np.asarray(cross_points)

    drawpoints(img,img_pontos,bifur_points,cross_points)

    
    tp,fp,fn = count.count_agreements(img_bifur,bifur_points)
    print(img_pontos,result,"- Bifurcation - tp:",tp," fp:",fp," fn:",fn)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    precision_bifur.append(precision)
    recall_bifur.append(recall)


    tp,fp,fn = count.count_agreements(img_cross,cross_points)
    print(img_pontos,result,"- Crossover - tp:",tp," fp:",fp," fn:",fn)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    precision_cross.append(precision)
    recall_cross.append(recall)

plt.plot(precision_bifur,recall_bifur,'or')
plt.plot(precision_cross,recall_cross,'og')
plt.ylabel("Recall")
plt.xlabel("Precision")
plt.ylim((0,1))
plt.xlim((0,1))
plt.savefig("Precision.x.Recall - Detection",bbox_inches='tight',format='pdf')
plt.show()


