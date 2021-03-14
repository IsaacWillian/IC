import numpy as np
import matplotlib.pyplot as plt


'''
truePositive = and_sum/np.sum(Image_gold)
falsePositive = np.sum(Image_exseg)/np.sum(Image_obtida)  # False Positive == Sum(Image_exseg) 
'''

def obterAccuracyAndIoU(Image_obtida, Image_gold):
    
    Image_and = np.bitwise_and(Image_obtida,Image_gold)
    Image_or = np.bitwise_or(Image_obtida,Image_gold)
    Image_or = np.bitwise_and(Image_or,Image_or) #For transform in binary image
    Image_gold_invert = np.logical_not(Image_gold) 
    obtida_invert = np.logical_not(Image_obtida) 
    Image_exseg = np.logical_and(Image_gold_invert,Image_obtida)
    trueNegativeImg = np.bitwise_and(obtida_invert,Image_gold_invert)  #Background e foi detectado como background
    falseNegativeImg = np.bitwise_and(obtida_invert,Image_gold) 

    trueNegative = np.sum(trueNegativeImg) #Background e foi detectado como background
    falseNegative = np.sum(falseNegativeImg)  #É vaso e foi detectado como background
    truePositive = np.sum(Image_and)  #È vaso e foi detectado como vaso
    falsePositive = np.sum(Image_exseg) #Background e foi detectado como vaso

    I_U = np.sum(Image_and)/np.sum(Image_or)
    accuracy = (truePositive + trueNegative)/(trueNegative + truePositive + falseNegative + falsePositive)

    print("TP =", truePositive," TN = ",trueNegative, "FN = ", falseNegative," FP = ",falsePositive)


    return accuracy,I_U


Image = plt.imread("Calvo_results/01_manual1.tif")
Image_gold = plt.imread("Imagens_gold/01_manual1.gif") 

accuracy,IoU = obterAccuracyAndIoU(Image,Image_gold)

print("Accuracy = ",accuracy," IoU = ",IoU)





