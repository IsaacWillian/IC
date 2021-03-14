import pickle
import matplotlib.pyplot as plt

file = open('metrics_textCalvoeUnet','rb')
metrics = pickle.load(file)
#print(metrics)

file = open('metrics_CalvoCalvoeUnet','rb')
metrics = pickle.load(file)
print(metrics["Imagens_gold"].keys())
#print(metrics)
paths = metrics['paths']
images = metrics['images']
figs = ['o','d','*']
for image in images:
    for path,f in zip(paths,figs):
        try:
            plt.plot(metrics[path][image][0],metrics[path][image][1],f+'b')
            plt.plot(metrics[path][image][2],metrics[path][image][3],f+'r')
        except:
            image = image.split('.')[0] + ".tif"
            plt.plot(metrics[path][image][0],metrics[path][image][1],f+'b')
            plt.plot(metrics[path][image][2],metrics[path][image][3],f+'r')

    plt.ylabel("Recall")
    plt.xlabel("Precision")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title("Precision x Recall " + image)
    plt.show()