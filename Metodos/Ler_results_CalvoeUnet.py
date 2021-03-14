import pickle
import matplotlib.pyplot as plt

precision_bifur = 0
recall_bifur = 1
precision_cross = 2
recall_cross = 3

mPrecisionBifur = 0
mRecallBifur = 0
mPrecisionCross = 0
mRecallCross = 0

file = open('metrics_textCalvoeUnet','rb')
metrics = pickle.load(file)
#print(metrics)

error=0.1

file = open('metrics_CalvoCalvoeUnet','rb')
metrics = pickle.load(file)
paths = metrics['paths']
images = metrics['images']
figs = ['o','d','*']
cor = ['r','b','g']

for path,f,c in zip(paths,figs,cor):
    print(path)
    for image in images:
        print(image)
        try:
            mPrecisionBifur = mPrecisionBifur + metrics[path][image][precision_bifur]
            mRecallBifur = mRecallBifur + metrics[path][image][recall_bifur]
            #print(metrics[path][image][precision_bifur],metrics[path][image][recall_bifur])
            #plt.plot(metrics[path][image][precision_bifur],metrics[path][image][recall_bifur],'o'+c)
        except:
            image = image.split('.')[0] + ".tif"
            mPrecisionBifur = mPrecisionBifur + metrics[path][image][precision_bifur]
            mRecallBifur = mRecallBifur + metrics[path][image][recall_bifur]
            #print(metrics[path][image][precision_bifur],metrics[path][image][recall_bifur])
            #plt.plot(metrics[path][image][precision_bifur],metrics[path][image][recall_bifur],'o'+c)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title("Precision x Recall para bifurcações " )
    plt.savefig("ResultsBifurcation5Images"+".pdf")
plt.show()


for path,f,c in zip(paths,figs,cor):
    print(path)
    for image in images:
        print(image)
        try:
            mPrecisionCross = mPrecisionCross + metrics[path][image][precision_cross]
            mRecallCross = mRecallCross + metrics[path][image][recall_cross]
            #print(metrics[path][image][precision_cross],metrics[path][image][recall_cross])
            #plt.plot(metrics[path][image][precision_cross],metrics[path][image][recall_cross],'o'+c)
        except:
            image = image.split('.')[0] + ".tif"
            mPrecisionCross = mPrecisionCross + metrics[path][image][precision_cross]
            mRecallCross = mRecallCross + metrics[path][image][recall_cross]
            #print(metrics[path][image][precision_cross],metrics[path][image][recall_cross])
            #plt.plot(metrics[path][image][precision_cross],metrics[path][image][recall_cross],'o'+c)

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title("Precision x Recall para cruzamentos" )
    plt.savefig("ResultsCrossove5Images"+".pdf")
plt.show()

num_images = len(images)

print("mPrecisionBifur: ",mPrecisionBifur/num_images," mRecallBifur: ",mRecallBifur/num_images\
    ," mPrecisionCross: ",mPrecisionCross/num_images," mRecallCross: ",mRecallCross/num_images)

