import pickle
import matplotlib.pyplot as plt

file = open('metrics_text2','rb')
metrics = pickle.load(file)
#print(metrics)

file = open('metrics_Calvo2','rb')
metrics = pickle.load(file)
#print(metrics)

plt.plot(metrics[0],metrics[1],'ob')
plt.plot(metrics[2],metrics[3],'or')
plt.ylabel("Recall")
plt.xlabel("Precision")
plt.xlim((0,1))
plt.ylim((0,1))
plt.title("Precision x Recall")
plt.show()