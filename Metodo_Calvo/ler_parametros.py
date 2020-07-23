import pickle

file = open('parameters','rb')
result_dict = pickle.load(file)
file.close()

print(result_dict)