import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


x_train = np.load('x_train.npy',allow_pickle=True)
print("x_train.shape=",x_train.shape)
x_val = np.load('x_val.npy',allow_pickle=True)
print("x_val.shape=",x_val.shape)
y_train = np.load('y_train.npy',allow_pickle=True)
print("y_train.shape=",y_train.shape)
y_val = np.load('y_val.npy',allow_pickle=True)
print("y_val.shape=",y_val.shape)
x_test = np.load('x_test.npy',allow_pickle=True)
print("x_test.shape=",x_test.shape)


print("Creating F7: num non zero elem each row" )

f7_train = np.zeros(x_train.shape[0])
f7_val = np.zeros(x_val.shape[0])
f7_test = np.zeros(x_test.shape[0])
for i in tqdm(range(x_train.shape[0])):
	f7_train[i] = np.sum(x_train[i,:]!=0)
for i in range(x_val.shape[0]):
	f7_val[i] = np.sum(x_val[i,:]!=0)
for i in range(x_test.shape[0]):
	f7_test[i] = np.sum(x_test[i,:]!=0)
print(np.corrcoef(y_train,f7_train))
print(np.corrcoef(y_val,f7_val))
np.save('f7_train.npy', f7_train)
np.save('f7_val.npy', f7_val)
np.save('f7_pred.npy',f7_test)
print(f7_train.shape,f7_val.shape,f7_test.shape)

