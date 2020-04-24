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

print("Creating F4: sum distance to 64 nearest neighbours of each class")
from scipy.spatial import distance_matrix

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)



ind_1 = np.where(y_train==1)
ind_0 = np.where(y_train==0)
x_train_1 = x_train[ind_1]
x_train_0 = x_train[ind_0]
f1_train = np.zeros(x_train.shape[0])
f0_train = np.zeros(x_train.shape[0])
print(x_train.shape,x_train_1.shape)
d_train_1 = distance_matrix(x_train,x_train_1)
d_train_0 = distance_matrix(x_train,x_train_0)
print(d_train_1.shape,d_train_0.shape)
for i in range(x_train.shape[0]):
	f1_train[i] = np.sum(np.sort(d_train_1[i])[-64:])
	f0_train[i] = np.sum(np.sort(d_train_0[i])[-64:])

f1_val = np.zeros(x_val.shape[0])
f0_val = np.zeros(x_val.shape[0])
print(x_val.shape,x_train_1.shape)
d_val_1 = distance_matrix(x_val,x_train_1)
d_val_0 = distance_matrix(x_val,x_train_0)
print(d_val_1.shape,d_val_0.shape)
for i in range(x_val.shape[0]):
	f1_val[i] = np.sum(np.sort(d_val_1[i])[-64:])
	f0_val[i] = np.sum(np.sort(d_val_0[i])[-64:])

f1_test = np.zeros(x_test.shape[0])
f0_test = np.zeros(x_test.shape[0])
print(x_test.shape,x_train_1.shape)
d_test_1 = distance_matrix(x_test,x_train_1)
d_test_0 = distance_matrix(x_test,x_train_0)
print(d_test_1.shape,d_test_0.shape)
for i in range(x_test.shape[0]):
	f1_test[i] = np.sum(np.sort(d_test_1[i])[-64:])
	f0_test[i] = np.sum(np.sort(d_test_0[i])[-64:])


np.save('F4_0_train.npy',f0_train)
np.save('F4_1_train.npy',f1_train)
np.save('F4_0_val.npy',f0_val)
np.save('F4_1_val.npy',f1_val)
np.save('F4_0_test.npy',f0_test)
np.save('F4_1_test.npy',f1_test)


