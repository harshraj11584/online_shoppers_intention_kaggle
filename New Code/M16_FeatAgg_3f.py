import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import seaborn as sns
sns.set()

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train, x_val, x_test = np.log(x_train+1), np.log(x_val+1), np.log(x_test+1)

# from sklearn.cluster import KMeans
# model = KMeans(n_clusters=3,n_jobs=-1,max_iter=100000)
# model.fit(x_train)
# x_train_2 = model.transform(x_train)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_train_2[:,0],x_train_2[:,1],x_train_2[:,2],c=y_train)

# sns.scatterplot(x_train_2[:,0],x_train_2[:,1],hue=y_train)
# plt.show()

from sklearn.cluster import FeatureAgglomeration
model = FeatureAgglomeration(n_clusters=3,linkage='complete',pooling_func=np.mean)
model.fit(x_train)
x_train_2 = model.transform(x_train)
x_val_2 = model.transform(x_val)
x_test_2 = model.transform(x_test)

np.save('m16_pred',x_test_2)
print("x_test_2.shape=",x_test_2.shape)
np.save('m16_val',x_val_2)
print("x_val_2.shape=",x_val_2.shape)
np.save('m16_train',x_train_2)
print("x_train_2.shape=",x_train_2.shape)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_train_2[:,0],x_train_2[:,1],x_train_2[:,2],c=y_train)
# plt.show()





