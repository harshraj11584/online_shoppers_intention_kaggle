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


# x_train, x_val, x_test = np.log(x_train+1), np.log(x_val+1), np.log(x_test+1)

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,n_iter=300,method='exact',n_jobs=-1)
x_train_embedded = tsne.fit_transform(x_train)
print("Embedded x_train")
sns.scatterplot(x_train_embedded[:,0], x_train_embedded[:,1], hue=y_train, legend='full', palette=palette)
plt.show()

# x_val_embedded = tsne.fit_transform(x_val)
# print("Embedded x_val")
# x_test_embedded = tsne.fit_transform(x_test)
# print("Embedded x_test")

# np.save('m11_train',x_train_embedded)
# print("x_train_embedded.shape=",x_train_embedded.shape)
# np.save('m11_val',x_val_embedded)
# print("x_val_embedded.shape=",x_val_embedded.shape)
# np.save('m11_pred',x_test_embedded)
# print("x_test_embedded.shape=",x_test_embedded.shape)