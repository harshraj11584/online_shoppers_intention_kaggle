import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data.dropna(inplace=True)
print("data.shape=",data.shape)
data = pd.get_dummies(data)
data_test = pd.get_dummies(data_test)
le = LabelEncoder()
data['Revenue'] = le.fit_transform(data['Revenue'])
# data_test['Revenue'] = le.transform(data_test['Revenue'])

x = data.drop(['Revenue'], axis = 1)
y = data['Revenue']

x_test = data_test

x = np.array(x).astype(np.float32)
x_test = np.array(x_test).astype(np.float32)

for i in range(x.shape[1]):
	if (np.max(x[:,i]) - np.min(x[:,i]))!=0 and (np.max(x_test[:,i]) - np.min(x_test[:,i]))!=0:
		x[:,i] = (x[:,i] - np.min(x[:,i]))/(np.max(x[:,i]) - np.min(x[:,i]))
		x_test[:,i] = (x_test[:,i] - np.min(x_test[:,i]))/(np.max(x_test[:,i]) - np.min(x_test[:,i]))
y = np.array(y).astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, random_state = 0)


np.save('x_train',x_train)
print("x_train.shape=",x_train.shape)
np.save('x_val',x_val)
print("x_val.shape=",x_val.shape)
np.save('y_train',y_train)
print("y_train.shape=",y_train.shape)
np.save('y_val',y_val)
print("y_val.shape=",y_val.shape)
np.save('x_test',x_test)
print("x_test.shape=",x_test.shape)
