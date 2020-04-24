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

s_train = np.zeros(x_train.shape[0])
s_val = np.zeros(x_val.shape[0])
s_test = np.zeros(x_test.shape[0])
for i in range(x_train.shape[0]):
	s_train[i] = (np.sum(x_train[i,:]==np.min(x_train[i,:])))
for i in range(x_val.shape[0]):
	s_val[i] = (np.sum(x_val[i,:]==np.min(x_val[i,:])))
for i in range(x_test.shape[0]):
	s_test[i] = (np.sum(x_test[i,:]==np.min(x_test[i,:])))

s_train = (s_train - np.min(s_train) ) / (np.max(s_train) - np.min(s_train))
s_val = (s_val - np.min(s_val) ) / (np.max(s_val) - np.min(s_val))
s_test = (s_test - np.min(s_test) ) / (np.max(s_test) - np.min(s_test))

x_train = np.concatenate((x_train,s_train.reshape((len(s_train),1))),axis=1)
x_val = np.concatenate((x_val,s_val.reshape((len(s_val),1))),axis=1)
x_test = np.concatenate((x_test,s_test.reshape((len(s_test),1))),axis=1)
print("x_train.shape=",x_train.shape)
print("x_val.shape=",x_val.shape)
print("x_test.shape=",x_test.shape)

# from imblearn.over_sampling import SMOTE, ADASYN
# x_train, y_train = ADASYN().fit_resample(x_train, y_train)

x_train = np.concatenate((x_train,x_val),axis=0)
y_train = np.concatenate((y_train,y_val))

from xgboost import XGBClassifier
sumpos = np.sum(y_train==1)
sumneg = np.sum(y_train==0)
model = XGBClassifier(max_depth=2,n_estimators=300,learning_rate=0.05,
	reg_alpha=0,reg_lambda=1.0,n_jobs=-1,
	scale_pos_weight=sumneg/sumpos,eval_metric='auc')
print(model)
model.fit(x_train,y_train)

yh_val = model.predict_proba(x_val)[:,1]
print("auc=",roc_auc_score(y_val,yh_val))

yh_test = model.predict_proba(x_test)[:,1]
np.save('m15_pred',yh_test)
print("yh_test.shape=",yh_test.shape)
np.save('m15_val',yh_val)
print("yh_val.shape=",yh_val.shape)
yh_train = model.predict_proba(x_train)[:,1]
np.save('m15_train',yh_train)
print("yh_train.shape=",yh_train.shape)

pd.DataFrame(yh_test).to_csv('FINAL_M15XGBReg.csv')