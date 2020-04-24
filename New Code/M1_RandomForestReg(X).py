import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


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


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=600,random_state=420,class_weight='balanced')

x_train = np.concatenate((x_train,x_val),axis=0)
y_train = np.concatenate((y_train,y_val))

model.fit(x_train,y_train)
yh_val = model.predict_proba(x_val)[:,1]
print("AUC of Random Forest Regressor: ",roc_auc_score(y_val,yh_val))

yh_test = model.predict_proba(x_test)[:,1]
np.save('m1_pred',yh_test)
print("yh_test.shape=",yh_test.shape)
np.save('m1_val',yh_val)
print("yh_val.shape=",yh_val.shape)
yh_train = model.predict_proba(x_train)[:,1]
np.save('m1_train',yh_train)
print("yh_train.shape=",yh_train.shape)


pd.DataFrame(yh_test).to_csv('FINAL_RandomForest.csv')
