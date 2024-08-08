class meraLR:
    def __init__(self):
        self.m=None
        self.b=None
    
    def fit(self,X_train,y_train):
        # print(X_train.shape)
        num=0
        den=0 
        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean()) *(y_train[i] - y_train.mean()))
            den = den + ((X_train[i] - X_train.mean()) *(X_train[i] - X_train.mean()))
        self.m=num/den
        self.b=y_train.mean() - (self.m * X_train.mean())


    def predict(self,X_test):
        print(X_test)
        return self.m * X_test + self.b


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv(r'E:\datasets\placement_students.csv')
df.head()

X=df.iloc[:,0].values
y=df.iloc[:1].values
# print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

lr=meraLR()

lr.fit(X_train,y_train)



