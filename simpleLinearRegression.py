import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv(r'E:\datasets\placement_students.csv')
df.head()

plt.scatter(df['cgpa'],df['package'])
plt.xlabel('CGPA')
plt.ylabel("Package (in lpa)")
plt.show()

X=df.iloc[:,0:1]
y=df.iloc[:,-1]
# print(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)

lr.predict(X_test.iloc[0].values.reshape(1,1))


plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr.predict(X_train),color="red")
plt.xlabel('CGPA')
plt.ylabel("Package (in lpa)")
plt.show()

m=lr.coef_
print(m)

b=lr.intercept_


# y=mx+b  



