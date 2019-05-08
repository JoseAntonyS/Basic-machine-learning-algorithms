import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('./data/iris.csv')

dataset['name']=dataset['name'].replace('Iris-setosa',1)
dataset['name']=dataset['name'].replace('Iris-versicolor',2)
dataset['name']=dataset['name'].replace('Iris-virginica',3)

#Split dataset
x = dataset.iloc[:,0:4]
y = dataset.iloc[:,4]
x_train, x_test, y_train, y_test =  train_test_split(x,y,random_state=0, test_size=0.2)

entropy=DecisionTreeClassifier(criterion='entropy',random_state=100, max_depth=3, min_samples_leaf=5)
entropy.fit(x_train,y_train)
y_pred = entropy.predict(x_test)
print ("accuracy_score = "	+ str(accuracy_score(y_test,y_pred)*100))
