import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import inspect

def sklearn_method(x_train, x_test, y_train, y_test):
	#scaling data 
	s_x = StandardScaler()
	x_train = s_x.fit_transform(x_train)
	x_test = s_x.fit_transform(x_test)

	#actual classifier
	classifier=KNeighborsClassifier(n_neighbors=11	, p=2, metric='euclidean')
	classifier.fit(x_train,y_train)

	#predicting
	x_pred = classifier.predict(x_test)

	#metrics
	c=confusion_matrix(y_test,x_pred)
	print('confusion matrix = '+str(c))
	f=f1_score(y_test,x_pred)
	print('f1_score = '+str(f))
	a=accuracy_score(y_test,x_pred)
	print('accuracy_score = '+str(a))

dataset=pd.read_csv('./data/diabetes.csv')
#print(dataset.head(0))
should_not_be_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

for col in should_not_be_zero:
	dataset[col]=dataset[col].replace(0,np.NaN)
	mean=int(dataset[col].mean(skipna=True))
	dataset[col]=dataset[col].replace(np.NaN,mean)

#Split dataset
x = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
x_train, x_test, y_train, y_test =  train_test_split(x,y,random_state=0, test_size=0.2)

sklearn_method(x_train, x_test, y_train, y_test)
