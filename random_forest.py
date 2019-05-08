import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn import tree, linear_model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import csv
import xlrd

def csv_from_excel_train():

	wb = xlrd.open_workbook('./data/training_1502.xlsx')
	sh = wb.sheet_by_name('Sheet1')
	your_csv_file = open('./data/csv_file_train.csv', 'wb')
	wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

	for rownum in xrange(sh.nrows):
	    wr.writerow(sh.row_values(rownum))

	your_csv_file.close()

def csv_from_excel_test():

	wb = xlrd.open_workbook('./data/test_1502.xlsx')
	sh = wb.sheet_by_name('Sheet1')
	your_csv_file = open('./data/csv_file_test.csv', 'wb')
	wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

	for rownum in xrange(sh.nrows):
	    wr.writerow(sh.row_values(rownum))

	your_csv_file.close()


csv_from_excel_train()
csv_from_excel_test()

dataset = pd.read_csv('./data/csv_file_train.csv')
dataset = shuffle(dataset)

dataset_test = pd.read_csv('./data/csv_file_test.csv')


#Split dataset
y_train = dataset['class']
x_train = dataset.values[:,0:12]

x_test = dataset_test.values[:,0:12]
# y_test = dataset_test.values[0:-1,26]

#acutal classifier
# classifier=RandomForestClassifier(n_jobs=5 , random_state=0)
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

# # classifier.fit(x_train,y_train)
regr.fit(x_train,y_train)
# print(x_test)
# # y_pred_clas = classifier.predict(x_test)
y_pred_reg = regr.predict(x_test)


# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(x_train, y_train)

# # Make predictions using the testing set
# y_pred = regr.predict(x_test)
time=[]
f = open('results_without_conversion.csv','w')
for n,item in enumerate(y_pred_reg):
	time.append(n)
	print(item)
	f.write(str(item))
	f.write('\n')
f.close

plt.scatter(time, y_pred_reg)
plt.show()


# y_pred_reg = [int(i) for i in y_pred_reg]	
# #printing metrics
# print ("accuracy_score (classification) = "	+ str(accuracy_score(y_test,y_pred_clas)*100))
# print ("accuracy_score (regression) = "	+ str(accuracy_score(y_test,y_pred_reg)*100))