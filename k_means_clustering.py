import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sea ; sea.set()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin, accuracy_score

#implementation of kmeans(optional)
def kmeans_manual(x,n_clusters,rseed=2):
	r=np.random.RandomState(rseed)
	i=r.permutation(x.shape[0])[:n_clusters]
	centers=x[i]

	while True:
		labels=pairwise_distances_argmin(x, centers)

		new_centers=np.array([x[labels==i].mean(0) for i in range(n_clusters)])

		if np.all(centers==new_centers):
			break
		centers=new_centers
		pass
	return centers,labels

#initalize dataset
x,y = make_blobs(n_samples =500, centers=4, cluster_std=0.60, random_state=0) #test data
dataset=pd.read_csv('./data/iris.csv')

dataset['name']=dataset['name'].replace('Iris-setosa',0)
dataset['name']=dataset['name'].replace('Iris-versicolor',1)
dataset['name']=dataset['name'].replace('Iris-virginica',2)

#Split dataset
x = dataset.values[:,[0,1]]
y = dataset["name"]

#actual algorithm
cluster=KMeans(n_clusters=3)
cluster.fit(x)
y_kmeans=cluster.predict(x)

#actual algorithm(optional)
y_kmeans_optional,labels=kmeans_manual(x,3)

print("accuracy_score : "+str(accuracy_score(y,labels)))
#ploting
plt.scatter(x[:,0],x[:,1],c=labels,s=50)
plt.scatter(y_kmeans_optional[:,0],y_kmeans_optional[:,1],marker='*',c='black',s=50,alpha=0.5)
plt.show()