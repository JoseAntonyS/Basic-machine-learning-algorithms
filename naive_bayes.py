import numpy as np 
import matplotlib.pyplot as plt 
import seaborn 
seaborn.set()
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

model = make_pipeline(TfidfVectorizer(),MultinomialNB())

data=fetch_20newsgroups()
catogories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
train =fetch_20newsgroups(subset='train', categories=catogories)
test = fetch_20newsgroups(subset='test', categories=catogories)

model.fit(train.data, train.target)

labels = model.predict(test.data)

mat=confusion_matrix(labels, test.target)
seaborn.heatmap(mat.T, square=True, annot= True, fmt='d', cbar=True, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('True labels')
plt.ylabel("predicted values")
plt.show()

print("accuracy_score : "+str(accuracy_score(labels, test.target)*100)) 
