from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd 

data = pd.read_csv('./data/mnist_train.csv')

print(data.head())