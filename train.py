#importing libraries
import numpy as np 
import pandas as pd
import random as rd

#data visualization
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import plotly.express as px
from PIL import Image

#for the CNN model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.preprocessing.image import ImageDataGenerator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#setting seed for reproducability
from numpy.random import seed
seed(10)
tf.random.set_seed(20)

train = pd.read_csv('train.csv')
train.head()

test = pd.read_csv('test.csv')
test.head()

print(sum(train.isnull().sum()))
print(sum(test.isnull().sum()))

Y_test = test["label"]
X_test = test.drop(labels = ["label"],axis = 1)

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)


X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
print(X_train.shape)
print(X_test.shape)

fig = px.histogram(train, 
                   x='label', 
                   color = 'label',
                   title="Distrubition of Labels in the Training Set",
                   width=700, height=500)
fig.show()

fig = px.histogram(test, 
                   x='label',
                   color = 'label',
                   title="Distrubition of Labels in the Test Set",
                   width=700, height=500)
fig.show()