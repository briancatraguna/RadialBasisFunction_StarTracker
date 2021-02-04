import numpy as np
import pandas as pd
import tensorflow as tf

#Separating labels and features
dataset = pd.read_csv('Without_Noise.csv')
x = dataset.iloc[:,1:].values #76545 rows, 12 columns

#Do one hot encoding for the label
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc_y = pd.DataFrame(enc.fit_transform(dataset[['Star ID']]).toarray())
y = enc_y.iloc[:,:].values #76545 rows, 5103 columns

#Split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
minmaxsc = MinMaxScaler()
X_train = minmaxsc.fit_transform(X_train)
X_test = minmaxsc.transform(X_test)

#Initializing the ANN as a sequence of layers
ann = tf.keras.models.Sequential()
#Adding the input layer (automatically) and the first hidden layer
ann.add(tf.keras.layers.Dense(units=12,activation='relu'))
#Second hidden layer
ann.add(tf.keras.layers.Dense(units=144,activation='relu'))
#Third hidden layer
ann.add(tf.keras.layers.Dense(units=720,activation='relu'))
#Fourth hidden layer
ann.add(tf.keras.layers.Dense(units=3600,activation='relu'))
#Output layer
ann.add(tf.keras.layers.Dense(units=5103,activation='sigmoid'))

#Compile the ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the ANN on the Training set
# ann.fit(X_train,y_train,batch_size=300,epochs=100)