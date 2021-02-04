import numpy as np
import pandas as pd
import tensorflow as tf

#Separating labels and features
dataset = pd.read_csv('Without_Noise.csv')
x = dataset.iloc[:,1:].values #Features

#Do one hot encoding for the label
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc_y = pd.DataFrame(enc.fit_transform(dataset[['Star ID']]).toarray())
y = enc_y.iloc[:,:].values #76545 rows, 5103 columns

print(x.shape)
print(y.shape)