import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('Without_Noise.csv')
x = dataset.iloc[:,1:]
y = dataset.iloc[:,0]

print(x.head())
print(y.head())