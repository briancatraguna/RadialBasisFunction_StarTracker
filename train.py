#import classes and functions
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#Load the dataset
dataframe = pandas.read_csv("ann_features_binIncrement1.csv",header=None)
dataset = dataframe.values[1:,:] #Separating the header
X = dataset[:,1:].astype(float) #Features is from column 1 to the end, 12 feature dimensions
Y = dataset[:,0] #Label is from column 0

#Encode the output variable
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

#Define the neural network model
def baseline_model():
    model = Sequential()
    model.add(Dense(24,input_dim=12,activation='relu'))
    model.add(Dense(units=48,activation='relu'))
    model.add(Dense(units=96,activation='relu'))
    model.add(Dense(units=192,activation='relu'))
    model.add(Dense(units=384,activation='relu'))
    model.add(Dense(units=301,activation='relu'))
    model.add(Dense(units=301,activation='relu'))
    model.add(Dense(301,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model,epochs=200,batch_size=5,verbose=0)

#Evaluate the model with k-Fold cross validation
kfold = KFold(n_splits=10,shuffle=True)
results = cross_val_score(estimator,X,dummy_y,cv=kfold)
print("Baseline: {0} {1}".format(results.mean()*100,results.std()*100))