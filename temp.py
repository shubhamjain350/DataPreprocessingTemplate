#Preprocessing 

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[ : , :-1].values
y=dataset.iloc[:, 3].values

#Fixing missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values= 'NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X= LabelEncoder()
#It is done cause ML dont work on strings, so converting each column to numbers
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#one hot encoder(dummy encoding)
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#for Y
labelencoder_y= LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Splitting the data set into training and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)