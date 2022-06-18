# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
ct=ColumnTransformer([("Country",OneHotEncoder(),[1])],remainder='passthrough')
X = ct.fit_transform(X)
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing keras
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import InputLayer
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier=Sequential()
    input_matrix=InputLayer(input_shape=(11,))
    classifier.add(input_matrix)
    classifier.add(Dense(6,kernel_initializer='glorot_uniform',activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(6,kernel_initializer='glorot_uniform',activation='relu'))
    classifier.add(Dense(1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    return classifier

classifier=KerasClassifier(build_classifier,loss='binary_crossentropy',batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
mean=accuracies.mean()
variance=accuracies.std()

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred > 0.5)

new_prediction=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#tuned ANN
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import InputLayer
from sklearn.model_selection import GridSearchCV

def build_classifier():
    classifier=Sequential()
    input_matrix=InputLayer(input_shape=(11,))
    classifier.add(input_matrix)
    classifier.add(Dense(6,kernel_initializer='glorot_uniform',activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(6,kernel_initializer='glorot_uniform',activation='relu'))
    classifier.add(Dense(1,kernel_initializer='glorot_uniform',activation='sigmoid'))
    return classifier

classifier=KerasClassifier(build_classifier,loss='binary_crossentropy')
parameters={'batch_size':[24,30],
            'epochs':[100,200],
            }
grid_search=GridSearchCV(estimator=classifier, param_grid=parameters,cv=10,
                         scoring='accuracy')
grid_search=grid_search.fit(X_train,y_train)
best_params=grid_search.best_params_
best_accuracy=grid_search.best_score_
