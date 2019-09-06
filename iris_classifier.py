"""
Reference: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
"""
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

"""
fix random seed for reproducibility
"""
seed = 7
numpy.random.seed(seed)

"""
load dataset
"""
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]


"""
Iris-setosa,	Iris-versicolor,	Iris-virginica
1,		0,			0
0,		1, 			0
0, 		0, 			1
"""
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


"""
define baseline model
"""
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


"""
Train model
"""
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

"""
Evaluate the model with k-Fold Cross Validation
"""
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
for result in results:
    print("Result: %.2f%% (%.2f%%)" % (result * 100, result * 100))

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
