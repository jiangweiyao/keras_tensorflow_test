from keras.models import Sequential
from keras.layers import Dense
import numpy


# set seed 
numpy.random.seed(123)

# load data
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create trivial 1 hidden layer model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=100, batch_size=100, validation_split=0.2, verbose=1)

# evaluate the model
scores = model.evaluate(X, Y, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# save model as H5 file
model.save("model.h5")
