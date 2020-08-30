from numpy import loadtxt
from keras.models import load_model

model = load_model('model.h5')
model.summary()

dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
