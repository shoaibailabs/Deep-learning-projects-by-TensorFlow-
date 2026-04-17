import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import pandas as pd


from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape) 

x_train=x_train/255
x_test=x_test/255

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten

model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation='softmax'))
# print(model.summary())

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,validation_split=0.2)

from sklearn.metrics import accuracy_score
y_prd=model.predict(x_test)
y_prd=y_prd.argmax(axis=1)
print(y_prd.shape)

score=accuracy_score(y_test,y_prd)
print(score)

# prediction
plt.imshow(x_test[0])
plt.show()

prd=model.predict(x_test[0].reshape(1,28,28))
print(prd.argmax(axis=1))

