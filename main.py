import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import pandas as pd

df=pd.read_csv('D:\\my practise\\csv files\\Churn_Modelling (1).csv')
# pd.set_option('Display.max_columns',14)

nan=df.isnull().sum().sum()
duplicat=df.duplicated().sum()

df.drop(columns=['Surname','RowNumber','CustomerId'],inplace=True)
df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True,dtype='int64')

x=df.iloc[:,:-1]
y=df['Exited']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

import tensorflow
from keras.models import Sequential
from keras.layers import Dense


model=Sequential()
model.add(Dense(256,activation='relu',input_dim=11))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=3,validation_split=0.2)

summary=model.summary()
print(summary)

y_log=model.predict(x_test)
y_pred=np.where(y_log>0.5,1,0)
from sklearn.metrics import accuracy_score
acc_score=accuracy_score(y_test,y_pred)*100

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()
