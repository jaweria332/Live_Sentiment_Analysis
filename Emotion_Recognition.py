#Importing necessary libraries
import sys,os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation, Flatten
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


#Reading the dataset
df = pd.read_csv('fer2013.csv')

#Splitting into train and test set
X_train, Y_train,X_test, Y_test = [],[],[],[]

#Splitting on the basis of space
for index, row in df.iterrows():
    val=row['pixels'].split(" ")

    try:
        # Separating Training points
        if 'Training' in row['Usage']:
            X_train.append(np.array(val,'float32'))
            Y_train.append(row['emotion'])
        # And test points
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val,'float32'))
            Y_test.append(row['emotion'])

    except:
        print(f"error occured at index :{index} and row :{row}")


num_features=64
num_labels=7
batch_size=64
epochs=10
width,height=48,48

#Getting data for training set and test set
X_train = np.array(X_train,'float32')
Y_train = np.array(Y_train,'float32')
X_test = np.array(X_test,'float32')
Y_test = np.array(Y_test,'float32')

#Changing data into categorical
Y_train=np_utils.to_categorical(Y_train,num_classes=num_labels)
Y_test=np_utils.to_categorical(Y_test,num_classes=num_labels)

#Normalizing data between 0 and 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train,axis=0)

X_test -= np.mean(X_test,axis=0)
X_test /= np.std(X_test,axis=0)

X_train=X_train.reshape(X_train.shape[0],48,48,1)
X_test=X_test.reshape(X_test.shape[0],48,48,1)


#Defining Convolutional Neural Network Model
model=Sequential()

#Adding first layer
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#Adding 2nd Convolutional Layer
model.add(Conv2D(64, (3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

#Adding 3rd Convolutional Layer
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#Adding Flatten Layer
model.add(Flatten())

#Fully connected neural network
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels,activation='relu'))

#Compiling the model
model.compile(loss=categorical_crossentropy,optimizer=Adam(),metrics=['accuracy'])

#Training the model
model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,Y_test),shuffle=True)

#Saving the model
fer_json=model.to_json()
with open("fer.json","w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")