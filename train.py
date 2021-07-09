import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import pickle

numberList = os.listdir('dataFile')
images = []
label = []

for num in numberList:
    fileName = os.listdir('dataFile/' + num) 
    print(f"Copying images:{num}")
    for item in fileName:
        img = cv2.imread('dataFile/' + num + '/' + item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        img = img.reshape(32, 32, 1)
        img = img/255
        images.append(img)
        label.append(num)

imageArray = np.array(images)
labelArray = np.array(label)

print(imageArray.shape)
print(labelArray.shape)

X_train, X_test, Y_train, Y_test = train_test_split(imageArray, labelArray, test_size = 0.2)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = 0.2)

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
Y_validation = to_categorical(Y_validation, 10)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_validation.shape, Y_validation.shape)

datagen = ImageDataGenerator(width_shift_range=0.2, 
                            height_shift_range=0.2,
                            zoom_range= 0.2,
                            shear_range=0.2,
                            rotation_range=30,
                            fill_mode= 'nearest')


def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5,5), input_shape = (32, 32, 1), activation = 'relu'))
    model.add(Conv2D(60, (5,5), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(30, (3,3), activation = 'relu'))
    model.add(Conv2D(30, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(500, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

    return model 

model = myModel()
print(model.summary())

history = model.fit(datagen.flow(X_train, Y_train, batch_size = 100),
                                steps_per_epoch = 65,
                                epochs = 5,
                                validation_data = (X_validation, Y_validation),
                                shuffle = 1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title("Loss")
plt.xlabel('no. of epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title("Accuracy")
plt.xlabel('no. of epoch')
plt.show()

score = model.evaluate(X_test, Y_test, verbose=0)
print("Test Accuracy:", score[1])

model.save('detectorModel.h5', overwrite=True, include_optimizer=True)