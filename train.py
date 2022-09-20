from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import numpy as np

batch_size=8
epoch=100


model= Sequential([Conv2D(filters= 100,kernel_size= (3,3),activation='relu',padding='same',input_shape=(150,150,3)),
                   MaxPooling2D(pool_size=(2,2),strides=2),

                   Conv2D(filters=100,kernel_size=(3,3),activation='relu',padding='same'),
                   MaxPooling2D(pool_size=(2,2),strides=2),

                   Conv2D(filters=100,kernel_size=(3,3),activation='relu',padding='same'),
                   MaxPooling2D(pool_size=(2,2),strides=2),

                   Flatten(),
                   Dropout(0.5),
                   Dense(units=64,activation='relu'),
                   Dense(units=2,activation='softmax'),

])
# model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

train_directory = './train'

train_datagen = ImageDataGenerator(rescale = 1./255,            # Multiple the colors by a number between 0-1 to process data faster
                                   rotation_range=40,           # rotate the images
                                   width_shift_range=0.2,
                                   height_shift_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')         # add new pixels when the image is rotated or shifted

train_generator = train_datagen.flow_from_directory(
                                train_directory,
                                target_size = (150, 150),
                                batch_size = batch_size)            # Specify this is training set

valid_directory='./test'
validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = train_datagen.flow_from_directory(
                                valid_directory,
                                target_size = (150, 150),
                                batch_size = batch_size)            # Specify this is training set

checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

history=model.fit(train_generator, epochs=epoch, validation_data=validation_generator, batch_size=batch_size,callbacks=[checkpoint])


# history = model.fit_generator(train_generator,
#                               epochs=3,
#                               validation_data=validation_generator,
#                               callbacks=[checkpoint])