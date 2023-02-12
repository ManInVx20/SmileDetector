# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import image_utils
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse tha arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to the input dataset of faces')
ap.add_argument('-m', '--model', required=True, help='path to output model')
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []

# loop over the input images
# example of image path in list : SMILEs\positives\positives7\872.jpg
for image_path in sorted(list(paths.list_images(args['dataset']))):
    # load the image, pre-process it, and store in the data list
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = image_utils.img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the labels list
    label = image_path.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in the labeled data
class_totals = labels.sum(axis=0)
class_weight = class_totals.max() / class_totals

# partition the data into training and testing splits using 80% of
# split datta: 80% for training and 20% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# initialize the model
print('[INFO] compiling model...')

# LeNet architecture that will accept 28×28 single channel images.
# given that there are only two classes (smiling versus not smiling), we set classes=2
model = Sequential()
height, width, depth, classes = 28, 28, 1, 2
input_shape = (height, width, depth)

# if we are using 'channels first', update the input shape
if K.image_data_format() == 'channels_first':
    input_shape = (depth, height, width)

# first set of CONV => ReLU => POOL layers
model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second layer of CONV => ReLU => POOL layers
model.add(Conv2D(50, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# first (and only) set of FC => ReLU layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

# softmax classifier
model.add(Dense(classes))
model.add(Activation('softmax'))

# model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss=['binary_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

# train the network
print('[INFO] training network...')
print(class_weight)
print('classWeight datatype')
print(type(class_weight))

class_weight = dict(enumerate(reversed(class_weight), 0))
print(class_weight)

H = model.fit(train_x, train_y, validation_data=(test_x, test_y), class_weight=class_weight, batch_size=64, epochs=15, verbose=1)

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(test_x, batch_size=64)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print('[INFO] serializing network')
model.save(args['model'])

# show figure of training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 15), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 15), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 15), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, 15), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
