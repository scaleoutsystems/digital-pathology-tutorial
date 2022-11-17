import shutil
import pickle
from pathlib import Path
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tempfile import NamedTemporaryFile

import psutil
learning_rate = 0.001
num_classes=15
decay = 0
class_keys = {'BAS': 0, 'EBO': 1, 'EOS': 2, 'KSC': 3, 'LYA': 4, 'LYT': 5, 'MMZ': 6, 'MOB': 7, 'MON': 8, 'MYB': 9 , 'MYO': 10,
        'NGB': 11, 'NGS': 12, 'PMB': 13, 'PMO': 14}

classes = {'BAS': 'Basophil',
'EBO': 'Erythroblast',
'EOS': 'Eosinophil',
'KSC': 'Smudge cell',
'LYA': 'Lymphocyte (atypical)',
'LYT': 'Lymphocyte (typical)',
'MMZ': 'Metamyelocyte',
'MOB': 'Monoblast',
'MON': 'Monocyte',
'MYB': 'Myelocyte',
'MYO': 'Myeloblast',
'NGB': 'Neutrophil (band)',
'NGS': 'Neutrophil (segmented)',
'PMB': 'Promyelocyte (bilobled)',
'PMO': 'Promyelocyte'}


def construct_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=[100,100,4]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
    # opt = keras.optimizers.SGD(learning_rate=learning_rate, decay=decay)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                       optimizer=opt,
                       metrics=['accuracy'])

    return model




