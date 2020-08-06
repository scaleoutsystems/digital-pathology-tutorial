import pickle
import numpy as np
from matplotlib import pylab as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from sklearn.metrics import classification_report
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

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

    opt = keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
    # opt = keras.optimizers.SGD(learning_rate=learning_rate, decay=decay)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                       optimizer=opt,
                       metrics=['accuracy'])

    return model


class ML_model:

    def __init__(self):
        self.model = construct_model()
        self.history = []
        self.best_weights = None
        self.max_accuracy = 0
        self.val_accuracy = []
        self.val_loss = []




    def predict(self, inp):
        print('predicting')
        if 'file' in inp:
            print('file')
            filename = inp['file']
            print('loading image')
            img = load_img(filename)
            print('loaded image')
            img_array = img_to_array(img)
            if img_array.shape == (100,100,3):
                img_array = np.concatenate((img_array, 255*np.ones((100, 100, 1))), axis=2)
            if img_array.shape != (100,100,4):
                print('Image has wrong dimension')
#                 raise Exception('Image has wrong dimension')
            print('calling predict')
            pred = self.model.predict(np.array([img_array]))
            pred = pred[0].tolist()
            max_pred = pred.index(max(pred))
            for i, k in class_keys.items():
                if k==max_pred:
                    pred_type = i
                    break
            print('prediction')
            print(pred)
            print({"most likely":pred_type, "all_probabilities":pred})
            return {"most likely":pred_type, "all_probabilities":pred}
        if 'json' in inp:
            data = inp['json']
            return self.model.predict(np.array(data))



    def partial_fit(self, x_train, y_train, batch_size=50, data_augmentation=True):


        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=180,  #[M] randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True,  #[M] randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                validation_split=0.1)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                         epochs=1,
                                         workers=4)
                                         # callbacks=[mcp_save])




    def fit(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=50, data_augmentation=True, savings=None):

        for e in range(epochs):
            self.partial_fit(x_train,y_train,batch_size=batch_size, data_augmentation=data_augmentation)
            loss, accuracy = self.model.evaluate(x_val, y_val)

            self.val_loss += [loss]
            self.val_accuracy += [accuracy]

            if savings is not None:
                savings(self)

            if accuracy > self.max_accuracy:
                self.best_weights = self.model.get_weights()
                self.max_accuracy = accuracy


    def evaluate(self,x_data, y_data):

        temp_model = construct_model()

        try:
            temp_model.set_weights(self.best_weights)

        except:
            print("Model have no best weights to load")

        return temp_model.evaluate(x_data,y_data)








    def confusion_matrix(self, x_data, y_data):

        y_pred = self.model.predict_classes(x_data)
        y_data = np.argmax(y_data,1)
        M = np.zeros((16,16))

        for pred_, true_ in zip(y_pred,y_data):
            M[true_,pred_] +=1

        # for p in range(15):
        #     for l in range(15):
        #         p_s = set(np.where(y_pred == p)[0])
        #         l_s = set(np.where(y_data == l)[0])
        #         M[p,l] = len(p_s.intersection(l_s))
        #
        # for p in range(15):
        #     len(np.where(y_pred == p)[0])
        return M


    def classification_report(self, x_data, y_data):

        y_predict = self.model.predict_classes(x_data)
        y_data = np.argmax(y_data,1)
        abbr = list(class_keys.keys())
        name_classes = [[]] * 15
        for i in range(15):
            name_classes[class_keys[abbr[i]]] = classes[abbr[i]]
        report = classification_report(y_data, y_predict, target_names=name_classes)
        return report

