from AMLmodel import ML_model
import sys
import json
import io
import pickle
import requests
import uuid
from minio_commands import load_data, load_model, save_model
import keras
import os

def load_AMLdata(datasorts):
    
    print("in load_AMLdata")
    print(datasorts)
    num_classes = 15
    if type(datasorts) is not list:
        datasorts = list(datasorts)

    ret = []
    for datasort in datasorts:
        folder = 'AMLdata'
        x_data = load_data(folder + '/x_' + datasort + '.p')
        y_data = load_data(folder + '/y_' + datasort + '.p')
        
        y_data = keras.utils.to_categorical(y_data, num_classes)
        x_data = x_data.astype('float32')
        x_data /= 255
        ret += [x_data, y_data]
        
    return ret





if __name__ == '__main__':

    load_model_id = str(sys.argv[1])
    save_model_id = str(sys.argv[2])

    if len(sys.argv) > 3:
        epochs = int(sys.argv[3])

    print("arguments: ", sys.argv)

    if load_model_id == 'None':
        load_model_id = None

    if save_model_id == 'None':
        save_model_id = None



    x_train, y_train, x_val, y_val = load_AMLdata(datasorts=['train', 'val'])

    if load_model_id is not None:
        model = load_model(load_model_id)
    else:
        model = ML_model()

    def save_wrapper(model_obj):
        save_model(model_obj, name='CNN-model')

    model.fit(x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val, batch_size=10, epochs=epochs,
              data_augmentation=True, savings=save_wrapper)




