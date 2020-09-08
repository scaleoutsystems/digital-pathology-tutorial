import keras
import pickle
import numpy as np
import os

from src.models.AMLmodel import ML_model

from pathlib import Path

base_path = Path(__file__).parent
import_folder = str((base_path / "../../dataset").resolve())


x_train = pickle.load( open(import_folder + "/raw/x_train.p", "rb" ) )
y_train = pickle.load( open(import_folder + "/raw/y_train.p", "rb" ) )

x_val = pickle.load( open(import_folder + "/raw/x_val.p", "rb" ) )
y_val = pickle.load( open(import_folder + "/raw/y_val.p", "rb" ) )

x_test = pickle.load( open(import_folder + "/raw/x_test.p", "rb" ) )
y_test = pickle.load( open(import_folder + "/raw/y_test.p", "rb" ) )

def pre_process(x_data, y_data):
    num_classes = 15
    y_data = keras.utils.to_categorical(y_data, num_classes)
    x_data = x_data.astype('float32')
    x_data /= 255
    return x_data, y_data

x_train, y_train = pre_process(x_train,y_train)
x_val, y_val = pre_process(x_val,y_val)

proc_data = (x_train, y_train, x_val, y_val)

f = open(import_folder+"/processed/data.p", 'wb')
pickle.dump(proc_data, f)