import os
from __future__ import print_function
import sys
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.models as krm
import pickle
import yaml

from models.AMLmodel import AMLModel
from data.load_data import load_processed
from data.read_data import read_data

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, settings):
    print("-- RUNNING TRAINING --", flush=True)

    (x_train, y_train, x_val, y_val) = load_processed('data.p')
    model.fit(x_train, y_train, x_val, y_val, batch_size=8, data_augmentation=False)

    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    from models.AMLmodel import construct_model
    model = construct_model()
    model.set_weights(weights)
    
    model = train(model,settings)
    helper.save_model(model.get_weights(),sys.argv[2])
