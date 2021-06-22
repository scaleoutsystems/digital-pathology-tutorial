from __future__ import print_function
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.models as krm
import pickle
import yaml
import numpy as np
from data.datagenerator import DataGenerator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, settings):
    print("-- RUNNING TRAINING --", flush=True)

    labels_path = '../data/labels.npy' # replace this with the relevant labels path
    data_path = '../data/data_singlets'
    labels = np.load(labels_path, allow_pickle=True).item()
    ids = np.array([label for label in labels])
    np.random.shuffle(ids)

    try:
        train_ids = np.load('trainids.npy')
        val_ids = np.load('valids.npy')
    except:
        train_split_index = int(len(ids)*0.9)
        train_ids = ids[:train_split_index]
        val_ids = ids[train_split_index:]

        np.save('trainids.npy', train_ids)
        np.save('valids.npy', val_ids)


    train_gen = DataGenerator(train_ids, labels,data_path, dim=(100,100), batch_size=32)
    val_gen = DataGenerator(val_ids, labels,data_path, dim=(100,100), batch_size=32)

    model.fit(train_gen, validation_data=val_gen)
    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    from models.AMLmodel import construct_model
    model = construct_model()
    model.set_weights(weights)
    settings = []
    model = train(model,settings)
    helper.save_model(model.get_weights(),sys.argv[2])
