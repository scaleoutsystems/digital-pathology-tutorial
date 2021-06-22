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
from models.AMLmodel import AMLModel
#from data.load_data import load_processed
#from data.read_data import read_data
from data.datagenerator import DataGenerator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(model, settings):
    print("-- RUNNING TRAINING --", flush=True)

   # (x_train, y_train, x_val, y_val) = load_processed('data.p')
   # model.fit(x_train, y_train, x_val, y_val, batch_size=8, data_augmentation=False)

    labels_path = 'dataset/processed/data_partitions/partition0/labels.npy' # replace this with the relevant labels path
    data_path = 'dataset/processed/data_partitions/partition0/data_singlets'
    labels = np.load(labels_path, allow_pickle=True).item()
    ids = np.array([label for label in labels])
    np.random.shuffle(ids)

    train_split_index = int(len(ids)*0.9)
    train_ids = ids[:train_split_index]
    val_ids = ids[train_split_index:]

    train_gen = DataGenerator(train_ids, labels,data_path, dim=(100,100), batch_size=32)
    val_gen = DataGenerator(val_ids, labels,data_path, dim=(100,100), batch_size=32)

    model.fit(train_gen, validation_data=val_gen)
    print("-- TRAINING COMPLETED --", flush=True)
    return model

if __name__ == '__main__':

    #with open('settings.yaml', 'r') as fh:
    #    try:
     #       settings = dict(yaml.safe_load(fh))
      #  except yaml.YAMLError as e:
       #     raise(e)

    from fedn.utils.kerashelper import KerasHelper
    helper = KerasHelper()
    weights = helper.load_model(sys.argv[1])

    from models.AMLmodel import construct_model
    model = construct_model()
    model.set_weights(weights)
    settings = []
    model = train(model,settings)
    helper.save_model(model.get_weights(),sys.argv[2])
