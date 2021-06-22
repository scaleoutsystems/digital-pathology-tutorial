from __future__ import print_function
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow.keras.models as krm
import pickle
import yaml
import json
import numpy as np
from models.AMLmodel import AMLModel
#from data.load_data import load_processed
#from data.read_data import read_data
from data.datagenerator import DataGenerator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def validate(model, settings):
    print("-- RUNNING VALIDATION --", flush=True)


    labels_path = 'dataset/processed/data_partitions/partition0/labels.npy' # replace this with the relevant labels path
    data_path = 'dataset/processed/data_partitions/partition0/data_singlets'
    labels = np.load(labels_path, allow_pickle=True).item()


    try:
        train_ids = np.load('trainids.npy')
        val_ids = np.load('valids.npy')
    except:
        ids = np.array([label for label in labels])
        np.random.shuffle(ids)
        train_split_index = int(len(ids)*0.9)
        train_ids = ids[:train_split_index]
        val_ids = ids[train_split_index:]

        np.save('trainids.npy', train_ids)
        np.save('valids.npy', val_ids)


    train_gen = DataGenerator(train_ids, labels,data_path, dim=(100,100), batch_size=32)
    val_gen = DataGenerator(val_ids, labels,data_path, dim=(100,100), batch_size=32)

    model_score = model.evaluate(train_gen)
    model_score_test = model.evaluate(val_gen)

    report = {
        "classification_report": '',
        "training_loss": model_score[0],
        "training_accuracy": model_score[1],
        "test_loss": model_score_test[0],
        "test_accuracy": model_score_test[1],
    }
    print("-- VALIDATION COMPLETED --", flush=True)
    return report

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
    report = validate(model, settings)
    with open(sys.argv[2],"w") as fh:
        fh.write(json.dumps(report))