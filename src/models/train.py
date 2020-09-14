import keras
import pickle
import numpy as np
import os

from AMLmodel import ML_model

from src.data.load_data import load_processed


model = ML_model()

(x_train, y_train, x_val, y_val) = load_processed('data.p')

model.fit(x_train, y_train, x_val, y_val, batch_size=8, data_augmentation=False)
model_json = model.model.to_json()
f = open('model.json', 'w')
f.write(model_json)
f.close()
model.model.save_weights('model_weights.h5')