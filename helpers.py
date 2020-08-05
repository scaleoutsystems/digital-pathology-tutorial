import tensorflow as tf
import AMLmodel

def load_model():
    # Create AML base model
    model = AMLmodel.ML_model()

    # Load keras model
    f = open('model.json', 'r')
    json_savedModel = f.read() # Load the model architecture
    model_j = tf.keras.models.model_from_json(json_savedModel)
    model_j.load_weights('model_weights.h5') # Load the weights
    model.model = model_j
    
    return model
