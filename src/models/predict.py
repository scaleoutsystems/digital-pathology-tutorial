import os
import tarfile
import tensorflow as tf
from fastapi import UploadFile

from src.models.AMLmodel import ML_model

def model_load():
    # Should load and return the model.
    # Optional, but if present will be loaded during
    # startup in the "default-python" environment.

    # Create AML base model
    model = ML_model()

    # Load keras model
    # from pathlib import Path
    # base_path = Path(__file__).parent
    # import_folder = str((base_path / "model/models").resolve())
    import_folder = 'models/models'
    files = os.listdir('models/models')
    model_file = ''
    for ff in files:
        if '.tar' in ff:
            model_file = ff
            break
    print(os.path.join(import_folder, model_file))
    
    model_tar = tarfile.TarFile(os.path.join(import_folder, model_file))
    mjson = model_tar.getmember('./model.json')
    mweights = model_tar.getmember('./model_weights.h5')
    js_file = model_tar.extractfile(mjson)
    w_file = model_tar.extractfile(mweights)
    
    tmp_name = 'tmp_weights.h5'
    tmp_f = open(tmp_name, 'wb')
    tmp_f.write(w_file.read())
    tmp_f.close()

    json_savedModel = js_file.read() # Load the model architecture
    model_j = tf.keras.models.model_from_json(json_savedModel)
    model_j.load_weights('tmp_weights.h5') # Load the weights
    model.model = model_j
    os.remove('tmp_weights.h5')
    return model



def model_predict(inp, model=[]):
    # Called by default-python environment.
    # inp -- default is a string, but you can also specify
    # the type in "input_type.py".
    # model is optional and the return value of load_model.
    # Should return JSON.
    res = model.predict(inp)
    return res

# if __name__ == "__main__":
#     from input_type import PredType
#     # cellfile = {'pred_request': open('notebooks/cell.png','rb')}
#     inp = PredType(UploadFile('notebooks/cell.png','rb'))
#     model = model_load()
#     res = model_predict(inp, model)
#     print(res)