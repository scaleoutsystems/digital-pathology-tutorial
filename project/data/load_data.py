import pickle

from pathlib import Path

base_path = Path(__file__).parent

def load_processed(version):
    f = open("dataset/processed/"+version, 'rb')
    (x_train, y_train, x_val, y_val) = pickle.load(f)
    return (x_train, y_train, x_val, y_val)