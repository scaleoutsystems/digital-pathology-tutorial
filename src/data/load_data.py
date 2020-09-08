import pickle

from pathlib import Path

base_path = Path(__file__).parent
import_folder = str((base_path / "../../dataset").resolve())


def load_processed(version):
    f = open(import_folder+"/processed/"+version, 'rb')
    (x_train, y_train, x_val, y_val) = pickle.load(f)
    return (x_train, y_train, x_val, y_val)