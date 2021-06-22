from fedn.utils.kerashelper import KerasHelper
from models.AMLmodel import construct_model

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = construct_model()
	outfile_name = "initial_model.npz"

	weights = model.get_weights()
	helper = KerasHelper()
	helper.save_model(weights, outfile_name)
