import numpy as np
import os
import sys
from PIL import Image
import shutil

if __name__ == '__main__':

    nr_of_partitions = int(sys.argv[1])

    #amldata = r'dataset/raw/AML-Cytomorphology/'
    amldata = r'/home/jovyan/work/minio-vol/data/AML-Cytomorphology'
    classes = {}
    class_index = 0
    class_sizes = {}
    for c in os.listdir(amldata):
        classes[c] = class_index
        class_index += 1
        dir_ = os.path.join(amldata, c)
        class_size = len(os.listdir(dir_))
        class_sizes[c] = class_size



    singlet_path = os.path.join('dataset','processed','data_singlets')
    if not os.path.isdir(singlet_path):
        os.makedirs(singlet_path)

    ids_ = []
    for c in os.listdir(amldata):
        for x in os.listdir(os.path.join(amldata,c)):
            id_name = x.split(".")[0]
            sample = Image.open(os.path.join(amldata, c, x)).resize((100, 100)) # remove resize if using original data shape
            sample = (np.array(sample)/255).astype('float32')
            np.save(os.path.join(singlet_path, id_name + '.npy'), sample)
            ids_ += [id_name]

    # Move files into iid distributed nr_of_partitions folders
    inds = np.arange(len(ids_))
    np.random.shuffle(inds)

    used = 0
    partitions = {}
    for i in range(nr_of_partitions):
        t = (len(inds) - used) // (nr_of_partitions - i)
        print(t)
        partitions[i] = [used, used+t]
        used += t

    data_partitions_path = os.path.join('dataset', 'processed', 'data_partitions')
    if not os.path.isdir(data_partitions_path):
        os.makedirs(data_partitions_path)

    for p in partitions:
        partition_p_path = os.path.join('dataset', 'processed', 'data_partitions', 'partition' + str(p))

        if not os.path.isdir(partition_p_path):
            os.makedirs(partition_p_path)

        singlet_p = os.path.join(partition_p_path, 'data_singlets')

        if not os.path.isdir(singlet_p):
            os.makedirs(singlet_p)

        labels = {}
        for s in inds[partitions[p][0]:partitions[p][1]]:
            file = os.path.join(singlet_path,ids_[s] + '.npy')
            shutil.move(file, singlet_p)
            labels[ids_[s]] = classes[ids_[s][:3]]
        np.save(os.path.join(partition_p_path, 'labels.npy'), labels)










