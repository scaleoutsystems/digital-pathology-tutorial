import os
import numpy as np
from PIL import Image
from matplotlib import pylab as plt
from scipy import misc
from utilities import split_train_test_set
import pickle

WIDTH, HEIGHT = 100, 100

class_keys = {'BAS': 0, 'EBO': 1, 'EOS': 2, 'KSC': 3, 'LYA': 4, 'LYT': 5, 'MMZ': 6, 'MOB': 7, 'MON': 8, 'MYB': 9 , 'MYO': 10,
        'NGB': 11, 'NGS': 12, 'PMB': 13, 'PMO': 14}

inv_class_keys = {v: k for k, v in class_keys.items()}


main_folder = "AML-Cytomorphology_LMU"

def load_data(roof=100,downsample=False):

    first = True
    image_data = []
    class_data = []
    #class_index = 0
    for filename in os.listdir(main_folder):
        class_index = class_keys[filename]
        print("class index: ", class_index)
        nr = 0
        for image in os.listdir(main_folder + "/" + filename):
            nr += 1
            im = Image.open(main_folder + "/" + filename + "/" + image)

            im = im.resize(0.25)
            #np_image = np.array(im)
            #np_image = misc.imresize(np_image, 0.25)
            np_image = np.array(im)
            image_data += [np_image]
            class_data += [class_index]
            if roof:
                if nr >= roof:
                    break
            if first:
                plt.imshow(np_image)
                plt.show()
                print("np_image shape: ", np_image.shape)

                first = False

        print(filename, " got ", nr, " sets.")
        #class_index += 1

    return np.array(image_data), np.array(class_data)

#creating an index list with names of the images divided into 11 buckets (10 members and 1 test set)
def create_split_index(buckets=11):

    name_dict = {}
    for filename in os.listdir(main_folder):
        class_index = class_keys[filename]
        print("class index: ", class_index)
        image_data = []
        class_data = []
        image_names = []
        nr = 0
        for image in os.listdir(main_folder + "/" + filename):
            nr += 1
            im = Image.open(main_folder + "/" + filename + "/" + image)
            np_image = np.array(im)
            np_image = misc.imresize(np_image, 0.25)
            image_data += [np_image]
            class_data += [class_index]
            image_names += [image]


        print(filename, " got ", nr, " sets.")
        splits = np.int32(np.round(np.linspace(0, nr, buckets+1)))

        image_names_splits = np.split(image_names,splits[1:-1])


        inner_dict = {}
        for i in range(11):
            for im in image_names_splits[i]:
                inner_dict[im] = i

        name_dict[filename] = inner_dict

    pickle.dump(name_dict, open('name_dict.p', "wb"))


def load_split_set():
    image_data = [[]]*11
    class_data = [[]]*11
    #print( pickle.load(open('name_dict.p', "rb")))
    outer_dict = pickle.load(open('name_dict.p', "rb"))

    for filename in os.listdir(main_folder):
        print("filename: ", filename)
        # print("inner_dict: ", inner_dict)
        if os.path.isdir(os.path.join(main_folder,filename)):
            inner_dict = outer_dict[filename]
            for image in os.listdir(main_folder + "/" + filename):
                #print("image: ", image)
                ind = inner_dict[image]
                if not image_data[ind]:

                    im = Image.open(main_folder + "/" + filename + "/" + image)
                    #print("image: ", image)
                    im = im.resize((WIDTH, HEIGHT))
                    # np_image = np.array(im)
                    # np_image = misc.imresize(np_image, 0.25)
                    np_image = np.array(im)
                    image_data[ind] = [np_image]
                    class_data[ind] = [class_keys[filename]]
                else:
                    im = Image.open(main_folder + "/" + filename + "/" + image)
                    #print("image: ", image)
                    im = im.resize((WIDTH, HEIGHT))
                    np_image = np.array(im)
                    image_data[ind] += [np_image]
                    class_data[ind] += [class_keys[filename]]

    return image_data, class_data









# image_data, class_data = load_data(roof=20)
# print("image data shape: ", image_data.shape)
# print("class data shape: ", class_data.shape)
#
# train_x, train_y, test_x, test_y = split_train_test_set(image_data, class_data, split=0.1)

def check_class_nr(class_data):

    classes, return_counts = np.unique(class_data, return_counts = True)
    print("nr of classes: ", len(classes), ", nr of data points: ", len(class_data))
    for i in range(len(classes)):
        #print("classes[i]: ", classes[i])
        print("class ", inv_class_keys[classes[i]], ": ", return_counts[i])


# print("train check: ")
# check_class_nr(train_y)
# print("test check: ")
# check_class_nr(test_y)

#create_split_index()

# outer_dict = pickle.load(open('name_dict.p', "rb"))
# print("outer_dict: ", main_dict)
# image_data, class_data = load_split_set()
#
# for i in range(len(class_data)):
#     print("bucket: ", i)
#     check_class_nr(class_data[i])
