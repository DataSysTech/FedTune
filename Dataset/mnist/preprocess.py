"""
    Preprocessed data are saved to Download/mnist/
"""

import os
import pathlib

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import skimage.io

# --- Configuration ---
# emulate number of training users
n_train_user = 30

n_test_user = 10  # does not matter
# -------

np.random.seed(321)

# dataset name
dataset_name = 'mnist'

# absolute path to Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to Download/{dataset_name}
dataset_dir = os.path.join(download_dir, dataset_name)
if not os.path.isdir(dataset_dir):
    print(f'Error: dataset directory {dataset_dir} does not exist')
    exit(-1)

# absolute path to Download/{dataset_name}/temp
temp_dir = os.path.join(dataset_dir, 'temp/')
if not os.path.isdir(temp_dir):
    print(f'Error: temp directory {temp_dir} does not exist')
    exit(-1)

# load data
train_image_arr = idx2numpy.convert_from_file(os.path.join(temp_dir, 'train-images-idx3-ubyte'))
train_label_arr = idx2numpy.convert_from_file(os.path.join(temp_dir, 'train-labels-idx1-ubyte'))
test_image_arr = idx2numpy.convert_from_file(os.path.join(temp_dir, 't10k-images-idx3-ubyte'))
test_label_arr = idx2numpy.convert_from_file(os.path.join(temp_dir, 't10k-labels-idx1-ubyte'))

# shuffle data and labels
train_image_arr, train_label_arr = shuffle(train_image_arr, train_label_arr)
test_image_arr, test_label_arr = shuffle(test_image_arr, test_label_arr)

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
if os.path.isdir(train_dir):
    os.system(f'rm -rf {train_dir}')
os.system(f'mkdir {train_dir}')
if os.path.isdir(test_dir):
    os.system(f'rm -rf {test_dir}')
os.system(f'mkdir {test_dir}')


n_train_user_data = len(train_image_arr) // n_train_user
n_test_user_data = len(test_image_arr) // n_test_user
print(f'#train users: {n_train_user}, each has {n_train_user_data} data points')
print(f'#test users: {n_test_user}, each has {n_test_user_data} data points')

global_user_id = 1
global_data_id = 0

# training set
n_user_data = n_train_user_data
for i_user in range(n_train_user):
    user_data = train_image_arr[i_user*n_user_data:(i_user+1)*n_user_data]
    user_label = train_label_arr[i_user*n_user_data:(i_user+1)*n_user_data]

    user_dir = os.path.join(train_dir, str(global_user_id))
    assert not os.path.isdir(user_dir)
    os.makedirs(user_dir)

    for i_img in range(len(user_label)):
        img_out_filename = f'{global_data_id}_{user_label[i_img]}.jpg'
        img_out_file_path = os.path.join(user_dir, img_out_filename)
        img_data = user_data[i_img].astype(np.uint8)
        print(f'saving image to {img_out_file_path}')
        skimage.io.imsave(img_out_file_path, img_data)
        global_data_id += 1

    global_user_id += 1

# test set
n_user_data = n_test_user_data
for i_user in range(n_test_user):
    user_data = test_image_arr[i_user * n_user_data:(i_user + 1) * n_user_data]
    user_label = test_label_arr[i_user * n_user_data:(i_user + 1) * n_user_data]

    user_dir = os.path.join(test_dir, str(global_user_id))
    assert not os.path.isdir(user_dir)
    os.makedirs(user_dir)

    for i_img in range(len(user_label)):
        img_out_filename = f'{global_data_id}_{user_label[i_img]}.jpg'
        img_out_file_path = os.path.join(user_dir, img_out_filename)
        img_data = user_data[i_img].astype(np.uint8)
        print(f'saving image to {img_out_file_path}')
        skimage.io.imsave(img_out_file_path, img_data)
        global_data_id += 1

    global_user_id += 1

# delete temp folder
os.system(f'rm -rf {temp_dir}')

print('Done!')

