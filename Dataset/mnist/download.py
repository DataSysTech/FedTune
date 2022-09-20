"""
    Downloaded files are saved to Download/mnist/temp/
"""

import os
import pathlib

# dataset name
dataset_name = 'mnist'

# url to download the dataset
train_image_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
train_image_filename = 'train-images-idx3-ubyte.gz'
train_label_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
train_label_filename = 'train-labels-idx1-ubyte.gz'
test_image_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
test_image_filename = 't10k-images-idx3-ubyte.gz'
test_label_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
test_label_filename = 't10k-labels-idx1-ubyte.gz'

# absolute path to FedTuning/Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to the dataset directory
dataset_dir = os.path.join(download_dir, dataset_name)

# absolute path to the temporary directory for data
temp_dir = os.path.join(dataset_dir, 'temp/')

# switch to {download_dir}
os.chdir(download_dir)

if os.path.isdir(dataset_dir):
    # remove {dataset_dir} if exists
    os.system(f'rm -rf {dataset_dir}')

# create {dataset_name} directory
os.mkdir(dataset_name)
# create {temp_dir} directory
os.mkdir(temp_dir)
# switch to {temp_dir}
os.chdir(temp_dir)
# download dataset
os.system(f'wget {train_image_url}')
os.system(f'wget {train_label_url}')
os.system(f'wget {test_image_url}')
os.system(f'wget {test_label_url}')
# uncompress dataset
os.system(f'gunzip {train_image_filename}')
os.system(f'gunzip {train_label_filename}')
os.system(f'gunzip {test_image_filename}')
os.system(f'gunzip {test_label_filename}')

print('Done!')
