"""
    calculate mean and std
"""

import glob
import re
import os

from PIL import Image
import numpy as np

from Dataset import DATASET_DIR

target_dir = 'mnist/test'
num_input_feature = 1
feature_index = 0  # only needed if num_input_feature > 1

assert num_input_feature == 1 or num_input_feature == 3

all_user_dir = os.path.join(DATASET_DIR, target_dir)
all_users = os.listdir(all_user_dir)

n = 0
x = 0
x_1 = 0
delta2 = 0
delta2_1 = 0
for i_user, user in enumerate(all_users):
    print(f'{i_user} / {len(all_users)}')
    image_files = glob.glob(os.path.join(all_user_dir, user, '*.jpg'))
    for image_file in image_files:
        if num_input_feature == 1:
            im = np.array(Image.open(image_file).convert('L')) / 255.0  # for grayscale
        elif num_input_feature == 3:
            im = np.array(Image.open(image_file).convert('RGB'))[:, :, feature_index] / 255.0   # for rgb
        for row in im:
            for pixel in row:
                n += 1
                x = x_1 + (pixel - x_1) / n
                delta2 = delta2_1 + ((pixel - x_1) * (pixel - x) - delta2_1) / n

                x_1 = x
                delta2_1 = delta2

print(f'{target_dir}, #input_feature = {num_input_feature}, feature_index = {feature_index}, '
      f'mean = {x}, delta = {np.sqrt(delta2)}')

