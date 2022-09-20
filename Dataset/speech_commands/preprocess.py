"""
    Preprocessed data are saved to Download/speech_commands/
"""

import os
import pathlib
import glob
import re

import numpy as np
import skimage.io
import librosa


# dataset name
dataset_name = 'speech_commands'

# labels
labels = ['up', 'two', 'sheila', 'zero', 'yes', 'five', 'one', 'happy', 'marvin', 'no',
          'go', 'seven', 'eight', 'tree', 'stop', 'down', 'forward', 'learn', 'house', 'three',
          'six', 'backward', 'dog', 'cat', 'wow', 'left', 'off', 'on', 'four', 'visual',
          'nine', 'bird', 'right', 'follow', 'bed']

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

# users for validation and testing, the remains are for training
validation_list_file = 'validation_list.txt'
testing_list_file = 'testing_list.txt'

# user ids for train, valid, and test
#   set() is better than dict{} in this case, but anyway, it works...
user_train_dict = {}
user_valid_dict = {}
user_test_dict = {}

# Users for validation
with open(os.path.join(temp_dir, validation_list_file), 'r') as f_in:
    lines = f_in.readlines()
    for line in lines:
        user_id = re.split('[/_]', line)[-3]
        user_valid_dict[user_id] = user_valid_dict.get(user_id, 0) + 1

# Users for testing
with open(os.path.join(temp_dir, testing_list_file), 'r') as f_in:
    lines = f_in.readlines()
    for line in lines:
        user_id = re.split('[/_]', line)[-3]
        user_test_dict[user_id] = user_valid_dict.get(user_id, 0) + 1

# Users for training
root_dir, data_dirs = next(os.walk(temp_dir))[:2]
for data_dir in data_dirs:

    # ignore _background_noise_ folder
    if data_dir.startswith("_"):
        continue

    data_dir_path = os.path.join(root_dir, data_dir)
    wav_files = glob.glob(os.path.join(data_dir_path, '*.wav'))

    for wav_file in wav_files:
        user_id = re.split('[/_]', wav_file)[-3]
        if user_id in user_valid_dict or user_id in user_test_dict:
            continue
        else:
            user_train_dict[user_id] = user_train_dict.get(user_id, 0) + 1

# change working directory to {dataset_dir}
os.chdir(f'{dataset_dir}')
if os.path.isdir('train'):
    os.system('rm -rf train')
if os.path.isdir('test'):
    os.system('rm -rf test')
if os.path.isdir('valid'):
    os.system('rm -rf valid')
# create folders for train, valid, and test
os.system('mkdir train test valid')

# Pre-processing happens here: convert wav to spectrogram and then image, save to corresponding train, valid, and test
img_id = 0
user_id_dict = {}  # reset user id starting from 1
root_dir, data_dirs = next(os.walk(temp_dir))[:2]
for data_dir in data_dirs:

    # save spectrogram to image
    def spectrogram_image(y, sr, out, hop_length, n_mels):
        def scale_minmax(X, min=0.0, max=1.0):
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max - min) + min
            return X_scaled

        # use log-melspectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=hop_length * 2, hop_length=hop_length)
        mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mels, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img = 255 - img  # invert. make black==more energy

        # save as jpg
        skimage.io.imsave(out, img)

    # ignore _background_noise_ folder
    if data_dir.startswith("_"):
        continue

    data_dir_path = os.path.join(root_dir, data_dir)
    wav_files = glob.glob(os.path.join(data_dir_path, "*.wav"))

    for wav_file in wav_files:
        wav_file_split = re.split('[/_\.]', wav_file)

        user_original_id = wav_file_split[-4]
        if user_original_id not in user_id_dict:
            user_id_dict[user_original_id] = str(len(user_id_dict)+1)  # user id starts from 1
        user_id = user_id_dict[user_original_id]
        label_id = labels.index(wav_file_split[-5])

        image_out_filename = f'{img_id}_{label_id}.jpg'

        n_mels = 64
        n_time_steps = 63
        hop_length = 256

        y, sr = librosa.load(wav_file, sr=16000)
        y = np.concatenate((y[:n_time_steps * hop_length], [0] * (n_time_steps * hop_length - len(y))))

        # main workload
        if user_original_id in user_train_dict:
            user_dir = os.path.join(dataset_dir, 'train', user_id)
            if not os.path.isdir(user_dir):
                os.system('mkdir {}'.format(user_dir))
            img_file = os.path.join(user_dir, image_out_filename)
            print(f'{wav_file} -> {img_file}')
            spectrogram_image(y, sr=sr, out=img_file, hop_length=hop_length, n_mels=n_mels)
        elif user_original_id in user_valid_dict:
            user_dir = os.path.join(dataset_dir, 'valid', user_id)
            if not os.path.isdir(user_dir):
                os.system('mkdir {}'.format(user_dir))
            img_file = os.path.join(user_dir, image_out_filename)
            print(f'{wav_file} -> {img_file}')
            spectrogram_image(y, sr=sr, out=img_file, hop_length=hop_length, n_mels=n_mels)
        else:
            user_dir = os.path.join(dataset_dir, 'test', user_id)
            if not os.path.isdir(user_dir):
                os.system('mkdir {}'.format(user_dir))
            img_file = os.path.join(user_dir, image_out_filename)
            print(f'{wav_file} -> {img_file}')
            spectrogram_image(y, sr=sr, out=img_file, hop_length=hop_length, n_mels=n_mels)

        img_id += 1

# delete temp folder
os.system(f'rm -rf {temp_dir}')

print('Done!')
