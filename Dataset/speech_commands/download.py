"""
    Downloaded files are saved to Download/speech_commands/temp/
"""

import os
import pathlib

# dataset name
dataset_name = 'speech_commands'

# url to download the speech-to-command dataset
dataset_url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
# compressed file
compressed_filename = 'speech_commands_v0.02.tar.gz'

# absolute path to FLF/Download/
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
os.system(f'wget {dataset_url}')
# uncompress dataset
os.system(f'tar -xvf {compressed_filename}')
# remove the compressed file
os.system(f'rm {compressed_filename}')

print('Done!')
