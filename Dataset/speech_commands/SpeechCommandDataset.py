"""
    Dataset for speech_commands: one client
"""

import glob
import re

from torchvision import transforms
from PIL import Image

from Dataset import *
from Dataset.DatasetBase import DatasetForClient, DatasetForSet

from Dataset.speech_commands import *


class SpeechCommandsForClient(DatasetForClient):

    def __init__(self, *, client_id):
        """ Each client's local data
        :param client_id:
        """

        super(SpeechCommandsForClient, self).__init__()

        self.client_id = client_id
        self.client_dir = f'{DATASET_DIR}/speech_commands/train/{self.client_id}'
        assert os.path.isdir(self.client_dir)
        self.data_files = glob.glob(os.path.join(self.client_dir, '*.jpg'))
        filenames = [os.path.basename(image_file) for image_file in self.data_files]
        # label names
        self.y_label_ids = [int(re.split('[_\.]', x)[1]) for x in filenames]

        self.transforms = transforms.Compose([
            transforms.Resize(SPEECH_COMMANDS_INPUT_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=SPEECH_COMMANDS_TRAIN_MEANS,
                std=SPEECH_COMMANDS_TRAIN_STDS
            )
        ])

    def __len__(self):
        """ Number of local data samples
        :param self:
        :return:
        """

        return len(self.data_files)

    def __getitem__(self, index):
        """ Get one data sample based on index

        :param self:
        :param index:
        :return: the data sample
        """

        file_path = self.data_files[index]

        im = Image.open(file_path).convert('L')
        X = self.transforms(im)
        y = self.y_label_ids[index]

        return X, y


class SpeechCommandsForSet(DatasetForSet):

    def __init__(self, set_name):

        super(SpeechCommandsForSet, self).__init__()

        self.data_files = []
        self.y_label_ids = []

        assert set_name == 'train' or set_name == 'valid' or set_name == 'test'

        self.set_name = set_name

        target_dir = os.path.join(DATASET_DIR, f'speech_commands/{self.set_name}')
        assert os.path.isdir(target_dir)
        self.client_ids = os.listdir(target_dir)

        for client_id in self.client_ids:
            one_client_data_files = glob.glob(os.path.join(target_dir, client_id, '*.jpg'))
            one_client_filenames = [os.path.basename(image_file) for image_file in one_client_data_files]
            one_client_y_label_ids = [int(re.split('[_\.]', x)[1]) for x in one_client_filenames]

            self.data_files.extend(one_client_data_files)
            self.y_label_ids.extend(one_client_y_label_ids)

        self.transforms = transforms.Compose([
            transforms.Resize(SPEECH_COMMANDS_INPUT_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=SPEECH_COMMANDS_TRAIN_MEANS,
                std=SPEECH_COMMANDS_TRAIN_STDS
            )
        ])

    def __len__(self):
        """ total number of data points

        :return:
        """
        return len(self.data_files)

    def __getitem__(self, index):
        """ Get one data sample based on index

                :param self:
                :param index:
                :return: the data sample
                """

        file_path = self.data_files[index]

        im = Image.open(file_path).convert('L')
        X = self.transforms(im)
        y = self.y_label_ids[index]

        return X, y


if __name__ == '__main__':

    print(f' --- debugging {__file__}')

    # debugging SpeechCommandsForClient
    test_client_data = SpeechCommandsForClient(client_id="999")

    print(f'\t\t --- self.data_files: {test_client_data.data_files}')
    print(f'\t\t --- self.y_label_ids: {test_client_data.y_label_ids}')

    print(f'\t --- Client tensor data ---')
    for tensor_data in test_client_data:
        print(tensor_data)

    # debugging SpeechCommandsForSet
    print(f' --- debugging {__file__}')

    set_clients = SpeechCommandsForSet(set_name='test')

    print(f'\t\t --- number of data points: {len(set_clients)}')
    print(f'\t\t --- set_name: {set_clients.set_name}')
    print(f'\t\t --- client ids ({len(set_clients.client_ids)}): {set_clients.client_ids}')


