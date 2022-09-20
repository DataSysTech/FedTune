"""
    Wrapper class.

    Based on dataset and model structure, it selects the correct model for it.
"""

import re
import sys

from torch import nn


from Model.ResNet import ResNet
from Model.MnistNet import MnistNet


class ModelWrapper:

    @staticmethod
    def build_model_for_dataset(*, model_name: str, dataset_name: str) -> nn.Module:

        model_name = model_name.strip().lower()
        dataset_name = dataset_name.strip().lower()

        n_target_class = -1
        n_input_feature = -1

        if dataset_name == 'speech_commands':
            n_target_class = 35
            n_input_feature = 1
        elif dataset_name == 'mnist':
            n_target_class = 10
            n_input_feature = 1
        else:
            sys.exit(f'Unknown dataset_name: {dataset_name}')

        model_details = re.split('[_]', model_name)

        if model_details[0] == 'resnet':
            n_layers = int(model_details[1])
            return ResNet(depth=n_layers, num_input_feature=n_input_feature, num_classes=n_target_class)
        elif model_details[0] == 'mnistnet':
            return MnistNet()
        else:
            sys.exit(f'Unknown model_name: {model_name}')


if __name__ == '__main__':

    print(f' --- debugging {__file__}')

    from torchinfo import summary

    model_name = 'resnet_10'
    dataset_name = 'speech_commands'

    model = ModelWrapper.build_model_for_dataset(model_name=model_name, dataset_name=dataset_name)

    batch_size = 1
    summary(model, input_size=(batch_size, 1, 32, 32))


