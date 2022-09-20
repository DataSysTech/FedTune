"""
    MLConfig for ML training configurations
"""
import sys

import torch
from torch import nn

from MLConfig.MnistConfig import MnistConfig
from MLConfig.SpeechCommandsConfig import SpeechCommandsConfig


class MLConfig:

    def __init__(self, model_name: str, dataset_name: str, gpu_device: torch.device):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.gpu_device = gpu_device

        if self.dataset_name == 'mnist':
            self.ml_config = MnistConfig(
                model_name=self.model_name, dataset_name=self.dataset_name, gpu_device=self.gpu_device)
        elif self.dataset_name == 'speech_commands':
            self.ml_config = SpeechCommandsConfig(
                model_name=self.model_name, dataset_name=self.dataset_name, gpu_device=self.gpu_device)
        else:
            sys.exit(f'{__file__} unknown dataset_name: {self.dataset_name}')

    def build_fl_clients(self) -> tuple[list[str], dict[str]]:
        """

        :return: all_client_ids, all_clients
        """
        return self.ml_config.build_fl_clients()

    def evaluate_model_performance(self, model: nn.Module) -> float:
        """

        :return:
        """

        return self.ml_config.evaluate_model_performance(model)


