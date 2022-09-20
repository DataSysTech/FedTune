"""
    Interface for all ML configs.
"""

import abc
from torch import nn


class MLConfigInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, model_name, dataset_name, gpu_device):
        pass

    @abc.abstractmethod
    def build_fl_clients(self) -> tuple[list[str], dict[str]]:
        pass

    @abc.abstractmethod
    def evaluate_model_performance(self, model: nn.Module) -> float:
        pass
