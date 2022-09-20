"""
    Interface for all FL aggregation algorithms
"""

import abc
from typing import OrderedDict

from torch import nn
from torch import Tensor


class AggregatorInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, *, aggregator_name, **kwargs):
        pass

    @abc.abstractmethod
    def aggregate(self, *, server_model: nn.Module, all_clients: dict[str], target_client_ids: list[str]) -> OrderedDict[str, Tensor]:
        pass

