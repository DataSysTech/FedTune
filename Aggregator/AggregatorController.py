import sys
from typing import OrderedDict

from torch import nn
from torch import Tensor

from Aggregator.FedAvg import FedAvg
from Aggregator.FedNova import FedNova
from Aggregator.FedAdagrad import FedAdagrad
from Aggregator.FedYogi import FedYogi
from Aggregator.FedAdam import FedAdam


class AggregatorController:

    def __init__(self, aggregator_name: str):

        self.aggregator_name = aggregator_name.strip().lower()

        if self.aggregator_name == 'fedavg':
            self.aggregator = FedAvg(aggregator_name=self.aggregator_name)
        elif self.aggregator_name == 'fednova':
            self.aggregator = FedNova(aggregator_name=self.aggregator_name)
        elif self.aggregator_name == 'fedadagrad':
            self.aggregator = FedAdagrad(aggregator_name=self.aggregator_name)
        elif self.aggregator_name == 'fedyogi':
            self.aggregator = FedYogi(aggregator_name=self.aggregator_name)
        elif self.aggregator_name == 'fedadam':
            self.aggregator = FedAdam(aggregator_name=self.aggregator_name)
        else:
            sys.exit(f'{__file__} unknown aggregator_name: {self.aggregator_name}')

    def aggregate(self, *, server_model: nn.Module, all_clients: dict[str], target_client_ids: list[str]) -> OrderedDict[str, Tensor]:
        """

        :param server_model:
        :param all_clients:
        :param target_client_ids:
        :return:
        """

        return self.aggregator.aggregate(
            server_model=server_model, all_clients=all_clients, target_client_ids=target_client_ids)