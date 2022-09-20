import torch
from typing import OrderedDict

from torch import nn
from torch import Tensor

from Aggregator.AggregatorInterface import AggregatorInterface


class FedAvg(AggregatorInterface):

    def __init__(self, aggregator_name, **kwargs):
        self.aggregator_name = aggregator_name

    def aggregate(self, *, server_model: nn.Module, all_clients: dict[str], target_client_ids: list[str]) -> OrderedDict[str, Tensor]:
        """

        :param server_model:
        :param all_clients:
        :param target_client_ids:
        :return: the caller model can use .load_state_dict() to update the model
        """

        with torch.no_grad():

            total_samples = 0
            for client_id in target_client_ids:
                total_samples += all_clients[client_id].get_number_of_data_points()

            sd_global = server_model.state_dict()
            for name, param in server_model.named_parameters():
                for client_id in target_client_ids:
                    sd_local = all_clients[client_id].get_model_delta_state_dict()
                    sd_global[name] += sd_local[name] * all_clients[client_id].get_number_of_data_points() / total_samples

        return sd_global
