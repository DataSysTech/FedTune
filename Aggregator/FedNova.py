import torch
from typing import OrderedDict

import numpy as np
from torch import nn
from torch import Tensor

from Aggregator.AggregatorInterface import AggregatorInterface


class FedNova(AggregatorInterface):

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

            p_i_arr = dict()
            tao_i_arr = dict()

            total_samples = 0
            for client_id in target_client_ids:
                total_samples += all_clients[client_id].get_number_of_data_points()

            p_tao_sum = 0
            for client_id in target_client_ids:
                n_i = all_clients[client_id].get_number_of_data_points()
                p_i_arr[client_id] = n_i / total_samples
                tao_i_arr[client_id] = all_clients[client_id].data_log['training_pass'] * n_i / all_clients[client_id].data_log['batch_size']
                p_tao_sum += p_i_arr[client_id] * tao_i_arr[client_id]

            sd_global = server_model.state_dict()
            for client_id in target_client_ids:
                weight_i = p_i_arr[client_id] * p_tao_sum / tao_i_arr[client_id]
                sd_local_delta = all_clients[client_id].get_model_delta_state_dict()
                for name, param in server_model.named_parameters():
                    sd_global[name] += sd_local_delta[name] * weight_i

        return sd_global




