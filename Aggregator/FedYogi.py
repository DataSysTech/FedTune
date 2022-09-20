import copy

import torch
from typing import OrderedDict

import numpy as np
from torch import nn
from torch import Tensor

from Aggregator.AggregatorInterface import AggregatorInterface


class FedYogi(AggregatorInterface):

    def __init__(self, aggregator_name, **kwargs):
        self.aggregator_name = aggregator_name

        self.delta_t_prv = None
        self.v_prv = None

        self.learning_rate = 0.1

        self.beta1 = 0.0
        self.beta2 = 0.5
        self.tao = 1e-3

    def aggregate(self, *, server_model: nn.Module, all_clients: dict[str], target_client_ids: list[str]) -> OrderedDict[str, Tensor]:
        """

        :param server_model:
        :param all_clients:
        :param target_client_ids:
        :return: the caller model can use .load_state_dict() to update the model
        """

        with torch.no_grad():

            p_i_arr = dict()

            total_samples = 0
            for client_id in target_client_ids:
                total_samples += all_clients[client_id].get_number_of_data_points()

            for client_id in target_client_ids:
                n_i = all_clients[client_id].get_number_of_data_points()
                p_i_arr[client_id] = n_i / total_samples

            x_t = copy.deepcopy(server_model.state_dict())

            delta_t_cur = copy.deepcopy(x_t)
            for name, param in server_model.named_parameters():
                delta_t_cur[name].zero_()

            for client_id in target_client_ids:
                weight_i = p_i_arr[client_id]
                sd_local_delta = all_clients[client_id].get_model_delta_state_dict()
                for name, param in server_model.named_parameters():
                    delta_t_cur[name] += sd_local_delta[name] * weight_i

            delta_t = copy.deepcopy(delta_t_cur)

            if self.delta_t_prv is None:
                self.delta_t_prv = copy.deepcopy(delta_t)
                for name, param in server_model.named_parameters():
                    self.delta_t_prv[name] = 0.0

            for name, param in server_model.named_parameters():
                delta_t[name] = self.beta1 * self.delta_t_prv[name] + (1 - self.beta1) * delta_t_cur[name]
            self.delta_t_prv = delta_t

            if self.v_prv is None:
                self.v_prv = copy.deepcopy(delta_t)
                for name, param in server_model.named_parameters():
                    self.v_prv[name] = self.tao**2

            v_t = copy.deepcopy(self.v_prv)
            for name, param in server_model.named_parameters():
                v_t[name] = self.v_prv[name] \
                            - (1 - self.beta2) * delta_t[name] * delta_t[name] \
                            * torch.sign(self.v_prv[name] - delta_t[name] * delta_t[name])
            self.v_prv = v_t

            x_t_nxt = copy.deepcopy(x_t)
            for name, param in server_model.named_parameters():
                x_t_nxt[name] = x_t[name] + self.learning_rate * delta_t[name]/(torch.sqrt(v_t[name]) + self.tao)

        return x_t_nxt




