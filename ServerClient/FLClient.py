"""
    For one client
"""

import sys
from typing import Any

import copy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader


class FLClient:

    def __init__(self, *, client_id: str, client_dataloader: DataLoader, client_model: nn.Module, client_optimizer: optim,
                 client_criterion: nn.modules.loss, gpu_device: torch.device, **kwargs: Any):
        """ Client: model, and local data
        :param client_id: the id of this client
        :param client_dataloader:
        :param client_model: client's model
        :param client_optimizer:
        :param client_criterion
        :param gpu_device: run on which cuda
        :param kwargs: other parameters
        """

        assert client_id.isdigit()

        # member variables
        self.client_id = client_id
        self.client_model = client_model
        self.client_model_delta = None
        self.client_dataloader = client_dataloader

        self.cpu_device = torch.device('cpu')
        self.gpu_device = gpu_device

        """
            Training construction
        """

        self.optimizer = client_optimizer
        self.criterion = client_criterion

        # information for logging
        self.data_log = dict()

    def train_one_round(self, *, training_pass: float = 1.0, **kwargs: Any) -> None:
        """ Train the client for one training round

        :param training_pass
        :param kwargs: for extension
        :return:
        """

        if training_pass <= 0:
            sys.exit(f'#training_pass must > 0, given {training_pass}')

        self.client_model_delta = copy.deepcopy(self.client_model)

        # ------ Project related code ------

        client_config = kwargs.get('client_config', dict())

        # percentage of data used for, for the CrossSilo Project
        data_percentage = client_config.get('data_percentage', 1.0)
        training_pass *= data_percentage

        # training pass changed to T/ni, for the FedNova aggregator
        if 'T' in client_config:
            training_pass = client_config['T'] / self.get_number_of_data_points()

        tot_training_data_point_count = int(training_pass * len(self.client_dataloader))
        # at least one training data point
        tot_training_data_point_count = max(tot_training_data_point_count, 1)

        # data info for the current round
        self.data_log = {
            'training_pass': training_pass,
            'round_train_data_point_count': tot_training_data_point_count,
            'batch_size': self.client_dataloader.batch_size
        }

        self.client_model.to(self.gpu_device)

        while tot_training_data_point_count > 0:
            for inputs, labels in self.client_dataloader:

                # break if no need to train more
                if tot_training_data_point_count <= 0:
                    break

                if tot_training_data_point_count <= len(inputs):
                    inputs = inputs[:tot_training_data_point_count]
                    labels = labels[:tot_training_data_point_count]
                    tot_training_data_point_count -= len(inputs)
                    assert tot_training_data_point_count == 0

                tot_training_data_point_count -= len(inputs)

                inputs = inputs.to(self.gpu_device)
                labels = labels.to(self.gpu_device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.client_model(inputs)
                loss = self.criterion(outputs, labels)

                # gradients here
                loss.backward()

                self.optimizer.step()

        # put clients to CPU when not active
        self.client_model.to(self.cpu_device)

        # calculate delta
        sd_model = self.client_model.state_dict()
        sd_model_delta = self.client_model_delta.state_dict()
        for name, param in self.client_model.named_parameters():
            sd_model_delta[name] = sd_model[name] - sd_model_delta[name]
        self.client_model_delta.load_state_dict(state_dict=sd_model_delta)

        # remove GPU cache
        torch.cuda.empty_cache()

    def replace_model_from_server(self, *, server_model: nn.Module) -> None:
        """ Replace client's model with server model

        :param server_model:
        :return:
        """

        self.client_model.load_state_dict(server_model.state_dict())

    def get_model_delta_state_dict(self) -> dict:
        """ Return this client's state_dict of its model's delta
            For server model aggregation

        :return:
        """

        return self.client_model_delta.state_dict()

    def get_model_state_dict(self) -> dict:
        """ Return this client's state_dict of its model
            For server model aggregation

        :return:
        """

        return self.client_model.state_dict()

    def get_number_of_data_points(self) -> int:
        """ Return the number of its local sample points

        :return:
        """

        return len(self.client_dataloader.dataset)

    def get_data_log(self) -> dict[str]:
        """

        """

        return self.data_log


if __name__ == '__main__':

    print(f' --- debugging {__file__}')

    from Model.ResNet import ResNet
    from Dataset.speech_commands.SpeechCommandDataset import SpeechCommandsForClient

    client_id = "999"
    client_dataset = SpeechCommandsForClient(client_id=client_id)
    client_dataloader = DataLoader(client_dataset, batch_size=5, shuffle=True)
    client_model = ResNet(num_input_feature=1, num_classes=35, depth=10)
    server_model = ResNet(num_input_feature=1, num_classes=35, depth=10)
    client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    client_criterion = nn.CrossEntropyLoss()
    gpu_device = torch.device('cuda:0')

    fl_client = FLClient(client_id=client_id, client_dataloader=client_dataloader, client_model=client_model,
                         client_optimizer=client_optimizer, client_criterion=client_criterion, gpu_device=gpu_device)
    print(f'\t --- number of data points: {fl_client.get_number_of_data_points()}')

    print(f'\t --- train one round')

    fl_client.replace_model_from_server(server_model=server_model)
    fl_client.train_one_round()

    sd_local = fl_client.get_model_state_dict()
    sd_global = server_model.state_dict()
    with torch.no_grad():
        for name, param in server_model.named_parameters():
            sd_global[name] = sd_local[name]
    server_model.load_state_dict(state_dict=sd_global)

    print(f'\t --- train another round')

    fl_client.replace_model_from_server(server_model=server_model)
    fl_client.train_one_round()





