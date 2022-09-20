
import random

import torch

from ServerClient.FLServer import FLServer

from ClientsSelection.ClientSelectionInterface import ClientSelectionInterface, ClientSelectionData


class RandomClientSelection(ClientSelectionInterface):

    def __init__(self, client_selection_method_name: str, fl_server: FLServer):
        self.client_selection_method_name = client_selection_method_name
        self.fl_server = fl_server
        self.all_client_ids = self.fl_server.all_client_ids

        self.client_selection_score = dict()

        for client_id in self.all_client_ids:
            self.client_selection_score[client_id] = ClientSelectionData(active=True, score=random.random())

    def select(self, num_target_clients: int) -> list[str]:

        for client_id in self.client_selection_score.keys():
            if self.client_selection_score[client_id].is_active():
                self.client_selection_score[client_id].score = random.random()

        client_selection_values = {client_id: self.client_selection_score[client_id]
                                   for client_id in self.client_selection_score.keys()
                                   if self.client_selection_score[client_id].is_active()}

        sorted_client_ids = [k for k, v in sorted(client_selection_values.items(),
                                                  key=lambda item: item[1].score, reverse=True)]

        selected_client_ids = sorted_client_ids[:num_target_clients]

        return selected_client_ids


if __name__ == '__main__':

    print(f' --- debugging {__file__}')

    dataset_name = 'speech_commands'
    model_name = 'resnet_10'
    aggregator_name = 'FedAvg'
    gpu_device = torch.device('cuda:7')

    fl_server = FLServer(dataset_name=dataset_name, model_name=model_name, aggregator_name=aggregator_name, gpu_device=gpu_device)

    client_selection = RandomClientSelection(client_selection_method_name='random', fl_server=fl_server)

    print(client_selection.select(num_target_clients=10))
    print(client_selection.select(num_target_clients=10))
