"""
    FL Server

    Controls all
"""

import torch

from Model.ModelWrapper import ModelWrapper
from MLConfig.MLConfig import MLConfig
from Aggregator.AggregatorController import AggregatorController


class FLServer:

    def __init__(self, *, model_name: str, dataset_name: str, aggregator_name: str,
                 gpu_device: torch.device = torch.device('cuda:0'), **kwargs):
        """

        :param model_name:
        :param dataset_name:
        :param kwargs:
        """

        self.model_name = model_name.strip().lower()
        self.dataset_name = dataset_name.strip().lower()
        self.aggregator_name = aggregator_name.strip().lower()

        self.server_model = ModelWrapper.build_model_for_dataset(model_name=model_name, dataset_name=dataset_name)
        self.gpu_device = gpu_device

        self.ml_config = MLConfig(dataset_name=self.dataset_name, model_name=self.model_name, gpu_device=self.gpu_device)
        self.all_client_ids, self.all_clients = self.ml_config.build_fl_clients()

        self.aggregator = AggregatorController(aggregator_name=self.aggregator_name)

    def replace_client_model_with_server(self, *, target_client_ids: list[str]) -> None:
        """ Replace client's model with the server's

        :param target_client_ids:
        :return:
        """

        with torch.no_grad():
            for client_id in target_client_ids:
                self.all_clients[client_id].replace_model_from_server(server_model=self.server_model)

    def update_server_model_from_clients(self, *, target_client_ids: list[str]) -> None:
        """ FedAvg. will extend to others

        :param target_client_ids:
        :return: model's state_dict that the global model can directly use via global_model.load_state_dict()
        """

        sd_global = self.aggregator.aggregate(
            server_model=self.server_model, all_clients=self.all_clients, target_client_ids=target_client_ids)

        self.server_model.load_state_dict(sd_global)

    def train_one_round(self, *, target_client_ids: list[str], training_pass: float = 1.0, **kwargs) -> None:
        """ Train the models of target_client_ids for one round

        :param target_client_ids:
        :param training_pass:
        :param kwargs:
        :return:
        """

        client_config = kwargs.get('client_config', dict())

        for client_id in target_client_ids:
            self.all_clients[client_id].train_one_round(training_pass=training_pass, client_config=client_config.get(client_id, dict()))

    def evaluate_model_performance(self) -> float:
        """ evaluate server model performance

        :return: accuracy
        """

        self.server_model.to(self.gpu_device)

        return self.ml_config.evaluate_model_performance(self.server_model)

    def get_client_data_log(self, *, target_client_ids: list[str]) -> dict[str]:

        selected_client_data_logs = dict()

        for client_id in target_client_ids:
            selected_client_data_logs[client_id] = self.all_clients[client_id].get_data_log()

        return selected_client_data_logs


if __name__ == '__main__':

    print(f' --- debugging {__file__}')

    dataset_name = 'speech_commands'
    model_name = 'resnet_10'
    aggregator_name = 'FedAvg'
    gpu_device = torch.device('cuda:7')

    fl_server = FLServer(dataset_name=dataset_name, model_name=model_name, aggregator_name=aggregator_name, gpu_device=gpu_device)

    # training
    print('\t --- training')
    for _ in range(10):
        fl_server.replace_client_model_with_server(target_client_ids=fl_server.all_client_ids[:10])
        fl_server.train_one_round(target_client_ids=fl_server.all_client_ids[:10])
        fl_server.update_server_model_from_clients(target_client_ids=fl_server.all_client_ids[:10])
        fl_server.get_client_data_log(target_client_ids=fl_server.all_client_ids[:10], project_name='FedTune')
        print(f'\t\t --- accuracy: {fl_server.evaluate_model_performance()}')




        

