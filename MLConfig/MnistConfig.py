import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

from Model.ModelWrapper import ModelWrapper
from ServerClient.FLClient import FLClient
from MLConfig.MLConfigInterface import MLConfigInterface

from Metric.Classification import Classification

from Dataset.mnist.MnistDataset import MnistForSet, MnistForClient


class MnistConfig(MLConfigInterface):

    def __init__(self, model_name, dataset_name, gpu_device):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.gpu_device = gpu_device

    def build_fl_clients(self):
        batch_size = 64
        lr = 1.0
        # lr = 0.01
        momentum = 0.9

        all_clients = dict()

        dataset = MnistForSet('train')
        all_client_ids = dataset.client_ids

        for client_id in all_client_ids:
            client_dataset = MnistForClient(client_id=client_id)
            client_dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            client_model = ModelWrapper.build_model_for_dataset(model_name=self.model_name, dataset_name=self.dataset_name)
            client_optimizer = optim.Adadelta(client_model.parameters(), lr=lr)
            # client_optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=momentum)

            client_criterion = nn.CrossEntropyLoss()

            fl_client = FLClient(client_id=client_id, client_dataloader=client_dataloader, client_model=client_model,
                                 client_optimizer=client_optimizer, client_criterion=client_criterion,
                                 gpu_device=self.gpu_device)
            all_clients[client_id] = fl_client

        return all_client_ids, all_clients

    def evaluate_model_performance(self, model: nn.Module) -> float:

        batch_size = 1000
        dataset_test = MnistForSet('test')
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

        n_correct = n_incorrect = 0

        # test
        _n_correct, _n_incorrect = Classification.accuracy(dataloader=dataloader_test, model=model,
                                                           gpu_device=self.gpu_device)
        n_correct += _n_correct
        n_incorrect += _n_incorrect

        return n_correct / (n_correct + n_incorrect)
