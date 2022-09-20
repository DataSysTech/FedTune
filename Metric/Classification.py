import numpy as np
import torch

class Classification:

    def __init__(self):
        pass

    @staticmethod
    def accuracy(*, top: int = 1, dataloader, model, gpu_device):

        # leave it as future work to support top
        assert top == 1

        cpu_device = torch.device('cpu')

        model.to(gpu_device)

        n_correct = n_incorrect = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(gpu_device), labels.to(gpu_device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            labels = labels.detach().cpu().numpy()
            predicted = predicted.detach().cpu().numpy()

            _n_correct = (predicted == labels).sum().item()

            n_correct += _n_correct
            n_incorrect += len(inputs) - _n_correct

        model.to(cpu_device)

        return n_correct, n_incorrect
