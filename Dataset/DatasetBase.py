from abc import ABC

from torch.utils.data import Dataset


"""
    Interface for one client only
"""


class DatasetForClient(Dataset, ABC):

    def __init__(self):

        super(DatasetForClient, self).__init__()


"""
    Interface for all clients under train or valid or test folder
"""


class DatasetForSet(Dataset, ABC):

    def __init__(self):

        super(DatasetForSet, self).__init__()

        self.client_ids = None


if __name__ == "__main__":

    print(f' --- debugging {__file__}')
    print(f'\t --- interface, nothing to debug!')

