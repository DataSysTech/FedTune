"""
    Client Selection Interface
"""

import abc

from ServerClient.FLServer import FLServer


class ClientSelectionInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, client_selection_method_name: str, fl_server: FLServer):
        pass

    @abc.abstractmethod
    def select(self, num_target_clients: int) -> list[str]:
        pass


class ClientSelectionData:
    """
        Data structure for client selection
    """

    def __init__(self, active: bool, score: float):
        # whether included for client selection
        self.active = active
        # the higher, the better
        self.score = score

    def is_active(self) -> bool:

        return self.active

