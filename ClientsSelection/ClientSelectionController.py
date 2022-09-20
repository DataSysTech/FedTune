import sys

from ServerClient.FLServer import FLServer

from ClientsSelection.RandomClientSelection import RandomClientSelection


class ClientSelectionController:

    def __init__(self, *, fl_server: FLServer, client_selection_method_name: str):
        """

        """

        self.client_selection_method_name = client_selection_method_name.strip().lower()
        self.fl_server = fl_server

        if self.client_selection_method_name == 'random':
            self.client_selection = RandomClientSelection(
                client_selection_method_name=self.client_selection_method_name, fl_server=self.fl_server)
        else:
            sys.exit(f'{__file__} unknown client_selection_method_name: {self.client_selection_method_name}')

    def select(self, num_target_clients: int) -> list[str]:
        """

        """

        return self.client_selection.select(num_target_clients=num_target_clients)








