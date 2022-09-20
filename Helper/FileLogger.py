"""
    Helper class to log data to file
"""

import sys


class FileLogger:

    def __init__(self, *, file_path: str):
        """ Write data to file

        :param file_path:
        """

        self.file_path = file_path

        try:
            self.file_writer = open(self.file_path, 'w')
        except IOError:
            sys.exit(f'Cannot open {self.file_path} to write')

    def write(self, *, message: str) -> None:
        """ Write message to file

        :param message:
        :return:
        """

        try:
            self.file_writer.write(message)
            self.file_writer.flush()
        except IOError:
            sys.exit(f'Error write message {message}')

    def get_file_path(self) -> str:
        """

        :return:
        """

        return self.file_path

    def close(self) -> None:
        """

        :return:
        """

        try:
            self.file_writer.close()
        except IOError:
            sys.exit(f'Cannot close the file')
