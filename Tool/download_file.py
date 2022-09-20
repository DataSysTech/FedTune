"""
    Download file from remote server

    Must run locally
"""
import os

file_path_on_server = 'Download/mnist'

project_on_server = '/home/dtczhl/FLF'

download_on_local = '/home/dtc/Downloads/'

command = f'scp -r dtczhl@barbel:{project_on_server}/{file_path_on_server} {download_on_local}'

print(command)

os.system(command)
