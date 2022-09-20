"""
    Overall performance. Factor 10 vs 1
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Project.FedTune.ResultAnalysis.ReadTrace import read_traces_of_preference_penalty


# ------ Configurations ------

dataset_name = 'speech_commands'
model_name = 'resnet_10'
aggregator_name = 'fedadagrad'
initial_M = 20
initial_E = 20
penalty_arr = [10]
trace_id_arr = [1, 2, 3]

# --- End of Configuration ---


# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(project_dir)

preference_combine_all = [
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 1, 1, 0),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 1)
]
preference_combine_all = np.array(preference_combine_all).astype(float)
for i_row in range(len(preference_combine_all)):
    row_sum = sum(preference_combine_all[i_row])
    if row_sum > 0:
        preference_combine_all[i_row] /= row_sum
float_2_decimal = lambda x: float('{:.2f}'.format(x))
vfunc = np.vectorize(float_2_decimal)
preference_combine_all = vfunc(preference_combine_all)

result_map = {}

baseline = read_traces_of_preference_penalty(aggregator_name=aggregator_name,
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=(0, 0, 0, 0), penalty=10, trace_id_arr=trace_id_arr)


for preference in preference_combine_all:

    for penalty in penalty_arr:
        trace_result = read_traces_of_preference_penalty(aggregator_name=aggregator_name,
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=preference, penalty=penalty, trace_id_arr=trace_id_arr, baseline=baseline)
        map_key = np.append(preference, [penalty])
        result_map[tuple(map_key)] = trace_result

mean_improve_ratio = []
std_improve_ratio = []
for k, v in result_map.items():
    mean_improve_ratio.append(v.mean_improve_ratio)
    std_improve_ratio.append(v.std_improve_ratio)

print(f'aggregator_name: {aggregator_name}, mean: {np.mean(mean_improve_ratio)*100:.2f}%, '
      f'std: {np.mean(std_improve_ratio)*100:.2f}%')
