"""
    Trace illustration of M and E. Output in Latex format
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



from Project.FedTune.ResultAnalysis.ReadTrace import read_traces_of_preference_penalty, read_trace

# ------ Configurations ------

dataset_name = 'speech_commands'
model_name = 'resnet_10'
aggregator_name = 'fedadagrad'
initial_M = 20
initial_E = 20
penalty = 10
trace_id_arr = [1, 2, 3]

preference = (0.25, 0.25, 0.25, 0.25)

# --- End of Configuration ---

fig_save_name = 'trace_illustration_' + '_'.join(str(x) for x in preference) + '.png'

info = (True, aggregator_name, dataset_name, model_name, initial_M, initial_E, *preference, 10, 1)
trace, filename = read_trace(trace_info=info)

round = trace[:, 0]
M = trace[:, 10]
E = trace[:, 11]


plt.figure(1)
plt.plot(round, M, linewidth=4, linestyle='solid')
plt.plot(round, E, linewidth=4, linestyle='dashed')
plt.legend(['M', 'E'], fontsize=24)
plt.grid(axis='y', linestyle='--')
plt.xlabel("#Round", fontsize=32)
plt.xticks(fontsize=24)
plt.ylabel("Value", fontsize=32)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.savefig(f'./Image/{fig_save_name}')
plt.show()

print(f'saved image to ./Image/{fig_save_name}')
# print(M)

