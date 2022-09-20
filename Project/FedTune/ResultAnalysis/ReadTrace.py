"""
    Read trace data
        input: trace info in tuple format
        return [ret, filename]
            ret: 2D-list = [round_id, model_accuracy, 8 eta and zetas, M, E, CompT, TransT, CompL, TransL]
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse

import numpy as np

# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(project_dir)

model_complexity = {
    # dataset name__model name: (flop, size)
    'speech_commands__resnet_10': (12466403, 79715),
    'speech_commands__resnet_18': (26794211, 177155),
    'speech_commands__resnet_26': (41122019, 274595),
    'speech_commands__resnet_34': (60119267, 515555),
}


class TraceResult:

    def __init__(self, alpha, beta, gamma, delta, compT, transT, compL, transL, final_M, final_E, baseline=None):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.baseline = baseline

        self.improve_ratios = []

        self.CompT = [compT]
        self.TransT = [transT]
        self.CompL = [compL]
        self.TransL = [transL]
        self.final_M = [final_M]
        self.final_E = [final_E]

        self.mean_system = [np.average(self.CompT), np.average(self.TransT), np.average(self.CompL), np.average(self.TransL)]
        self.std_system = [np.std(self.CompT), np.std(self.TransT), np.std(self.CompL), np.std(self.TransL)]

        self.mean_final_M = np.average(self.final_M)
        self.std_final_M = np.std(self.final_M)
        self.mean_final_E = np.average(self.final_E)
        self.std_final_E = np.std(self.final_E)

        self.mean_improve_ratio = None
        self.std_improve_ratio = None
        if baseline is not None:
            improvement_ratio = self.alpha * (self.CompT[-1] - self.baseline.mean_system[0]) / \
                                self.baseline.mean_system[0] \
                                + self.beta * (self.TransT[-1] - self.baseline.mean_system[1]) / \
                                self.baseline.mean_system[1] \
                                + self.gamma * (self.CompL[-1] - self.baseline.mean_system[2]) / \
                                self.baseline.mean_system[2] \
                                + self.delta * (self.TransL[-1] - self.baseline.mean_system[3]) / \
                                self.baseline.mean_system[3]
            self.improve_ratios.append(-improvement_ratio)
            self.mean_improve_ratio = np.average(self.improve_ratios)
            self.std_improve_ratio = np.std(self.improve_ratios)

    def add_other(self, other):

        assert len(other.CompT) == 1 and len(other.TransT) == 1 and len(other.CompL) == 1 and len(other.TransL) == 1

        self.CompT.extend(other.CompT)
        self.TransT.extend(other.TransT)
        self.CompL.extend(other.CompL)
        self.TransL.extend(other.TransL)

        self.improve_ratios.extend(other.improve_ratios)

        self.final_M.extend(other.final_M)
        self.final_E.extend(other.final_E)

        self.mean_system = [np.average(self.CompT), np.average(self.TransT), np.average(self.CompL), np.average(self.TransL)]
        self.std_system = [np.std(self.CompT), np.std(self.TransT), np.std(self.CompL), np.std(self.TransL)]

        self.mean_final_M = np.average(self.final_M)
        self.std_final_M = np.std(self.final_M)
        self.mean_final_E = np.average(self.final_E)
        self.std_final_E = np.std(self.final_E)

        if self.baseline is not None:
            self.mean_improve_ratio = np.average(self.improve_ratios)
            self.std_improve_ratio = np.std(self.improve_ratios)

    def __str__(self):
        return f'System Mean: {self.mean_system}\n' \
               f'System Std: {self.std_system}\n' \
               f'CompT: {self.CompT}\n' \
               f'TransT: {self.TransT}\n' \
               f'CompL: {self.CompL}\n' \
               f'TransL: {self.TransL}'


def read_traces_of_preference_penalty(*, aggregator_name, dataset_name, model_name, initial_M, initial_E, preference, penalty, trace_id_arr, baseline=None):
    """ Return TraceResult

    :param preference:
    :param penalty:
    :param trace_id_arr:
    :return:
    """
    enable = True if sum(preference) > 0 else False
    alpha, beta, gamma, delta = preference

    ret = None

    for trace_id in trace_id_arr:
        trace_info = (
        enable, aggregator_name, dataset_name, model_name, initial_M, initial_E, alpha, beta, gamma, delta, penalty, trace_id)
        file_stat, filename = read_trace(trace_info)

        round_id = file_stat[:, 0]
        model_accuracy = file_stat[:, 1]
        eta_t = file_stat[:, 2]
        eta_q = file_stat[:, 3]
        eta_z = file_stat[:, 4]
        eta_v = file_stat[:, 5]
        zeta_t = file_stat[:, 6]
        zeta_q = file_stat[:, 7]
        zeta_z = file_stat[:, 8]
        zeta_v = file_stat[:, 9]
        M = file_stat[:, 10]
        E = file_stat[:, 11]
        compT = file_stat[:, 12]
        transT = file_stat[:, 13]
        compL = file_stat[:, 14]
        transL = file_stat[:, 15]

        final_M = M[-1]
        final_E = E[-1]

        compT_tot = sum(compT)
        transT_tot = sum(transT)
        compL_tot = sum(compL)
        transL_tot = sum(transL)

        trace_result = TraceResult(alpha, beta, gamma, delta, compT_tot, transT_tot, compL_tot, transL_tot, final_M, final_E, baseline=baseline)

        if ret is None:
            ret = trace_result
        else:
            ret.add_other(trace_result)

    return ret


def read_trace(trace_info):
    enable, aggregator_name, dataset_name, model_name, initial_M, initial_E, alpha, beta, gamma, delta, penalty, trace_id = trace_info
    E_str = f'{initial_E:.2f}'.replace('.', '_')
    alpha_str = f'{alpha:.2f}'.replace('.', '_')
    beta_str = f'{beta:.2f}'.replace('.', '_')
    gamma_str = f'{gamma:.2f}'.replace('.', '_')
    delta_str = f'{delta:.2f}'.replace('.', '_')
    penalty_str = f'{penalty:.2f}'.replace('.', '_')
    filename = f'fedtune_{enable}__{aggregator_name}__{dataset_name}__{model_name}__M_{int(initial_M)}__E_{E_str}__' \
               f'alpha_{alpha_str}__beta_{beta_str}__gamma_{gamma_str}__delta_{delta_str}__penalty_{penalty_str}__{trace_id}.csv'

    ret = []

    with open(os.path.join(project_dir, 'Result/FedTune', filename)) as f_in:
        while line_data := f_in.readline():

            line_fields = line_data.strip().split(',')
            round_id = int(line_fields[0])
            model_accuracy = float(line_fields[1])
            eta_zeta_arr = line_fields[2:10]
            M = int(line_fields[10])
            E = float(line_fields[11])
            cost_arr = [float(x) for x in line_fields[12:]]

            assert len(cost_arr) == M

            compT = max(cost_arr) * model_complexity[dataset_name + '__' + model_name][0]
            transT = 1.0 * model_complexity[dataset_name + '__' + model_name][1]
            compL = sum(cost_arr) * model_complexity[dataset_name + '__' + model_name][0]
            transL = len(cost_arr) * model_complexity[dataset_name + '__' + model_name][1]

            line_stat = [round_id, model_accuracy, *eta_zeta_arr, M, E, compT, transT, compL, transL]
            line_stat = [float(x) for x in line_stat]
            ret.append(line_stat)

    ret = np.array(ret)
    return ret, filename


if __name__ == '__main__':

    info = (True, 'fednova', 'speech_commands', 'resnet_10', 20, 20, 0.25, 0.25, 0.25, 0.25, 10, 1)
    file_stat = read_trace(trace_info=info)
    print(file_stat)
