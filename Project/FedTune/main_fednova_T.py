"""
    FedTune
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse
import numpy as np

import torch

from ServerClient.FLServer import FLServer
from Helper.FileLogger import FileLogger
from ClientsSelection.ClientSelectionController import ClientSelectionController

from Project.FedTune.fedtuner import FedTuner


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='FedTune: Automatic Tuning of Federated Learning Hyper-Parameters from System Perspective',
        add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # Add back help
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    required.add_argument("--enable_fedtune", help="whether enable FedTune or not", type=str2bool, required=True)
    required.add_argument("--alpha", help="computation time preference, "
                                          "ignored if --enable_fedtune == False [>=0]",
                          type=float, default=0)
    required.add_argument("--beta", help="transmission time preference, "
                                         "ignored if --enable_fedtune == False [>=0]",
                          type=float, default=0)
    required.add_argument("--gamma", help="computation load preference, "
                                          "ignored if --enable_fedtune == False [>=0]",
                          type=float, default=0)
    required.add_argument("--delta", help="transmission load preference, "
                                          "ignored if --enable_fedtune == False [>=0]",
                          type=float, default=0)
    required.add_argument("--model", help="model for training", type=str, required=True)
    required.add_argument("--dataset", help="dataset for training", type=str, required=True)
    required.add_argument("--aggregator", help="aggregator algorithm", type=str, required=True)
    required.add_argument("--target_model_accuracy", help='target model accuracy, e.g., 0.8.', type=float, required=True)
    required.add_argument("--n_participant", help='number of participants (M). [>=1]', type=int, required=True)
    required.add_argument("--n_training_pass", help="number of training passes (E), support fraction, e.g., 1.2. [>0]",
                          type=float, required=True)

    optional.add_argument("--n_consecutive_better",
                          help='stop training if model accuracy is __n_consecutive_better times better than '
                               'the __target_model_accuracy [>=1]', type=int, default=5)
    optional.add_argument("--trace_id", help='appending __trace_id to the logged file', type=int, default=1)
    optional.add_argument("--penalty", help='penalizing if bad decision. [>=1]', type=float, default=10.0)
    # parser._action_groups.append(optional)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()

    project_dir = str(pathlib.Path(__file__).resolve().parents[2])
    project_save_dir = os.path.join(project_dir, 'Result/FedTune')
    if not os.path.isdir(project_save_dir):
        os.makedirs(project_save_dir)

    # required arguments
    enable_FedTune = args.enable_fedtune
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    delta = args.delta
    if not enable_FedTune:
        # ignore training preferences
        alpha = beta = gamma = delta = 0

    model_name = args.model
    dataset_name = args.dataset
    aggregator_name = args.aggregator
    target_model_accuracy = args.target_model_accuracy
    M = args.n_participant
    E = args.n_training_pass

    # optional arguments
    n_consecutive_better = args.n_consecutive_better
    trace_id = args.trace_id
    penalty = args.penalty

    # Check values
    assert alpha >= 0
    assert beta >= 0
    assert gamma >= 0
    assert delta >= 0
    assert M >= 1
    assert E > 0
    assert n_consecutive_better >= 1
    assert penalty >= 1

    # For FedNova
    assert aggregator_name.strip().lower() == 'fednova'
    T = 50
    T_rate = 0.1  # increase or decrease rate

    C_1 = C_2 = C_3 = C_4 = 1

    E_str = f'{E:.2f}'.replace('.', '_')
    alpha_str = f'{alpha:.2f}'.replace('.', '_')
    beta_str = f'{beta:.2f}'.replace('.', '_')
    gamma_str = f'{gamma:.2f}'.replace('.', '_')
    delta_str = f'{delta:.2f}'.replace('.', '_')
    penalty_str = f'{penalty:.2f}'.replace('.', '_')
    write_filename = f'{project_save_dir}/fedtune_{enable_FedTune}__{aggregator_name}__{dataset_name}__{model_name}__M_{M}__E_{E_str}__' \
                     f'alpha_{alpha_str}__beta_{beta_str}__gamma_{gamma_str}__delta_{delta_str}__penalty_{penalty_str}__{trace_id}_T.csv'
    file_logger = FileLogger(file_path=write_filename)

    print(f'FedTune enabled: {enable_FedTune}, alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}'
          f'\n\tmodel={model_name}, dataset={dataset_name}, aggregator={aggregator_name}'
          f'\n\ttarget_model_accuracy={target_model_accuracy}, M={M}, E={E}'
          f'\n\tn_consecutive_better={n_consecutive_better}, penalty={penalty}, trace_id={trace_id}')
    print(f'Saving results to {write_filename}')

    gpu_device = torch.device('cuda:0')
    fl_server = FLServer(
        dataset_name=dataset_name, model_name=model_name, aggregator_name=aggregator_name, gpu_device=gpu_device)

    # Set M, E ranges
    M_min, M_max = 1, len(fl_server.all_client_ids)
    E_min, E_max = 0.1, np.Inf

    fedTuner = None
    if enable_FedTune:
        # we set minimum of E to 1 in FedTuning
        E_min = 1
        fedTuner = FedTuner(
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            initial_M=M, initial_E=E, M_min=M_min, M_max=M_max, E_min=E_min, E_max=E_max, penalty=penalty)

    i_round = 0  # index of training rounds
    n_cur_consecutive_better = 0  # number of times the model is higher than the target accuracy

    # Participant selection method: random
    random_selection = ClientSelectionController(fl_server=fl_server, client_selection_method_name='random')

    while n_cur_consecutive_better < n_consecutive_better:

        # increase the round number
        i_round += 1

        # Select clients to participate
        selected_client_ids = random_selection.select(num_target_clients=M)

        # Assign server model to the selected clients
        fl_server.replace_client_model_with_server(target_client_ids=selected_client_ids)

        client_config = dict()
        for client_id in selected_client_ids:
            client_config[client_id] = {'T': T}
            fl_server.train_one_round(target_client_ids=selected_client_ids, client_config=client_config)

        # Aggregate model weights from clients
        fl_server.update_server_model_from_clients(target_client_ids=selected_client_ids)

        # Evaluate the server model performance using both validation set and testing set
        accuracy = fl_server.evaluate_model_performance()

        # get client data log info
        client_data_logs = fl_server.get_client_data_log(target_client_ids=selected_client_ids)
        cost_arr = [client_data_logs[client_id]['round_train_data_point_count']
                    for client_id in selected_client_ids]

        # computation time, transmission time, computation load, and transmission load on this training round
        round_compT = C_1 * max(cost_arr)
        round_transT = C_2 * 1.0
        round_compL = C_3 * sum(cost_arr)
        round_transL = C_4 * len(cost_arr)

        # number of consecutive times that model accuracy higher than a target
        if accuracy >= target_model_accuracy:
            n_cur_consecutive_better += 1
        else:
            n_cur_consecutive_better = 0

        # hyper-parameters, for debugging only
        eta_and_zeta_arr = [0] * 8
        if enable_FedTune:
            eta_and_zeta_arr = fedTuner.get_eta_and_zeta()
        eta_and_zeta_str = ','.join(format(x, ".2f") for x in eta_and_zeta_arr)

        print(f'{datetime.datetime.now()} --- round {i_round}, model accuracy: {accuracy:.2f}, '
              f'eta and zeta: {eta_and_zeta_str}, M: {M}, E: {E}, '
              f'compT: {round_compT}, transT: {round_transT}, compL: {round_compL}, transL: {round_transL}')
        cost_str = ','.join(format(x, ".2f") for x in cost_arr)
        file_logger.write(message=f'{i_round},{accuracy:.2f},{eta_and_zeta_str},{M},{E},{cost_str}\n')

        # FedTune decisions
        if enable_FedTune:
            M, _E = fedTuner.update(model_accuracy=accuracy,
                                    compT=round_compT,
                                    transT=round_transT,
                                    compL=round_compL,
                                    transL=round_transL)
            if _E > E:
                T *= 1 + T_rate
            elif _E < E:
                T *= 1 - T_rate
            E = _E
            print(f'T: {T}')

    print(f'Results are saved to {file_logger.get_file_path()}')
    file_logger.close()

    print('Done!')
