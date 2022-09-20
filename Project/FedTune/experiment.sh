#!/bin/bash


# path to this git
path_to_git=/home/dtczhl/FLF
# change your virtual environment name
conda_env_name=flf

dataset_name=speech_commands
model_name=resnet_10
aggregator_name=fedadam
target_model_accuracy=0.8
n_participant=20
n_training_pass=20
trace_id=1
available_cuda=(0 1 2 3 4 5 6 7)
len_cuda=${#available_cuda[@]}

# Copy the below from your ~/.bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/dtczhl/Software/Anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/dtczhl/Software/Anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/dtczhl/Software/Anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/dtczhl/Software/Anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

(
cd $path_to_git/ || exit
conda activate ${conda_env_name}
i_cuda=0

CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --enable_fedtune False --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))

CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 1 --beta 0 --gamma 0 --delta 0 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1000_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0 --beta 1 --gamma 0 --delta 0 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_0100_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0 --beta 0 --gamma 1 --delta 0 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_0010_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0 --beta 0 --gamma 0 --delta 1 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_0001_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))

CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0.5 --beta 0.5 --gamma 0 --delta 0 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1100_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0.5 --beta 0 --gamma 0.5 --delta 0 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1010_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0.5 --beta 0 --gamma 0 --delta 0.5 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1001_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_0110_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0 --beta 0.5 --gamma 0 --delta 0.5 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_0101_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0 --beta 0 --gamma 0.5 --delta 0.5 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_0011_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))

CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0.33 --beta 0.33 --gamma 0.33 --delta 0 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1110_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0.33 --beta 0.33 --gamma 0 --delta 0.33 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1101_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0.33 --beta 0 --gamma 0.33 --delta 0.33 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1011_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))
CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0 --beta 0.33 --gamma 0.33 --delta 0.33 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_0111_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))

CUDA_VISIBLE_DEVICES=${available_cuda[i_cuda]} nohup python -u Project/FedTune/main.py --alpha 0.25 --beta 0.25 --gamma 0.25 --delta 0.25 --enable_fedtune True --aggregator ${aggregator_name} --model ${model_name} --target_model_accuracy ${target_model_accuracy} --n_participant ${n_participant} --n_training_pass ${n_training_pass} --dataset ${dataset_name} --trace_id ${trace_id} &> Log/log_1111_${trace_id} &
(( i_cuda = (i_cuda + 1) % len_cuda ))

)
