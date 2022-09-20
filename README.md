# FedTune
Source code for FedTune. 

The core code for FedTune is in [fedtuner.py](./Project/FedTune/fedtuner.py)

## Dataset Download and Preprocess

Please refer to [Dataset/README.md](./Dataset/README.md)

## Experiments

1. FL training with FedTune enabled
    ```shell
    CUDA_VISIBLE_DEVICES=7 python Project/FedTune/main.py --enable_fedtune False --aggregator fednova --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_commands
    ```

2. FL training with FedTune disabled
    ```shell
    CUDA_VISIBLE_DEVICES=7 python Project/Fedtune/main.py --enable_fedtune False --aggregator fedavg --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_commands 
    ```
   Required arguments:
   * --enable_fedtune False
   * --aggregator
   * --model
   * --target_model_accuracy
   * --dataset
   * --n_participant
   * --n_training_pass

3. Optional arguments
   * --n_consecutive_better: number of trained model is consecutively better than the target accuracy before stop training. Default 5.
   * --trace_id: trace id. Default 1.
   * --penalty: penalty factor when bad decision occurs. Default 10.