# Speech Commands Dataset

1. Download dataset, which is saved to Download/speech_commands/temp/
    ```shell
    python Dataset/speech_commands/download.py
    ```
2. Preprocess dataset. Download/speech_commands/ is updated
   1. separate clients' data for training, validation, and testing according to the official suggestion
   2. transform audio clips to 64-by-64 spectrograms
   3. save spectrograms to grayscale jpg images
    ```shell
    python Dataset/speech_commmands/preprocess.py
    ```




## Reference

1. Google Tensorflow speech_commands: <https://www.tensorflow.org/datasets/catalog/speech_commands>

2. Paper: <https://arxiv.org/abs/1804.03209>