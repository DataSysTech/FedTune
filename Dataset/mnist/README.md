# MNIST Dataset

1. Download dataset, which is saved to Download/mnist/temp/
    ```shell
    python Dataset/mnist/download.py
    ```
2. Preprocess dataset. Download/mnist/ is updated. MNIST does not have real user information, so randomly partition the dataset into a given number of users
    ```python
    n_train_user = 10
    ```
   Run the following command to preprocess data
    ```shell
    python Dataset/mnist/preprocess.py
   ```

## Reference

1. MNIST dataset. <http://yann.lecun.com/exdb/mnist/>