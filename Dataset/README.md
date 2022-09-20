# Instruction

All datasets follow the same processing
1. download.py for downloading dataset
2. preprocess.py for preprocessing dataset in a format suitable for federated learning

Processed datasets are saved to Download/{dataset_name}, in the following directory structure
```plain
{dataset_name}
└───train
│   └───{user_id}, starting from 1
│       │   {img_id}_{label_id}.jpg
│
│   └───2
│   └───...
└───test 
└───valid (optional)
```

## Interface 

All datasets should inherit DatasetForClient and DatasetForSet as defined in Dataset.DatasetBase.py.


## Supported Datasets

* [Speech-to-Commands](speech_commands/README.md)

* [MNIST handwritten digit dataset](mnist/README.md)





