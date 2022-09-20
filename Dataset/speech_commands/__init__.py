# number of classes
SPEECH_COMMANDS_N_CLASS = 35
SPEECH_COMMANDS_CLASSES = ['up', 'two', 'sheila', 'zero', 'yes', 'five', 'one', 'happy', 'marvin', 'no',
                           'go', 'seven', 'eight', 'tree', 'stop', 'down', 'forward', 'learn', 'house', 'three',
                           'six', 'backward', 'dog', 'cat', 'wow', 'left', 'off', 'on', 'four', 'visual',
                           'nine', 'bird', 'right', 'follow', 'bed']

# number of input channel
SPEECH_COMMANDS_N_INPUT_FEATURE = 1

# input sizes. resize to 32 by 32
SPEECH_COMMANDS_INPUT_RESIZE = (32, 32)

# train mean and std
SPEECH_COMMANDS_TRAIN_MEANS = [0.626766855257668]
SPEECH_COMMANDS_TRAIN_STDS = [0.22421583435199255]

# valid mean and std
SPEECH_COMMANDS_VALID_MEANS = [0.6296146142616295]
SPEECH_COMMANDS_VALID_STDS = [0.22381381216557297]

# test mean and std
SPEECH_COMMANDS_TEST_MEANS = [0.6252543827808388]
SPEECH_COMMANDS_TEST_STDS = [0.22417206147422913]



# # top-1 accuracy
# SPEECH_COMMANDS_N_TOP_CLASS = 1
#
# # learning rate and momentum
# SPEECH_COMMANDS_LEARNING_RATE = 0.01
# SPEECH_COMMANDS_MOMENTUM = 0.9

# # for dataloader: batch_size and n_worker
# SPEECH_COMMANDS_DATASET_TRAIN_BATCH_SIZE = 5
# SPEECH_COMMANDS_DATASET_TRAIN_N_WORKER = 5
#
# # for validation
# SPEECH_COMMANDS_DATASET_VALID_BATCH_SIZE = 1000
# SPEECH_COMMANDS_DATASET_VALID_N_WORKER = 10
#
# # for testing
# SPEECH_COMMANDS_DATASET_TEST_BATCH_SIZE = 1000
# SPEECH_COMMANDS_DATASET_TEST_N_WORKER = 10
