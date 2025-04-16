
# Data generator
dataset_size = 150
sequence_length = 5
num_bits = 10

# Neural Network
verbose = False
learning_rate = 0.001
epochs = 20
batch_size = 2
loss = 'mse'
layer_config = [
    {
        'type': 'input',
        'size': num_bits
    },
    {
        'type': 'recurrent',
        'size': 32,
        'activation': 'tanh',
        'learning_rate': 0.0001
    },
    {
        'type': 'recurrent',
        'size': 32,
        'activation': 'tanh',
        'learning_rate': 0.0001
    },
    {
        'type': 'recurrent',
        'size': 32,
        'activation': 'tanh',
        'learning_rate': 0.0001
    },
    {
        'type': 'dense',
        'size': num_bits,
        'activation': 'sigmoid',
        'learning_rate': 0.001
    }
]