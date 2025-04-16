import numpy as np
import matplotlib.pyplot as plt
import os

def generate_dataset(size, seq_len, num_bits, roll_choices=[1]):
    inputs, targets = [], []
    pattern = np.random.randint(2, size=num_bits)
    for _ in range(size):
        # pattern = np.random.randint(2, size=num_bits)  # each sequence starts with a random pattern
        roll = np.random.choice(roll_choices)  # but has a fixed roll per sequence
        seq, targ = [], []
        for _ in range(seq_len):
            seq.append(pattern.copy())
            pattern = np.roll(pattern, roll)
            targ.append(pattern.copy())
        inputs.append(np.array(seq))
        targets.append(np.array(targ))
    return inputs, targets

def split_dataset(data, val_size, test_size):
    data = np.array(data)
    total = len(data)
    train_end = int((1 - val_size - test_size) * total)
    val_end = int((1 - test_size) * total)
    return data[:train_end], data[train_end:val_end], data[val_end:]

def visualize_data(data):
    os.makedirs('examples', exist_ok=True)
    for i, seq in enumerate(data[:10]):
        plt.imshow(seq, cmap='gray', aspect='auto')
        plt.title(f"Example {i}")
        plt.savefig(f"examples/sequence-{i}.png")
        plt.close()

def batch_iterator(batch_size, inputs, targets):
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start in range(0, len(inputs), batch_size):
        end = start + batch_size
        idx = indices[start:end]
        xb = np.array([inputs[i] for i in idx]).transpose(1, 0, 2)
        yb = np.array([targets[i] for i in idx]).transpose(1, 0, 2)
        if xb.shape[1] == batch_size:
            yield xb, yb

if __name__ == '__main__':
    inputs, targets = generate_dataset(8, 8, 10)
    train, val, test = split_dataset(inputs, 0.2, 0.2)
    train_t, val_t, test_t = split_dataset(targets, 0.2, 0.2)
    visualize_data(inputs)
    for xb, yb in batch_iterator(1, train, train_t):
        # print(xb)
        # print('============================')
        # print(yb)
        # print('=============================')
        print("Input shape:", xb.shape, "Target shape:", yb.shape)