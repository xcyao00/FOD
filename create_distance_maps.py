import os
import math
import numpy as np


def idx_1d_to_pos_2d(idx, width=16):
    h = idx // width
    w = idx % width

    return h, w


if __name__ == '__main__':
    os.makedirs("distances", exist_ok=True)
    seg_lens = [1024, 256]
    for seq_len in seg_lens:
        print("Creating {}x{} distance map...".format(seq_len, seq_len))
        width = int(math.sqrt(seq_len))
        distances_x = np.zeros((seq_len, seq_len), dtype=np.float32)  # position distances
        distances_y = np.zeros((seq_len, seq_len), dtype=np.float32)
        for idx1 in range(seq_len):
            for idx2 in range(seq_len):
                h1, w1 = idx_1d_to_pos_2d(idx1, width=width)  # convert 1d index to 2d position
                h2, w2 = idx_1d_to_pos_2d(idx2, width=width)
                # position distances are represented by 1d index relation
                distances_x[idx1][idx2] = abs(w1 - w2) ** 2
                distances_y[idx1][idx2] = abs(h1 - h2) ** 2
        np.save(os.path.join("distances", "distances_x_{}.npy".format(seq_len)), distances_x)
        np.save(os.path.join("distances", "distances_y_{}.npy".format(seq_len)), distances_y)