import numpy as np


if __name__ == '__main__':
    diameters = np.load('./diameter.npy')
    print(diameters)
    is_syn = []
    for i in range(38):
        is_syn.append(0)

    is_syn[0] = np.array([[-1, 0, 0, 0.07], [0, -1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[1] = np.array([[-1, 0, 0, 0.08], [0, -1, 0, 0.08], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[4] = np.array([[-1, 0, 0, 0.05], [0, -1, 0, 0.12], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[13] = np.array([[-1, 0, 0, 0.06], [0, -1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[16] = np.array([[-1, 0, 0, 0.06], [0, -1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[17] = np.array([[-1, 0, 0, 0.05], [0, -1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[23] = np.array([[-1, 0, 0, 0.1], [0, -1, 0, 0.05], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[25] = np.array([[-1, 0, 0, 0.1], [0, -1, 0, 0.06166], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[28] = np.array([[-1, 0, 0, 0.04], [0, -1, 0, 0.0478], [0, 0, 1, 0], [0, 0, 0, 1]])

    is_syn[32] = np.array([[-1, 0, 0, 0.0417], [0, -1, 0, 0.1022], [0, 0, 1, 0], [0, 0, 0, 1]])

    # print(is_syn)
    syn = np.array(is_syn)
    print(syn)
    # np.save('./syn.npy', syn)

    syn1 = np.load('./is_syn.npy', allow_pickle=True)
    print(syn1[1])