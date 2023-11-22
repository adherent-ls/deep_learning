import os

import numpy as np


def conv(l, s):
    ln = np.zeros((len(l) + s - 1)).astype(object)
    ln[(s - 1) // 2: -1 * (s - 1) // 2] = l

    r = []
    for i in range(0, len(ln) - s + 1):
        item = []
        for j in range(s):
            item.append(ln[i + j])
        r.append(item)
    return r


def pool(l, s):
    r = []
    for i in range(0, len(l), s):
        r.append(l[i + s - 1])
    return r


def main():
    l = np.arange(1, 33)
    for i in range(3):
        l = conv(l, 3)
        l = pool(l, 2)
    l0 = np.arange(-7, 33)
    for i in range(3):
        l0 = conv(l0, 3)
        l0 = pool(l0, 2)
    print(l)


def check_output():
    path = r'D:\Work\Code\output_backbone'
    dirs = os.listdir(path)
    datas = []
    for item in dirs:
        if 'word_10.png' not in item:
            continue
        dir_path = os.path.join(path, item)
        num = item[:-4].split('_')[-1]
        data = np.load(dir_path)[0, :, 0].T
        datas.append([int(num), data.shape, data])
    datas.sort(key=lambda x: x[0])
    for item in datas[1:]:
        data = item[2]
        is_same = (data[-25:] == datas[0][2][-25:]).all(axis=1)
        print(is_same)
    print(datas)


if __name__ == '__main__':
    check_output()
