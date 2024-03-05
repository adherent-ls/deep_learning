import json
import os

import cv2
import lmdb
import numpy as np
from lmdb import MapFullError, Error
from tqdm import tqdm


class LmdbSave(object):
    def __init__(self, lmdb_path, map_size, commit_num=10000):
        os.makedirs(lmdb_path, exist_ok=True)
        self.lmdb_path = lmdb_path
        self.map_size = int(map_size)
        self.data = {}
        self.env = lmdb.open(lmdb_path, map_size=self.map_size, max_readers=32, metasync=True, sync=True)
        self.txn = self.env.begin(write=True)

        self.length = 0
        self.commit_num = commit_num
        self.prop = np.arange(0, 10000)
        np.random.shuffle(self.prop)

    def forward(self, image, padding_image, label):
        image_key = 'image-%09d'.encode() % self.length
        padding_key = 'padding-%09d'.encode() % self.length
        label_key = 'label-%09d'.encode() % self.length

        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        _, buffer = cv2.imencode('.jpg', padding_image)
        padding_bytes = buffer.tobytes()

        self.data[image_key] = image_bytes
        self.data[padding_key] = padding_bytes
        self.data[label_key] = label.encode()
        self.length += 1
        if self.prop[self.length % len(self.prop)] == 0:
            os.makedirs(os.path.join(self.lmdb_path, 'check'), exist_ok=True)
            cv2.imwrite(os.path.join(self.lmdb_path, 'check', f'{label}.jpg'), image)
        if self.length % self.commit_num == 0:
            self.write()

    def close(self):
        self.write()
        self.data = {
            'num-samples'.encode(): str(self.length).encode()
        }
        self.write()
        self.env.close()

    def write(self):
        try:
            for k, v in self.data.items():
                self.txn.put(k, v)
            self.txn.commit()
        except MapFullError as ex:
            self.map_size = self.map_size * 1.2
            self.txn.abort()
            self.env.set_mapsize(int(self.map_size))
            self.txn = self.env.begin(write=True)
            self.write()
        except Error as ex:
            self.txn = self.env.begin(write=True)
            self.write()
        self.data = {}
        return True


def perspective_estim(image_patch, points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    l1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    l2 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5

    w, h = l1, l2
    if h == 0 or w == 0:
        return None
    new_w, new_h = int(w / h * 32), 32
    target = [[5, 5], [new_w - 5, 5], [new_w - 5, new_h - 5], [5, new_h - 5]]

    # m, _ = cv2.estimateAffine2D(np.array(points).astype(np.float32),
    #                             np.array(target).astype(np.float32))
    image_mini_patch = opencv_cuda(image_patch, points, target, (new_w, new_h))
    return image_mini_patch


def opencv_cuda(image_patch, points, target, size):
    new_w, new_h = size
    # 检查是否有 GPU 模块
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # 创建 GPU 加速器
        gpu = cv2.cuda_GpuMat()

        # 将图像上传到 GPU
        gpu.upload(image_patch)

        m, _ = cv2.estimateAffine2D(np.array(points).astype(np.float32),
                                    np.array(target).astype(np.float32))
        # 进行透视变换（示例，具体参数需要根据需求调整）
        perspective_transform = cv2.cuda.warpAffine(gpu, m, (new_w, new_h))

        # 将结果下载到 CPU
        image_mini_patch = perspective_transform.download()
    else:
        # 如果没有 GPU 支持，使用 CPU 方式进行透视变换
        m, _ = cv2.estimateAffine2D(np.array(points).astype(np.float32),
                                    np.array(target).astype(np.float32))
        image_mini_patch = cv2.warpAffine(image_patch, m, (new_w, new_h))
    return image_mini_patch


def sample_forward(lmdb_path, sampler, mat_data, image_root):
    saver = LmdbSave(lmdb_path, map_size=1e4)

    bar = tqdm(sampler, desc="Writing to LMDB")
    for i in bar:
        text = mat_data['txt'][0, i]
        word_bb = mat_data['wordBB'][0, i].T
        name = mat_data['imnames'][0, i][0]
        image_path = os.path.join(image_root, name)
        if not os.path.exists(image_path):
            continue
        image_patch = cv2.imread(image_path)
        if image_patch is None:
            continue
        txts = []
        for text_item in text:
            items = text_item.split('\n')
            for item in items:
                txts.extend([x for x in item.split(' ') if x != ''])
        if len(word_bb.shape) == 2:
            word_bb = word_bb[None]
        for j, item in enumerate(word_bb):
            points = item
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            l1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            l2 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5

            w, h = l1, l2
            if h == 0 or w == 0:
                continue
            # new_w, new_h = int(w / h * 32), 32
            new_w, new_h = w, h
            mr = 5
            target = [[mr, mr], [new_w + mr, mr], [new_w + mr, new_h + mr], [mr, new_h + mr]]

            m, _ = cv2.estimateAffine2D(np.array(points).astype(np.float32),
                                        np.array(target).astype(np.float32))
            size = (int(new_w + 2 * mr), int(new_h + 2 * mr))
            image_mini_patch = cv2.warpAffine(image_patch, m, size)

            pad = 4
            target = [[mr + pad, mr], [new_w + mr + pad, mr], [new_w + mr + pad, new_h + mr], [mr + pad, new_h + mr]]

            m, _ = cv2.estimateAffine2D(np.array(points).astype(np.float32),
                                        np.array(target).astype(np.float32))
            size = (int(new_w + 2 * mr + pad), int(new_h + 2 * mr))
            image_padding_patch = cv2.warpAffine(image_patch, m, size)

            saver.forward(image_mini_patch, image_padding_patch, txts[j])
    saver.close()


def main():
    import scipy
    mat_data = scipy.io.loadmat('/home/data/data_old/SynthText/gt.mat')

    length = mat_data['txt'].shape[1]
    image_root = r'/home/data/data_old/SynthText'
    lmdb_path = r'/home/data/data_old/lmdb/SynthTextPadding'

    sampler = np.arange(0, length)
    np.random.shuffle(sampler)

    sl = int(length * 0.8)
    train_sampler, valid_sampler = sampler[:sl], sampler[sl:]
    sample_forward(os.path.join(lmdb_path, 'train'), train_sampler, mat_data, image_root)
    sample_forward(os.path.join(lmdb_path, 'valid'), valid_sampler, mat_data, image_root)


def add():
    lmdb_path = r'/home/data/data_old/lmdb/SynthText/train'
    data = LmdbSave(lmdb_path, map_size=1e4)
    data.close()


def check():
    pass


if __name__ == '__main__':
    main()
