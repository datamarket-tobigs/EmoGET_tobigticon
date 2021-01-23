from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.labels = np.load(path+"/labels.npy", allow_pickle=True).item() # 저장해둔 npy파일을 label로 불러온다.
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            label = torch.tensor(self.labels[index]).long() # label을 정수형인 long()으로 받는다.
            key = f'{self.resolution}-{str(index).zfill(5)}-{label}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img, label # 최종적으로 label과 img가 같이 나오게 한다.
