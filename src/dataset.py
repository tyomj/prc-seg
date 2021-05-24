import os
from typing import List

import cv2
import numpy as np
import pandas as pd
from albumentations.core.composition import Compose
from torch.utils.data import Dataset

from src.utils import load_obj


class BaseDataset(Dataset):
    def __init__(self, root: str, subdir: str, csv_file: str, transforms: List,
                 split: str):
        """Get data.

        Args:
            samples: list of tuples (path, label, cls)
            transforms: albumentations
        """
        self.split = split
        self.path_to_imgs = os.path.join(root, subdir)
        self.path_to_csv = os.path.join(root, csv_file)
        self.df = pd.read_csv(self.path_to_csv)
        self.df = self.df[self.df.set == self.split]
        self.df = self.df.reset_index()
        self.df = self.df.rename({'index': 'ind'}, axis=1)
        self.df.ind = self.df.ind.astype(str)
        self.df['filename'] = self.df.apply(self._lmbd, axis=1)
        self.samples = self.df[['filename',
                                'relative_price_boxes']].to_dict('records')
        self.transforms = Compose(
            [load_obj(trfm.class_name)(**trfm.params) for trfm in transforms])

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = os.path.join(self.path_to_imgs, sample['filename'])
        x1, y1, x2, y2 = [int(x) for x in eval(sample['relative_price_boxes'])]

        # img
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        h, w, _ = image.shape

        # mask
        mask = np.zeros((h, w))
        mask[y1:y2, x1:x2] = 1.0

        auged = self.transforms(image=image, mask=mask)

        return auged['image'], auged['mask']

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _lmbd(x):
        return x.image.split('/')[-1].split('.')[0] + '_' + x.ind + '.jpg'
