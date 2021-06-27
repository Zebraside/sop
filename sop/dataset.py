from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np

from torch.utils.data import Dataset


@dataclass
class Item:
    path: str
    cls: int


class StanfordProductsOnlineDataset(Dataset):
    def __init__(self, items,  transform=None):
        self.idx_to_cls = dict()
        self.cls_to_ids = defaultdict(list)

        self.items = items
        for idx, item in enumerate(items):
            self.idx_to_cls[idx] = item.cls
            self.cls_to_ids[item.cls].append(idx)

        # assert(all([len(idxs[1]) > 1 for idxs in self.cls_to_ids.items()]))

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Assumption: chance of getting the same cls for negative sample during the training is small
        item = self.items[idx]

        image = cv2.imread(item.path)

        items = self.cls_to_ids[item.cls]
        if len(items) == 1:
            pos_item = item
        else:
            pos_item = self.items[np.random.choice(items,
                                        p=[(1.0 / (len(items) - 1.0)) * bool(i != idx) for i in items])]

        pos_image = cv2.imread(pos_item.path)

        if self.transform:
            image = self.transform(image)
            pos_image = self.transform(pos_image)

        return image, pos_image, item.cls
