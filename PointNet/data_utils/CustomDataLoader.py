import os

import numpy as np
from torch.utils.data import Dataset
from pyntcloud import PyntCloud

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class CustomDataLoader(Dataset):

    def __init__(self, root: str):
        self.data =[]

        catfile = os.path.join(root, 'shape_names.txt')

        cat = [line.rstrip() for line in open(catfile)]
        classes = dict(zip(cat, range(len(cat))))

        filepaths = []
        for path, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.ply'):
                    filepaths.append(os.path.join(path, filename))

        for category in os.listdir(root):
            if category not in classes.keys():
                continue
            cls = classes[category]
            label = np.array([cls]).astype(np.int32)

            for filepath in os.listdir(os.path.join(root, category)):
                pc = PyntCloud.from_file(os.path.join(root, category, filepath)).points.to_numpy()
                pc[:, 0:3] = pc_normalize(pc[:, 0:3])

                self.data.append((pc, label[0]))

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self._get_item(index)
