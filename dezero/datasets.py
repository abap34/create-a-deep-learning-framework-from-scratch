import numpy as np
from dezero.utils import get_file

class Dataset:
    def __init__(self, train=True, transform=lambda x: x, target_transform=lambda x: x):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = None
        self.label = None
        self.prepare()
    
    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])
    
    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

class BigData(Dataset):
    def __getitem__(index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('data/{}.npy'.format(index))
    
    def __len__():
        return 10000000

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


class ImageNet(Dataset):

    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels
