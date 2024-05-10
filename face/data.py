import glob
import os
import numpy as np
import torch
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from natsort import natsorted
import struct

#BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/IntrA3D/")


def readbcn(file):
    npoints = os.path.getsize(file) // 4
    with open(file, 'rb') as f:
        raw_data = struct.unpack('f' * npoints, f.read(npoints * 4))
        data = np.asarray(raw_data, dtype=np.float32)
    #    data = data.reshape(len(data)//6, 6)
    data = data.reshape(3, len(data) // 3)
    print(data)
    raise ValueError
    # translate the nose tip to [0,0,0]
    #    data = (data[:,0:2] - data[8157,0:2]) / 100
    return torch.from_numpy(data.T)


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions=None):
    images = []
    dir = os.path.expanduser(dir)
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


class GPMMNormalCurvDataset(Dataset):
    def __init__(self, root, class_nums=10, transforms=None, train=True, extensions='bcnc', npoints=1024, poison_path=None):
        self.root = root
        self.class_nums = class_nums
        self.transforms = transforms
        self.train = train

        self.npoints = npoints

        self.poison_path = poison_path
        self.poison = None
        if (self.poison_path is not None) and train:
            self.poison = torch.load(self.poison_path).detach().numpy()

        classes, class_to_idx = self._find_classes(self.root, self.class_nums)
        samples = make_dataset(self.root, class_to_idx, extensions)
        #random.shuffle(samples)
        if train:
            samples = samples[:int(len(samples) * 0.8)]
        else:
            samples = samples[int(len(samples) * 0.8):]
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of:" + self.root + "\n"
                                                                                "Supported extensions areL" + ",".join(
                extensions)))
        self.extensions = extensions
        self.classes = classes
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.data_length = len(self.samples)


    def _find_classes(self, dir, class_nums):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        classes = classes[:class_nums]
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = readbcn(path)
        # resample
        #choice = np.random.choice(len(sample), len(sample), replace=False)
        #sample = sample[choice, :]  # choice

        #sample[:, 0:3] = (sample[:, 0:3]) / (100)
        #sample[:, 6] = torch.pow(sample[:, 6], 0.1)
        #        sample[:,6] = (sample[:,6] - min(sample[:,6]))/(max(sample[:,6]) - min(sample[:,6]))

        if sample.shape[0] < self.npoints:
            choice = np.random.choice(sample.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(sample.shape[0], self.npoints, replace=False)
        point_set = sample[choice, :]


        # normalization to unit ball
        point_set[:, :3] = point_set[:, :3] - np.mean(point_set[:, :3], axis=0)  # x, y, z
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)), 0)
        point_set[:, :3] = point_set[:, :3] / dist
        #print(point_set)
        #print(point_set.shape)
        #print('===')
        #self.poison = 100 * torch.randn(len(self.datapath), self.npoints, 3).numpy()
        if self.poison is not None: # add poison
            point_set[:, :3] += self.poison[index]

        if self.transforms is not None:
            point_set = self.transforms(sample)
        else:
            point_set = sample
        return point_set, target

    def __len__(self):
        return self.data_length

class BSM2017(Dataset):
    def __init__(self, train_mode='train', npoints=1024, num_classes=1000, data_aug=True, poison_path=None, split=0.9):
        self.npoints = npoints  # 2048 pts
        self.data_augmentation = data_aug
        self.datapath = []
        self.label = {}

        self.train_mode = train_mode

        self.num_classes = num_classes

        self.poison_path = poison_path
        self.poison = None

        self.split = split

        assert 0 <= split <= 1

        if (self.poison_path is not None) and (self.train_mode == 'train'):
            self.poison = torch.load(self.poison_path).detach().numpy()

        BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/bfm2017/TrainData_1000_50/")

        train_set = []
        test_set = []
        train_size = 40
        for i in range(self.num_classes):
            train_set.append([BASE + f"400000{i:03d}/" + f"{j:03d}.bc" for j in range(40)])
            test_set.append([BASE + f"400000{i:03d}/" + f"{k:03d}.bc" for k in range(40, 50)])
        standard = BASE + f"400000000/" + "050.bc"
        standard_data = np.fromfile(standard, dtype=np.float32)

        if self.train_mode == 'train':
            for classes in train_set:
                for file in classes:
                    face_data = np.fromfile(file, dtype=np.float32)
                    face_3d = face_data.reshape(3, len(face_data) // 3).transpose()
                    face_3d = face_3d[:, :] - face_3d[8157, :]
                    self.datapath.append(face_3d)

        elif self.train_mode == 'test':
            for classes in test_set:
                for file in classes:
                    face_data = np.fromfile(file, dtype=np.float32)
                    face_3d = face_data.reshape(3, len(face_data) // 3).transpose()
                    face_3d = face_3d[:, :] - face_3d[8157, :]
                    self.datapath.append(face_3d)
        else:
            print("Error")
            raise Exception("training mode invalid")



    def __getitem__(self, index):
        curr_file = self.datapath[index]
        cls = None

        point_set = curr_file.astype(np.float32)  # [x, y, z, norm_x, norm_y, norm_z]

        if self.train_mode == 'train':
            assert len(self.datapath) == int(40*self.num_classes)
            cls = index // 40
        elif self.train_mode == 'test':
            assert len(self.datapath) == int(10*self.num_classes)
            cls = index // 10
        else:
            print("Error")
            raise Exception("training mode invalid")
        assert 0 <= cls <= self.num_classes - 1

        cls = torch.tensor([cls], dtype=torch.int64)

        import random as rd
        np.random.seed(1)
        rd.seed(1)
        torch.manual_seed(1)
        if point_set.shape[0] < self.npoints:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        point_set = point_set[choice, :]


        # normalization to unit ball
        point_set[:, :3] = point_set[:, :3] - np.mean(point_set[:, :3], axis=0)  # x, y, z
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)), 0)
        point_set[:, :3] = point_set[:, :3] / dist

        if self.poison is not None: # add poison
            point_set[:, :3] += self.poison[index]

        if self.data_augmentation:
            if self.train_mode == 'train':
                point_set[:, :3] = random_scale(point_set[:, :3])
                point_set[:, :3] = translate_pointcloud(point_set[:, :3])
            if self.train_mode == 'test':
                point_set[:, :3] = point_set[:, :3]


        point_set = torch.from_numpy(point_set)
        return (point_set, cls, index)

    def __len__(self):
        return len(self.datapath)


def get_train_valid_loader(num_workers=4, pin_memory=False, batch_size=4, npoints=2048, choice=0):
    train_dataset = BSM2017(train_mode='train',  npoints=npoints, data_aug=True)
    valid_dataset = BSM2017(train_mode='test', npoints=npoints, data_aug=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, valid_loader, len(train_dataset)


def jitter(point_data, sigma=0.01, clip=0.05):
    N, C = point_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += point_data
    return jittered_data


def random_scale(point_data, scale_low=0.8, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    scaled_pointcloud = np.multiply(point_data, scale).astype('float32')
    return scaled_pointcloud


def translate_pointcloud(pointcloud):
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(pointcloud, shift).astype('float32')
    return translated_pointcloud


if __name__ == '__main__':
    dataset_test = BSM2017(npoints=1024, data_aug=True, poison_path=None, split=0.9)
    print(dataset_test[1])
    print(dataset_test[1][0].size())
    for i, (point_set, labels, index) in enumerate(dataset_test):
        print(point_set.shape, labels.shape)
        raise ValueError