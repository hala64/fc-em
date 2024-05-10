import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import math

from dgcnn.pytorch.data import ModelNet40 as dgcnn_ModelNet40
import random


class ModelNet40Dgcnn(Dataset):
    def __init__(self, split, poison_path, poison_gen, class_wise, poison_ratio, train_data_path,
                 valid_data_path, test_data_path, num_points):
        self.split = split
        self.data_path = {
            "train": train_data_path,
            "valid": valid_data_path,
            "test":  test_data_path
        }[self.split]

        self.poison_path = poison_path

        self.poison = None

        self.class_wise = class_wise

        self.poison_ratio = poison_ratio

        if (self.poison_path is not None) and (self.split in ['train', 'valid']):
            self.poison = torch.load(self.poison_path).detach().numpy()

        dgcnn_params = {
            'partition': 'train' if split in ['train', 'valid'] else 'test',
            'num_points': num_points,
            "data_path": self.data_path,
            'poison': self.poison,
            'class_wise': self.class_wise,
            'poison_ratio': self.poison_ratio
        }
        if poison_gen == 'shuffle':
            dgcnn_params['partition'] = 'poison_gen_shuf'
        elif poison_gen == 'standard':
            dgcnn_params['partition'] = 'test'
        elif poison_gen == 'only_poison' and (split in ['train', 'valid']):
            dgcnn_params['partition'] = 'test'
        elif poison_gen == 'only_poison' and (split in ['test']):
            dgcnn_params['partition'] = 'only_poison'

        self.dataset = dgcnn_ModelNet40(**dgcnn_params)


    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        pc, label, shuf_pc_ind = self.dataset.__getitem__(idx)
        return {'pc': pc, 'label': label.item(), 'index': idx, 'shuf_pc_ind': shuf_pc_ind}



def load_data(data_path,corruption,severity):

    DATA_DIR = os.path.join(data_path, 'data_' + corruption + '_' +str(severity) + '.npy')
    # if corruption in ['occlusion']:
    #     LABEL_DIR = os.path.join(data_path, 'label_occlusion.npy')
    LABEL_DIR = os.path.join(data_path, 'label.npy')
    all_data = np.load(DATA_DIR)
    all_label = np.load(LABEL_DIR)
    return all_data, all_label


import h5py
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


class ScanObjectNNDataLoader(Dataset):
    def __init__(self, split, poison_path, poison_gen, class_wise, poison_ratio, bg=False):

        self.splits = split
        self.folder = 'training' if self.splits in ['train', 'valid'] else 'test'
        self.poison_path = poison_path

        self.poison = None

        self.npoints = 1024

        if (self.poison_path is not None) and (self.splits in ['train', 'valid']):
            self.poison = torch.load(self.poison_path).detach().numpy()
        self.root = '../data/scanobject/h5_files'

        if bg:
            print('Use data with background points')
            dir_name = 'main_split'
        else:
            print('Use data without background points')
            dir_name = 'main_split_nobg'
        file_name = '_objectdataset.h5'
        h5_name = '{}/{}/{}'.format(self.root, dir_name, self.folder + file_name)
        with h5py.File(h5_name, mode="r") as f:
            self.data = f['data'][:].astype('float32')
            self.label = f['label'][:].astype('int64')
        print('The size of %s data is %d' % (split, self.data.shape[0]))


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        try:
            import random as rd
            np.random.seed(1)
            rd.seed(1)
            torch.manual_seed(1)
            pc = self.data[index]
            choice = np.random.choice(pc.shape[0], self.npoints, replace=False)
            pc = pc[choice, :]
            pc = pc[:, :3]
            if self.poison_path is not None and (self.splits in ['train', 'valid']):
                pc += self.poison[index]
        except:
            import random as rd
            np.random.seed(index)
            rd.seed(index)
            torch.manual_seed(index)
            pc = self.data[index]
            choice = np.random.choice(pc.shape[1], self.npoints, replace=False)
            pc = pc[:, choice]
            pc = pc[:, :, :3]
            if self.poison_path is not None and (self.splits in ['train', 'valid']):
                pc += self.poison[index]

        return {'pc': pc, 'label': self.label[index], 'index': index}



def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        # norm_pointcloud.dtype ="float32"
        return norm_pointcloud


class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                # ToTensor()
                              ])

def create_dataloader(split, cfg, cmd_args):
    num_workers = cfg.DATALOADER.num_workers
    batch_size = cfg.DATALOADER.batch_size

    if cfg.DATALOADER.poison_path is not None and cfg.EXP.poison_type in ['FC_EM','FC_EM_thre'] :
        poison_path = os.path.join(cfg.DATALOADER.poison_path, f'{cmd_args.temperature}_{cmd_args.eps}_{cmd_args.chamf_coeff}', f'poison_{cfg.EXP.poison_type}.pt')
    elif cfg.DATALOADER.poison_path is not None and cfg.EXP.poison_type in ['V_FC_EM'] :
        poison_path = os.path.join(cfg.DATALOADER.poison_path, f'{cmd_args.temperature}_{cmd_args.eps}', f'poison_{cfg.EXP.poison_type}.pt')
    elif cfg.DATALOADER.poison_path is not None and cfg.EXP.poison_type in ['EM', 'AP', 'AP_T']:
        poison_path = os.path.join(cfg.DATALOADER.poison_path, f'{cmd_args.eps}', f'poison_{cfg.EXP.poison_type}.pt')
    elif cfg.DATALOADER.poison_path is not None and cfg.EXP.poison_type in ['REG_EM', 'REG_AP', 'REG_AP_T'] :
        poison_path = os.path.join(cfg.DATALOADER.poison_path, f'{cmd_args.eps}_{cmd_args.chamf_coeff}', f'poison_{cfg.EXP.poison_type}.pt')
    elif cfg.DATALOADER.poison_path is not None:
        poison_path = os.path.join(cfg.DATALOADER.poison_path)
    else:
        poison_path = None
    print('poison path', poison_path)
    poison_gen = cfg.DATALOADER.poison_gen
    class_wise = cfg.EXP.class_wise
    poison_ratio = cmd_args.poison_ratio

    dataset_args = {
        "split": split,
        "poison_path": poison_path,
        "poison_gen": poison_gen,
        "class_wise": class_wise,
        "poison_ratio": poison_ratio
    }

    if cfg.EXP.DATASET == "modelnet40_dgcnn":
        dataset_args.update(dict(**cfg.DATALOADER.MODELNET40_DGCNN))
        dataset = ModelNet40Dgcnn(**dataset_args)
        num_points = dataset_args['num_points']
    elif cfg.EXP.DATASET == "scanobjectnn":
        dataset_args.update(dict(**cfg.DATALOADER.SCANOBJECTNN))
        dataset = ScanObjectNNDataLoader(**dataset_args)
        num_points = 1024
    else:
        assert False

    if "batch_proc" not in dir(dataset):
        dataset.batch_proc = None

    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        pin_memory=(torch.cuda.is_available()) and (not num_workers)
    ), len(dataset), num_points