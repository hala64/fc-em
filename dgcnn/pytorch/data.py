#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../../data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(data_path, main=False):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../../data')
    all_data = []
    all_label = []
    try:
        with open(data_path, "r") as f:
            # for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
            #print(f.readlines())
            for h5_name in f.readlines():
                #print(h5_name)
                if main:
                    h5_name = os.path.join(BASE_DIR, "../../", h5_name.strip())
                #print(h5_name)
                #print(BASE_DIR)
                #print('===')
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                all_data.append(data)
                all_label.append(label)
        #print(all_data)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    except:
        pass

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, data_path, partition='test', poison=None, class_wise=False, poison_ratio=1.0, main=False):
        if main:
            print(data_path)
        self.data, self.label = load_data(data_path, main=main)
        self.num_points = num_points
        self.partition = partition
        if poison is not None:
            if not class_wise:
                if poison_ratio != 1.0:
                    assert 0.0 <= poison_ratio <= 1.0
                    poison_index = np.random.choice(len(self.data), int((1 - poison_ratio) * len(self.data)), replace=False)
                    poison[poison_index, :, :] = 0.0
                #assert self.num_points == len(poison[0])
                if self.partition == 'only_poison':
                    self.data[:, :self.num_points, :] = poison[:, :self.num_points, :]
                else:
                    self.data[:, :self.num_points, :] += poison[:, :self.num_points, :]
                    self.data[:, :self.num_points, :] = np.clip(self.data[:, :self.num_points, :], -1., 1.)
            else:
                assert len(poison) == 40
                assert self.num_points == len(poison[0])
                target = np.array(self.label)
                target = target.ravel().tolist()
                if self.partition == 'only_poison':
                    self.data[:, :self.num_points, :] = poison[target, :self.num_points, :]
                else:
                    self.data[:, :self.num_points, :] += poison[target, :self.num_points, :]
                    self.data[:, :self.num_points, :] = np.clip(self.data[:, :self.num_points, :], -1., 1.)


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        ind = np.arange(len(pointcloud))
        #print(label)
        #raise ValueError
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(ind)
            pointcloud = pointcloud[ind]
            #np.random.shuffle(pointcloud)
        if self.partition == 'poison_gen_shuf':
            np.random.shuffle(ind)
            pointcloud = pointcloud[ind]
        return pointcloud, label, ind

    def __len__(self):
        return self.data.shape[0]
