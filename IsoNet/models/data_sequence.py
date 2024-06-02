import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import random


class Train_sets(Dataset):
    def __init__(self, paths, shuffle=True):
        super(Train_sets, self).__init__()
        zipped_path = list(map(list, zip(*paths)))
        if shuffle:
            np.random.shuffle(zipped_path)
        self.path_all = zipped_path
        # print(self.path_all[0])
        # print('\n')
        # print(self.path_all[1])
        # if shuffle:
        #     zipped_path = list(zip(self.path_all[0],self.path_all[1]))
        #     np.random.shuffle(zipped_path)
        #     self.path_all[0], self.path_all[1] = zip(*zipped_path)
        # print(self.path_all[0])
        # print('\n')
        # print(self.path_all[1])
        #if max_length is not None:
        #    if max_length < len(self.path_all):

    def __getitem__(self, idx):

        # with mrcfile.open(self.path_all[0][idx]) as mrc:
        #     x = mrc.data[np.newaxis,:,:,:]
        # with mrcfile.open(self.path_all[1][idx]) as mrc:
        #     y = mrc.data[np.newaxis,:,:,:]

        # rx = torch.as_tensor(x.copy())
        # ry = torch.as_tensor(y.copy())
        results = []
        #l = list(zip(self.path_all[0],self.path_all[1]))
        for i,p in enumerate(self.path_all[idx]):
            with mrcfile.open(p) as mrc:
                x = mrc.data[np.newaxis,:,:,:]
            x = torch.as_tensor(x.copy())
            results.append(x)
        return results

    def __len__(self):
        return len(self.path_all)

class Train_sets_backup(Dataset):
    def __init__(self, data_dir, max_length = None, shuffle=True, beta=0.5, prefix = "train"):
        super(Train_sets, self).__init__()
        self.beta=beta
        self.path_all = []
        for d in  [prefix+"_x1", prefix+"_y1", prefix+"_x2", prefix+"_y2"]:
            p = '{}/{}/'.format(data_dir, d)
            self.path_all.append(sorted([p+f for f in os.listdir(p)]))
        # shuffle=False
        # if shuffle:
        #     zipped_path = list(zip(self.path_all[0],self.path_all[1]))
        #     np.random.shuffle(zipped_path)
        #     self.path_all[0], self.path_all[1] = zip(*zipped_path)
        #if max_length is not None:
        #    if max_length < len(self.path_all):


    def __getitem__(self, idx):

        with mrcfile.open(self.path_all[0][idx]) as mrc:
            x1 = mrc.data[np.newaxis,:,:,:]
        with mrcfile.open(self.path_all[1][idx]) as mrc:
            y1 = mrc.data[np.newaxis,:,:,:]
        with mrcfile.open(self.path_all[2][idx]) as mrc:
            x2 = mrc.data[np.newaxis,:,:,:]
        with mrcfile.open(self.path_all[3][idx]) as mrc:
            y2 = mrc.data[np.newaxis,:,:,:]

        random_number = random.random()
        random_number2 = random.random()
        # x = y1
        # y = y2
        if random_number<self.beta:
            if random_number2>0.5:
                x = x1
                y = y2
            else:
                x = x2
                y = y1
        else:
            if random_number2>0.5:
                x = x1
                y = y1
            else:
                x = x2
                y = y2
        rx = torch.as_tensor(x.copy())
        ry = torch.as_tensor(y.copy())
        return rx, ry

    def __len__(self):
        return len(self.path_all[0])

class Predict_sets(Dataset):
    def __init__(self, mrc_list, inverted=True):
        super(Predict_sets, self).__init__()
        self.mrc_list=mrc_list
        self.inverted = inverted

    def __getitem__(self, idx):
        with mrcfile.open(self.mrc_list[idx]) as mrc:
            rx = mrc.data[np.newaxis,:,:,:].copy()
        # rx = mrcfile.open(self.mrc_list[idx]).data[:,:,:,np.newaxis]
        if self.inverted:
            #rx=normalize(-rx, percentile = True)
            rx=-rx
        return rx

    def __len__(self):
        return len(self.mrc_list)



def get_datasets(data_dir, max_length = None):
    train_dataset = Train_sets(data_dir, max_length, prefix="train")
    val_dataset = Train_sets(data_dir, max_length, prefix="test")
    return train_dataset, val_dataset#, bench_dataset