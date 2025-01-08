import numpy as np
import torch
import os
from torch.utils.data import Dataset


class MateCo2Dataset(Dataset):
    def __init__(self, path, spt_sz = 12):
        self.path = path
        self.spt_sz = spt_sz
        self.datalist = os.listdir(path)
        self.lenth = len(self.datalist)

    def __getitem__(self, item):
        task_x=np.load(f"{self.path}/task{item}/x.npy")
        task_y=np.load(f"{self.path}/task{item}/y.npy")
        
        if(self.spt_sz>=len(task_x)):
            self.spt_sz=len(task_x)
            x_spt = task_x[0:self.spt_sz]
            y_spt = task_y[0:self.spt_sz]
            x_spt = torch.tensor(x_spt[:,0,:])# num_sample, time_step, chennal, H ,W ,W
            y_spt = torch.tensor(y_spt[:,0,:])
            return  x_spt, y_spt, 0, 0

        x_spt = task_x[0:self.spt_sz]
        y_spt = task_y[0:self.spt_sz]
        x_spt = torch.tensor(x_spt[:,0,:])# num_sample, time_step, chennal, H ,W ,W
        y_spt = torch.tensor(y_spt[:,0,:])
        x_qry = task_x[self.spt_sz-1:-1]
        y_qry = task_y[self.spt_sz-1:-1]
        x_qry = torch.tensor(x_qry[:,0,:])
        y_qry = torch.tensor(y_qry[:,0,:])
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return self.lenth

class NomalCo2Dataset(Dataset):
    def __init__(self, path, spt_sz = 12, qry_sz = 4):
        self.path = path
        self.spt_sz = spt_sz
        self.qry_sz = qry_sz
        self.datalist = os.listdir(path)
        self.lenth = len(self.datalist)

    def __getitem__(self, item):
        task_x=np.load(f"{self.path}/task{item}/x.npy")
        task_y=np.load(f"{self.path}/task{item}/y.npy")
        x_spt = task_x[0:self.spt_sz]
        y_spt = task_y[0:self.spt_sz]
        x_qry = task_x[self.spt_sz-1:-1]
        y_qry = task_x[self.spt_sz-1:-1]
        x_spt = torch.tensor(x_spt[:,0,:])# num_sample, time_step, chennal, H ,W ,W
        y_spt = torch.tensor(y_spt[:,0,:])
        x_qry = torch.tensor(x_qry[:,0,:])
        y_qry = torch.tensor(y_qry[:,0,:])
        return x_spt, y_spt, x_qry, y_qry