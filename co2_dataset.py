import numpy as np
import torch
import os
import sys
from torch.utils.data import Dataset


class MateCo2Dataset(Dataset):
    def __init__(self, path, spt_sz = 12, TH=0, shuffle_task=True):
        self.path = path
        self.spt_sz = spt_sz
        self.datalist = os.listdir(path)
        self.lenth = len(self.datalist)
        self.TH = TH
        self.shuffle_task = shuffle_task

    def __getitem__(self, item):
        task_x=np.load(f"{self.path}/{self.datalist[item]}/x.npy")
        task_y=np.load(f"{self.path}/{self.datalist[item]}/y.npy")
        if self.shuffle_task:
            np.random.shuffle(task_x)
            np.random.shuffle(task_y)
        task_x=torch.tensor(task_x)
        task_y=torch.tensor(task_y)
        if(self.spt_sz>=len(task_x)):
            self.spt_sz=len(task_x)
            x_spt = task_x[0:self.spt_sz]
            y_spt = task_y[0:self.spt_sz]
            x_spt = x_spt[:,0,:]# num_sample, time_step, chennal, H ,W ,W
            y_spt = y_spt[:,self.TH,:]
            # y_spt = torch.tensor(y_spt)
            return  x_spt, y_spt, 0, 0

        x_spt = task_x[0:self.spt_sz]
        y_spt = task_y[0:self.spt_sz]
        x_spt = x_spt[:,0,:]# num_sample, time_step, chennal, H ,W ,W
        y_spt = y_spt[:,self.TH,:]
        # y_spt = torch.tensor(y_spt)
        x_qry = task_x[self.spt_sz-1:-1]
        y_qry = task_y[self.spt_sz-1:-1]
        x_qry = x_qry[:,0,:]
        y_qry = y_qry[:,self.TH,:]
        # y_qry = torch.tensor(y_qry)
        return x_spt, y_spt, x_qry, y_qry

    def __len__(self):
        return self.lenth
    
#Slow but low memory cost
class NomalCo2Dataset_loadOne(Dataset):
    def __init__(self, path, TH=0):
        self.path = path
        self.TH = TH
        self.tasklist = os.listdir(path)
        
        self.min_samples_each_task=sys.maxsize
        for task in self.tasklist:
            task_x=np.load(f"{self.path}/{task}/x.npy")
            if(self.min_samples_each_task>=task_x.shape[0]):
                self.min_samples_each_task=task_x.shape[0]
        self.lenth = len(self.tasklist)*self.min_samples_each_task
        print(f"Avaliable samples: {self.lenth}   Sample number in each task:{self.min_samples_each_task}")

    def __getitem__(self, item):
        task_item=item//self.min_samples_each_task
        samlple_item=item%self.min_samples_each_task
        task_x=np.load(f"{self.path}/task{task_item}/x.npy")
        task_y=np.load(f"{self.path}/task{task_item}/y.npy")
        x= task_x[samlple_item,0,:]
        y= task_y[samlple_item,self.TH,:]
  
        return x, y
    
    def __len__(self):
        return self.lenth
    
#High memory cost
class NomalCo2Dataset_loadALL(Dataset):
    def __init__(self, path, TH=0):
        self.path = path
        self.TH = TH
        self.tasklist = os.listdir(path)
        if(len(self.tasklist)==2):
            self.x=np.load(f"{self.path}/all_x.npy")
            self.y=np.load(f"{self.path}/all_y.npy")
        else:
            x=[]
            y=[]
            for task in self.tasklist:
                self.x=np.load(f"{self.path}/{task}/x.npy")
                self.y=np.load(f"{self.path}/{task}/y.npy")
                x.append(self.x)
                y.append(self.y)
            if(len(x)!=1):
                self.x=np.concatenate(x,axis=0)
                self.y=np.concatenate(y,axis=0)
            os.makedirs("./data_normal/",exist_ok=True)
            np.save("./data_normal/all_x.npy", self.x)
            np.save("./data_normal/all_y.npy", self.y)
                
        self.lenth = self.x.shape[0]
        print(f"Avaliable samples: {self.lenth}")
    def __getitem__(self, item):
        return self.x[item,0,:], self.y[item,self.TH,:]
    
    def __len__(self):
        return self.lenth
    


if __name__=='__main__':

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    epochs_outer=10
    epochs_inner=16
    lr_outer=0.001
    lr_inner=0.01
    
    batch_size_outer = 8
    batch_size_inner = 4

    testSet=NomalCo2Dataset_loadALL("./dataSet")
