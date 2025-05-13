
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from model import*
from torch.utils.data import DataLoader
from co2_dataset import NomalCo2Dataset_loadALL
from loss import using_loss,loss_r2,all_loss


def normal_eval(model,eval_loader, device="cpu"):
    model.eval()
    loss=[]
    for i, batch_eval in enumerate(eval_loader, 0):
        x_eval = batch_eval[0].to(device)
        y_eval = batch_eval[1].to(device)
        y_pred = model(x_eval.float())
        loss_i=using_loss(y_pred,y_eval.float())
        mse,mea,r2,ssim=all_loss(y_pred,y_eval.float())
        loss.append(loss_i.data.cpu())
    return mse.data.cpu(),mea.data.cpu(),r2.data.cpu(),ssim.data.cpu()

def save_model(model, path, num_cuda=1):
    if num_cuda>1:
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

def normal_train(model, dataset, eval_rate, epochs, batch_size, lr=0.001, device="cpu", save_path=None):
    loss_ls = []
    mea_ls=[]
    r2_ls = []
    ssim_ls=[]
    lenth=len(dataset)
    
    eval_size=int(lenth*eval_rate)
    train_size = lenth-eval_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, eval_size])
    data_loader = DataLoader(train_dataset, batch_size,True)

    dataset_=NomalCo2Dataset_loadALL("./target_task/data_normal")
    selected=[33,34,36,37,38,51,52,53,58,59,61,62]
    eval_dataset=torch.utils.data.Subset(dataset_,selected)
    eval_loader = DataLoader(eval_dataset, batch_size, True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        model.train()
        for i, batch in enumerate(data_loader, 0):
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(x.float())
            loss=using_loss(y_pred, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"epoch{e} s{i} loss:{loss.data.cpu().numpy()}")
        if len(eval_loader)!=0:
            loss_eval,mea,r2,ssim=normal_eval(model,eval_loader,device)
            loss_ls.append(loss_eval)
            mea_ls.append(mea)
            r2_ls.append(r2)
            ssim_ls.append(ssim)
            print(f"epoch{e}  eval loss:{loss_eval} r2:{r2}")
    if save_path is not None:
        save_model(model, save_path)
    return np.array(loss_ls),np.array(mea_ls),np.array(r2_ls),np.array(ssim_ls)


# if __name__=='__main__':

#     DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(DEVICE)

#     dataset=NomalCo2Dataset_loadALL("./target_task/data_meta")

#     model = build_resunetplusplus()
#     model = model.to(DEVICE)
#     normal_train(model, dataset, 0.25, EPOCH, BATCH_SIZE, LR, DEVICE)


