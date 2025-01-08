import matplotlib as mpl
import torch
mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('bmh')

from model1203 import*

from torch.utils.data import DataLoader
from co2_dataset import MateCo2Dataset


BATCH_SIZE = 1
EPOCH = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def inner_eval(model,x_eval,y_eval):
    model.eval()
    lossFunc = IdxLoss()
    y_pred = model(x_eval.float())
    loss=lossFunc(y_pred,y_eval)
    return loss.data.cpu()

def nomal_loop(model,inner_epoch,task_i,batch_size_inner=4, inner_lr=0.01, create_graph=False,device="cpu"):
    model.train()
    loss_ls=[]
    loss_eval_ls=[]
    lossFunc = IdxLoss()
    inner_optim = torch.optim.SGD(model.parameters(), lr=inner_lr)
    for epoch in range(inner_epoch):
        for batch_start in range(0, len(task_i), batch_size_inner):
            if inner_optim is not None:
                inner_optim.zero_grad()
            if (batch_start+batch_size_inner>len(task_i)):
                batch_end=len(task_i)
            else:
                batch_end=batch_start+batch_size_inner
            x = task_i[0][0,batch_start:batch_end].to(device)
            y = task_i[1][0,batch_start:batch_end].to(device)
            y_pred = model(x.float())
            loss = lossFunc(y_pred, y)
            loss.backward(create_graph=create_graph, retain_graph=True)
            loss_ls.append(loss.data.cpu())
            if inner_optim is not None:
                inner_optim.step()
            if len(task_i[2].size())is not 1:
                x_eval = task_i[2][0,batch_start:batch_end].to(device)
                y_eval = task_i[3][0,batch_start:batch_end].to(device)
                loss_eval=inner_eval(model,x_eval,y_eval)
                loss_eval_ls.append(loss_eval)
    return loss_ls,loss_eval_ls

if __name__=='__main__':

    testSet=MateCo2Dataset("./dataSet", 2)
    train_loader = DataLoader(testSet, 1, True, num_workers=1)
    model = build_resunetplusplus()
    model = model.to(DEVICE)
    inner_loop(model,EPOCH,train_loader,BATCH_SIZE,0.001,if_eval=False,device=DEVICE)


# if __name__=='__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)

#     testSet=MateCo2Dataset("./dataSet", 2, 1)
#     model = build_resunetplusplus()
#     model = model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     train_loader = DataLoader(testSet, 1, True, num_workers=1)
#     loss = IdxLoss()
#     losses = []
#     # acces=[]

#     for i in range(EPOCH):
#         print(f"epoch:{i}\n")
#         for batch_idx,data in enumerate(train_loader, 0):
#             start_time = time.time()
#             [x_spt, y_spt, x_qry, y_qry] = data
#             num_sample=x_spt.shape[0]
#             x_spt = x_spt[0].to(device).float()#batach_task, num_sample,  chennal, H ,W ,W
#             y_spt = y_spt[0].to(device).float()#batach_task, num_sample, time_step, chennal, H ,W ,W
#             y_spt_pred = model(x_spt)
#             loss = loss(y_spt_pred, y_spt)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())
#             print(loss)

