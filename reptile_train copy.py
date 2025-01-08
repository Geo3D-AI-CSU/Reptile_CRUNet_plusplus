import numpy as np
from model1203 import*
from torch.utils.data import DataLoader
from co2_dataset import MateCo2Dataset
from torch.autograd import Variable
from normal_train import inner_eval
import torch.nn.functional as F

def inner_loop(model, task_i,epoch_inner,batch_size_inner,device):
    new_model = build_resunetplusplus()
    new_model.load_state_dict(model.state_dict())
    new_model = new_model.to(device)
    inner_optim = torch.optim.SGD(new_model.parameters(), lr=lr_inner)
    for i in range(epoch_inner):
        for batch_start in range(0, len(task_i), batch_size_inner):
            if (batch_start+batch_size_inner>len(task_i)):
                batch_end=len(task_i)
            else:
                batch_end=batch_start+batch_size_inner
            x = Variable(task_i[0][0,batch_start:batch_end].to(DEVICE))
            y = Variable(task_i[1][0,batch_start:batch_end].to(DEVICE))
            y_pred = model(x.float())
            loss = F.mse_loss(y_pred, y.float())
            inner_optim.zero_grad()
            loss.backward()
            inner_optim.step()
    return new_model

def inner_eval(model, task_i, batch_size_inner):
    # net.eval()
    for batch_start in range(0, len(task_i), batch_size_inner):
        if (batch_start+batch_size_inner>len(task_i)):
            batch_end=len(task_i)
        else:
            batch_end=batch_start+batch_size_inner
        x = Variable(task_i[2][0,batch_start:batch_end].to(DEVICE))
        y = Variable(task_i[3][0,batch_start:batch_end].to(DEVICE))
        y_pred = model(x.float())
        loss = F.mse_loss(y_pred, y.float())
    return loss.data.cpu().numpy()

def outer_eval(model,inner_epoch,task_i,batch_size_inner=4, inner_lr=0.01, create_graph=False, if_eval=True, device="cpu"):
    model.train()
    loss_eval_ls=[]
    loss_eval = 0.0
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
            if inner_optim is not None:
                inner_optim.step()
        if if_eval:
            for epoch in range(inner_epoch):
                x_eval = task_i[2][0,batch_start:batch_end].to(device)
                y_eval = task_i[3][0,batch_start:batch_end].to(device)
                loss_eval_ls.append[inner_eval(model,x_eval,y_eval)]
            loss_eval=(sum(loss_eval_ls)/len(loss_eval_ls))
    return loss.data.cpu().numpy()[0],loss_eval


def outter_loop(model, task_loader, epochs_outer, epochs_inner, eval_loader,lr_outer=0.001, lr_inner=0.01, batch_size_inner=4, device="cpu"):
    loss_ls=[]
    loss_evel_ls=[]
    loss_eval2_ls=[]
    k=epochs_inner
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)
    for e in range(epochs_outer):
        task = next(iter(task_loader))
        new_model =inner_loop(model,task,epochs_inner,batch_size_inner,device=device)
        loss=inner_eval(new_model,task,batch_size_inner)
        print(f"outer_epoch{e} mse_loss:{loss}")
        for p, new_p in zip(model.parameters(), new_model.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size()))
            p.grad.data.add_(p.data - new_p.data)
        optimizer.step()
        optimizer.zero_grad()
            
        if not len(eval_loader)==0:
            new_model = build_resunetplusplus()
            new_model.load_state_dict(model.state_dict())
            new_model = new_model.to(device)
            loss_evel,loss_eval2=outer_eval(new_model,k,eval_loader,batch_size_inner,lr_inner,device=device)
            loss_evel_ls.append(loss_evel)###
            loss_eval2_ls.append(loss_eval2)###
 

if __name__=='__main__':

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    epochs_outer=20
    epochs_inner=6
    lr_outer=0.001
    lr_inner=0.01
    
    batch_size_outer = 1
    batch_size_inner = 1

    testSet=MateCo2Dataset("./dataSet", 12)
    model = build_resunetplusplus()

    model = model.to(DEVICE)
    train_loader = DataLoader(testSet, 1, True)
    eval_loader = []
    outter_loop(model, train_loader, epochs_outer, epochs_inner, eval_loader, lr_outer, lr_inner, batch_size_inner,device=DEVICE)

