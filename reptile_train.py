import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from model import*
from torch.utils.data import DataLoader
from co2_dataset import *
from torch.autograd import Variable
from loss import *

def inner_loop(model,lr_inner, epoch_inner,task_i, batch_size_inner, device="cpu"):
    #net.train()
    new_model = build_resunetplusplus()
    new_model.load_state_dict(model.state_dict())
    new_model = new_model.to(device)
    inner_optim = torch.optim.Adam(new_model.parameters(), lr=lr_inner)
    for ie in range(epoch_inner):
        for batch_start in range(0, len(task_i), batch_size_inner):
            inner_optim.zero_grad()
            if (batch_start+batch_size_inner>len(task_i)):
                batch_end=len(task_i)
            else:
                batch_end=batch_start+batch_size_inner
            x = Variable(task_i[0][0,batch_start:batch_end].to(device))
            y = Variable(task_i[1][0,batch_start:batch_end].to(device))
            y_pred = new_model(x.float())
            loss = using_loss(y_pred, y.float())
            loss.backward()
            inner_optim.step()
        return new_model,loss.data.cpu().numpy()

def inner_eval(net, task, device="cpu"):
    x_eval=task[2][0].to(device) 
    y_eval=task[3][0].to(device)
    y_pred = net(x_eval.float())
    loss = using_loss(y_pred, y_eval.float())
    r2 = loss_r2(y_pred,y_eval.float()).detach()
    return loss.data.cpu().numpy(),r2.data.cpu().numpy()

def outer_eval(model,inner_epoch,task_loader,batch_size_inner=4, lr_inner=0.01,  device="cpu"):
    loss_eval_ls=[]
    for i,task_i in enumerate(task_loader,0):
        taski_eval_ls=[]
        new_model = build_resunetplusplus()
        new_model.load_state_dict(model.state_dict())
        new_model = new_model.to(device)
        inner_optim = torch.optim.Adam(new_model.parameters(), lr=lr_inner)
        for epoch in range(inner_epoch):
            for batch_start in range(0, len(task_i), batch_size_inner):
                if (batch_start+batch_size_inner>len(task_i)):
                    batch_end=len(task_i)
                else:
                    batch_end=batch_start+batch_size_inner
                x = task_i[0][0,batch_start:batch_end].to(device)
                y = task_i[1][0,batch_start:batch_end].to(device)
                y_pred = new_model(x.float())
                loss = using_loss(y_pred, y.float())
                inner_optim.zero_grad()
                loss.backward()
                inner_optim.step()
            eval_loss,r2=inner_eval(new_model,task_i,device)
            # print(f" eval inner_epoch{epoch} loss:{eval_loss} r2:{r2}")
            taski_eval_ls.append(eval_loss)
        loss_eval_ls.append(taski_eval_ls)
    loss_eval=np.array(loss_eval_ls)
    loss_eval=np.mean(loss_eval,axis=0)
    return loss_eval,r2

def outter_loop(model, task_loader, epochs_outer, epochs_inner, eval_loader,lr_outer=0.001, batch_size_outer=32, lr_inner=0.01, batch_size_inner=4, device="cpu",save_path=None):
    out_loss_ls=[]
    k=epochs_inner
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_outer)
    if epochs_outer>len(task_loader):
        epochs_outer=len(task_loader)
    for e,task in enumerate(task_loader):
        new_model,loss=inner_loop(model,lr_inner,k,task,batch_size_inner,device=device)
        name_to_param = dict(model.named_parameters())
        for name, param in new_model.named_parameters():
            cur_grad = (name_to_param[name].data - param.data)/k
            if name_to_param[name].grad is None:
                name_to_param[name].grad = Variable(torch.zeros(cur_grad.size())).to( param.grad.device)
            name_to_param[name].grad.data.add_(cur_grad / batch_size_outer)
        if (e+1)%batch_size_outer==0:
            optimizer.step()
            optimizer.zero_grad()
            if len(eval_loader)!=0:
                loss_evel,r2 = outer_eval(model,k,eval_loader,batch_size_inner,lr_inner,device=device)
                print(f"outer_epoch&task{e}  eval_loss:{loss_evel[-1]} r2{r2}")
                out_loss_ls.append(loss_evel)###
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return np.array(out_loss_ls)

