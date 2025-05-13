import torch.nn.functional as F
import torch
import pytorch_ssim
from torchmetrics.regression import R2Score
# import math
# def th_loss(y_pred, y):
#     loss_H=F.mse_loss(y_pred[:,0,:], y[:,0,:])
#     loss_T=F.mse_loss(y_pred[:,1,:], y[:,1,:])
#     ne=int(math.log10(loss_T/loss_H))
#     a=10**(-ne)
#     b=(1-a)
#     loss_T=a*loss_T
#     loss_H=b*loss_H
#     return loss_T+loss_H

def loss_mse(y_pred, y):
    return F.mse_loss(y_pred, y)

def loss_mea(y_pred, y):
    return torch.mean(torch.abs(y_pred-y))

def r_squared(y_pred, y_true):
    # 展平张量，处理任意形状的输入
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # 计算残差平方和与总平方和
    ss_res = torch.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = torch.sum((y_true_flat - torch.mean(y_true_flat)) ** 2)
    
    # 处理总平方和为0的情况
    ss_tot_is_zero = (ss_tot == 0)
    ss_res_is_zero = (ss_res == 0)
    
    # 使用torch.where进行条件判断，保持梯度计算
    r2 = torch.where(
        ss_tot_is_zero,
        torch.where(ss_res_is_zero, 
                    torch.tensor(1.0, device=y_pred.device), 
                    torch.tensor(-float('inf'), device=y_pred.device)),
        1 - (ss_res / ss_tot)
    )
    
    return r2

def loss_r2(y_pred, y):
    loss=1-torch.sum(( y_pred - y)**2) / torch.sum((y - torch.mean(y))**2)
    return loss

def loss_ssim3d(y_pred, y, window_size):
    ssim_loss = pytorch_ssim.SSIM3D(window_size)
    s_loss=ssim_loss(y_pred, y.float())
    return 1-s_loss

def loss_combine(y_pred, y, window_size,  sc):
    s_loss= loss_ssim3d(y_pred, y, window_size)
    m_loss = loss_mse(y_pred, y)
    # c_loss=sc*m_loss
    c_loss=sc * (m_loss + s_loss/((s_loss/m_loss).detach()))
    # c_loss=sc*s_loss
    return c_loss

def all_loss(y_pred, y):
    loss_m = loss_mse(y_pred, y)
    loss_me = loss_mea(y_pred, y)
    loss_r = loss_r2(y_pred, y)
    loss_s = loss_ssim3d(y_pred, y, 3)
    return loss_m, loss_me, loss_r, loss_s

def using_loss(y_pred, y,sc=100,wz=5):
    # return loss_combine(y_pred, y,window_size=wz,sc=sc)
    return loss_mse(y_pred, y)