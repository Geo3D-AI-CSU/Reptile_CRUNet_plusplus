
#%%
import matplotlib.pyplot as plt
import numpy as np
import gstools as gs
import random
import time

"""
任务的差别-》深度 分布 角度
    分布-》差别不大-》存在负值
    角度-》不好解释
    地质统计学——》数学课 文献

"""
import os
#%%
# seedt = int(time.time())


def ShowModel(X, Y, Z, Val, if_show=False, path=None):
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1, 1, 0.2])
    scatter=ax.scatter(X[:], Y[:], Z[:],s=100,c=Val[:], marker='.', cmap='jet')
    ax.set_title('porperty')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.colorbar(scatter)
    if path is not None:
        name=path.split("/")[-1]
        len_name=len(name)
        folder=path[:-len_name]
        os.makedirs(folder,exist_ok=True)
        plt.savefig(path,dpi=500)
    if if_show:
        plt.show()
    plt.close()

def GetCondPos(x,y,z,data,n_cond_pos,if_random=False):
    n_data=data.shape[0]
    min_cols=np.min(data,axis=0)
    max_cols=np.max(data,axis=0)
    cond_x=np.random.choice(x,n_cond_pos)
    cond_y=np.random.choice(y,n_cond_pos)
    cond_z=np.random.choice(z,n_cond_pos)
    a=np.arange(0, n_data, 1)
    if if_random:
        # idx_cond_pos = random.sample(a.tolist(),n_cond_pos)
        # random
        # cond_pos=data[idx_cond_pos,0:3]
        # cond_val=data[idx_cond_pos,3:5]
        idx_cond_pos = random.sample(a.tolist(),n_cond_pos)
        cond_pos=np.array([cond_x,cond_y,cond_z]).T
        cond_val=data[idx_cond_pos,3:5]
    else:
        idx_cond_pos = np.linspace(0, n_data, n_cond_pos, dtype=np.int16)
        cond_pos=data[idx_cond_pos,0:3]
        cond_val=data[idx_cond_pos,3:5]       
    return cond_pos,cond_val

def GetCondGrd(cond_pos, cond_val, x, y, z, model):
    cond_x = cond_pos[:,0].tolist()
    cond_y = cond_pos[:,1].tolist()
    cond_z = cond_pos[:,2].tolist()
    cond_val = cond_val.tolist()
    krige = gs.Krige(model, cond_pos=[cond_x, cond_y, cond_z],cond_val=cond_val)
    cond_grd = gs.CondSRF(krige)
    cond_grd.set_pos([x, y, z], "structured")
    return cond_grd

def GetWellxy(x, y):
    idx_xi, well_xi= random.sample(list(enumerate(x)), 1)[0]
    idx_yi, well_yi = random.sample(list(enumerate(y)), 1)[0]
    well_data=np.array([well_xi, well_yi, idx_xi, idx_yi])
    return well_data

def ChangeRange(cond_val, val):
    cond_range=[np.min(cond_val),np.max(cond_val)]
    val_range=[np.min(val.flatten()),np.max(val.flatten())]
    val=(val-val_range[0])*((cond_range[1]-cond_range[0])/(val_range[1]-val_range[0]))+cond_range[0]
    return val, cond_range, val_range

def GetSingleTask(x,y,z,n_samples, steps, t_stops, inj_rate,poro_cond_grd, perm_cond_grd, cond_val,cond_pos,task_id=0):
    path=f"./tasks/task{task_id}"
    os.makedirs(path,exist_ok=True)
    seedt=20240621
    seed = gs.random.MasterRNG(seedt)
    with open(f"{path}/discription.txt","a") as f:
        f.write(f"task{task_id}_discription: \n")
    for i in range(n_samples):
        poro_cond_grd(seed=seed(), store=[f"fld{i}", False, False])
        perm_cond_grd(seed=seed(), store=[f"fld{i}", False, False])    
    X,Y,Z = np.meshgrid(x, y, z)
    for i in range(n_samples):
        well_data=GetWellxy(x, y)
        print("sim_%d"%(i))
        poro_val = poro_cond_grd[i]
        perm_val = perm_cond_grd[i]
        np.save("poro_val.npy",poro_val)
        np.save("perm_val.npy",perm_val)
        [poro_val,poro_cond_range,poro_val_range]=ChangeRange(cond_val[:,0],poro_val)
        [perm_val,perm_cond_range,perm_val_range]=ChangeRange(cond_val[:,1],perm_val)
        with open(f"{path}/log.txt","a") as f:
            f.write("T:%d S:%d PORO_COND mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(cond_val[:,0]),np.var(cond_val[:,0]),poro_cond_range[0],poro_cond_range[1]))
            f.write("T:%d S:%d PERM_COND mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(cond_val[:,1]),np.var(cond_val[:,1]),perm_cond_range[0],perm_cond_range[1]))
            f.write("T:%d S:%d uPORO_VAL mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(poro_val.flatten()),np.var(poro_val.flatten()),poro_val_range[0],poro_val_range[1]))
            f.write("T:%d S:%d uPERM_VAL mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(perm_val.flatten()),np.var(perm_val.flatten()),perm_val_range[0],perm_val_range[1]))
            f.write("T:%d S:%d cPORO_VAL mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(poro_val.flatten()),np.var(poro_val.flatten()),np.min(poro_val.flatten()),np.max(poro_val.flatten())))
            f.write("T:%d S:%d cPERM_VAL mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(perm_val.flatten()),np.var(perm_val.flatten()),np.min(perm_val.flatten()),np.max(perm_val.flatten())))
            print("T%d S%d PORO_COND mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(cond_val[:,0]),np.var(cond_val[:,0]),poro_cond_range[0],poro_cond_range[1]))
            print("T%d S%d PERM_COND mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(cond_val[:,1]),np.var(cond_val[:,1]),perm_cond_range[0],perm_cond_range[1]))
            print("T%d S%d uPORO_VAL  mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(poro_val.flatten()),np.var(poro_val.flatten()),poro_val_range[0],poro_val_range[1]))
            print("T%d S%d uPERM_VAL  mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(perm_val.flatten()),np.var(perm_val.flatten()),perm_val_range[0],perm_val_range[1]))
            print("T%d S%d PORO_VAL  mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(poro_val.flatten()),np.var(poro_val.flatten()),np.min(poro_val.flatten()),np.max(poro_val.flatten())))
            print("T%d S%d PERM_VAL  mean:%.6f , var:%.6f , min:%.6f, max:%.6f \n"%(task_id,i,np.mean(perm_val.flatten()),np.var(perm_val.flatten()),np.min(perm_val.flatten()),np.max(perm_val.flatten())))
        
        np.savetxt(f"{path}/well{i}.csv", well_data, delimiter=",")
        ShowModel(X, Y, Z, poro_val, False,  f"{path}/poro{i}.png")
        ShowModel(X, Y, Z, perm_val, False,  f"{path}/perm{i}.png")

        hfs_data=np.concatenate([X.flatten()[:, np.newaxis], 
                                Y.flatten()[:, np.newaxis], 
                                Z.flatten()[:, np.newaxis], 
                                poro_val.flatten()[:, np.newaxis], 
                                perm_val.flatten()[:, np.newaxis]],
                                axis=1)
        np.savetxt(f"{path}/property{i}.csv", hfs_data, delimiter=",")

        dataList=[]
        for k in steps:
            well_grd = np.zeros(poro_val.shape)
            if k<=  t_stops:
                well_grd[well_data[2], well_data[3], :]=inj_rate
            data_ti=np.concatenate(
                [
                    poro_val[np.newaxis, :],
                    perm_val[np.newaxis, :],
                    well_grd[np.newaxis, :]
                ],
                axis=0
            )
            dataList.append(data_ti[np.newaxis, :])
        train_data=np.concatenate(dataList,axis=0)
        np.save(f"{path}/property{i}.npy", train_data)

def GetAllTask(x,y,z,task_num, sample_num, steps, t_stops, inj_rate, poro_var, perm_var, len_scale,cond_point_num=10):
    seedt = (time.localtime(time.time()))
    # log_file_name=f"log_conddata_{seedt.tm_year}.{seedt.tm_mon}.{seedt.tm_mday}_{seedt.tm_hour}{seedt.tm_min}.txt"
    # with open(log_file_name,"a") as f:
    #     f.write()
    file="./carbon_dioxide_storage_porper.csv"
    data=np.loadtxt(file, skiprows=1, delimiter=",")
    all_cond_data=np.empty((cond_point_num,5))
    # for i in range(100,256):
    for i in range(task_num):
        cond_pos,cond_val=GetCondPos(x,y,z,data, cond_point_num, if_random=True)
        cond_data=np.hstack((cond_pos,cond_val))
        if i==0:
            all_cond_data=cond_data
        else:
            all_cond_data=np.dstack((all_cond_data,cond_data))
        model_poro = gs.Gaussian(dim=3, var=poro_var, len_scale=len_scale)
        model_perm = gs.Gaussian(dim=3, var=perm_var, len_scale=len_scale)
        poro_cond_grd=GetCondGrd(cond_pos, cond_val[:,0], x, y, z, model_poro)
        perm_cond_grd=GetCondGrd(cond_pos, cond_val[:,1], x, y, z, model_perm)
        ShowModel(cond_pos[:,0], cond_pos[:,1], cond_pos[:,2],cond_val[:,0], if_show=False, path=f"./cond_data/fig/cond_poro{i}.png")
        ShowModel(cond_pos[:,0], cond_pos[:,1], cond_pos[:,2],cond_val[:,1], if_show=False, path=f"./cond_data/fig/cond_perm{i}.png")
        GetSingleTask(x,y,z,sample_num, steps, t_stops, inj_rate, poro_cond_grd, perm_cond_grd, cond_val,cond_pos,i)
    allcond_file_name=f"./cond_data/all_conddata_{seedt.tm_year}.{seedt.tm_mon}.{seedt.tm_mday}_{seedt.tm_hour}{seedt.tm_min}.npy"
    np.save(allcond_file_name,all_cond_data)

if __name__=='__main__':
    NUM_SAMPLE= 16
    N_WELL=1
    N_TASKS = 256

    PORO_VAR=0.004
    PERM_VAR=2500
    LEN_SCALE=[3200, 3200, 80]

    x = np.arange(1900, 8200, 200)#32 6400
    y = np.arange(1100, 7400, 200)#32 6400
    z = np.arange(-2975, -2820, 10)#16 160
    # X,Y,Z = np.meshgrid(x, y, z)

    STEPS=[0,1,3,9,16,22,28,34,40,50]
    T_STOP=25
    INJ_RATE=15
    GetAllTask(x,y,z,N_TASKS, PORO_VAR, PERM_VAR, LEN_SCALE, 10)













