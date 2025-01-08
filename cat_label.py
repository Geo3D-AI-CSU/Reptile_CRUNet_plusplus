
import numpy as np
import matplotlib.pyplot as plt
import re
import copy
from move import *

class CatLabels():
    def __init__(self):
        self.from_path='./'
        self.to_path='new312'
        self.save_path= 'dataSet'
        self.x = np.arange(1900, 8200, 200)#32 6400
        self.y = np.arange(1100, 7400, 200)#32 6400
        self.z = np.arange(-2975, -2820, 10)#16 160
        self.cols=np.linspace(3, 21, 10, dtype=np.int16)
    def __init__(self,from_path, to_path, save_path, x, y, z, cols):
        self.from_path=from_path
        self.to_path=to_path
        self.save_path= save_path
        self.x = x
        self.y = y
        self.z = z
        self.cols=cols
        if not os.path.exists(to_path):
            os.mkdir(to_path)
    def Display(self,X,Y,Z,poro,perm,well_loc,label,save_path):
        fig = plt.figure()
        ax1 = fig.add_subplot(221, projection='3d')
        scatter=ax1.scatter(X,Y,Z, c=poro, s=6, marker='.', cmap='jet')
        ax1.view_init(24,-140)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('poro')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        plt.colorbar(scatter)
        # 绘制插值结果

        ax2 = fig.add_subplot(222, projection='3d')
        # 选择一个特定;;的 Z 切片进行绘制
        scatter=ax2.scatter(X,Y,Z,c=perm, s=6, marker='.', cmap='jet')
        ax2.view_init(24,-140)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('perm')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        plt.colorbar(scatter)
        
        ax1 = fig.add_subplot(223, projection='3d')
        scatter=ax1.scatter(X,Y,Z, c=well_loc, s=6, marker='.', cmap='viridis')
        ax1.view_init(24,-140)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('well')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        plt.colorbar(scatter)

        # 绘制插值结果
        ax2 = fig.add_subplot(224, projection='3d')
        # 选择一个特定;;的 Z 切片进行绘制
        scatter=ax2.scatter(X,Y,Z,c=label[0,9,:], s=6, marker='.', cmap='YlOrRd')
        ax2.view_init(24,-140)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('lable')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        plt.colorbar(scatter)

        # fig.show()
        fig.savefig(save_path,dpi=300)
        plt.close()
    def GetLabelGrid(self,label, t_val_column):
        label_list=[]
        for col in t_val_column:
            label_ti=label[:,col]
            label_ti=np.reshape(label_ti,(16,32,32))#zyx
            label_ti=label_ti.transpose(2,1,0)#xyz
            label_list.append(label_ti)
        np_label=np.array(label_list)
        return np_label
    def NormalizeData(self,data):
        num_c = data.shape[1]
        for i in range(num_c):
            data[:,i,:] = (data[:,i,:]-np.min(data[:,i,:]))/(np.max(data[:,i,:])-np.min(data[:,i,:]))
            # print(f"{np.min(data[:,i,:])}   {np.max(data[:,i,:])} ")
        return data
    def DealSingleTask(self,task_i=0):
        all_labels=[]
        all_datas=[]
        task_folder=f"{self.save_path}/task{task_i}"
        pic_folder=f"{self.save_path}/img/tasks_catted"
        if not os.path.exists(task_folder):
            os.mkdir(task_folder)
        if not os.path.exists(pic_folder):
            os.mkdir(pic_folder)
            os.mkdir(f"{pic_folder}/task{task_i}")
        samples_ls=os.listdir(f"{self.to_path}/datas/task{task_i}")
        for sample in samples_ls:
            _, file_extension = os.path.splitext(sample)
            if not (file_extension == ".npy"):
                continue
            sample_i = re.findall(r'\d+', sample)[0]
            sample_i_data=np.load(f"{self.to_path}/datas/task{task_i}/property{sample_i}.npy")
            sample_i_label=np.loadtxt(f"{self.to_path}/labels/task{task_i}/label_s{sample_i}.csv", delimiter=',',skiprows=9)
            cols=copy.deepcopy(self.cols)
            labelC_i=self.GetLabelGrid(sample_i_label, cols)
            labelT_i=self.GetLabelGrid(sample_i_label, cols+1)
            label_i=np.concatenate([labelC_i[np.newaxis,:] , 
                                    labelT_i[np.newaxis,:] ],
                                    axis=0)
            X,Y,Z = np.meshgrid(self.x, self.y, self.z)
            poro_i=sample_i_data[0, 0, :, :, :]
            perm_i=sample_i_data[0, 1, :, :, :]
            well_loc_i=sample_i_data[0, 2, :, :, :]
            self.Display(X,Y,Z,poro_i,perm_i,well_loc_i,label_i,f"{pic_folder}/task{task_i}/sample{sample_i}.png")

            # shutil.copy(f"{all_data_path}/datas/task{task_i}/property{sample_i}.npy",f"{task_folder}/x_{sample_i}.npy")
            # np.save(f"{task_folder}/y_{sample_i}.npy", label_i)
            sample_i_data=self.NormalizeData(sample_i_data)
            all_labels.append(label_i)
            all_datas.append(sample_i_data)

        data=np.array(all_datas)
        label=np.array(all_labels)
        np.save(f"{task_folder}/x.npy", data)
        np.save(f"{task_folder}/y.npy", label)
    def DealAllTask(self, delete_org_file=False, delete_selected_file=False):
        move_file = MoveFiles(self.from_path, self.to_path,delete_org_file)
        move_file.MoveData()
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        all_tasks = os.listdir(f"{self.to_path}/labels")
        for task in all_tasks:
            task_i= re.findall(r'\d+', task)[0]
            print(f"task{task_i}")
            self.DealSingleTask(task_i)
        if delete_selected_file:
            shutil.rmtree(self.to_path)
        

# x = np.arange(1900, 8200, 200)#32 6400
# y = np.arange(1100, 7400, 200)#32 6400
# z = np.arange(-2975, -2820, 10)#16 160
# cols=np.linspace(3, 21, 10, dtype=np.int16)

# catlabels= CatLabels("./", "new312","./dataSet",x,y,z,cols)
# catlabels.DealAllTask()



# move_file = MoveFiles("./", "new312")
# move_file.MoveData()
# DealAllTask(x,y,z,cols,all_data_path,save_path)


# def GetAreaIdx(data, x, y, z):
#     steps = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
#     idex=[]
#     for i in  range(data.shape[0]):
#         xi, yi, zi = (data[i][0], data[i][1], data[i][2])
#         if ( xi < x[0] or xi > x[-1]+steps[0] or 
#              yi < y[0] or yi > y[-1]+steps[1] or 
#              zi < z[0] or zi > z[-1]+steps[2]):
#             continue
#         xn =int((xi-x[0])//steps[0])
#         yn =int((yi-y[0])//steps[1])
#         zn =int((zi-z[0])//steps[2])
#         idex.append([yn, xn, zn])#不知道为什么，这里必须XY反过来
#     df = pd.DataFrame(idex, columns=['x', 'y', 'z'])
#     df.drop_duplicates(inplace=True)
#     indexes=df.to_numpy()
#     return indexes

# def FiltData(data, x, y, z):
#     steps = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
#     idx_isnan = ~np.isnan(data).any(axis = 1)
#     data=data[idx_isnan, :]
#     data=data[np.where(data[:,0]>=x[0])]
#     data=data[np.where(data[:,0]< (x[-1]+200))]
#     data=data[np.where(data[:,1]< (y[-1]+200))]
#     data=data[np.where(data[:,1]>= (y[0]))]
#     return data

# def GetValGrd(cond_pos, cond_val, x, y, z, idex):
#     xg, yg ,zg = np.meshgrid(x, y, z)
#     data = griddata(cond_pos, cond_val, (xg, yg ,zg ), method='nearest')
#     data=data[idex[:,0],idex[:,1],idex[:,2]]
#     return data.flatten()
