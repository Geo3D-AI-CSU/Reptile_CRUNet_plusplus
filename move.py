import os
import shutil



class MoveFiles():
    def __init__(self,from_path, to_path, delete_org_file=False):
        self.from_path=from_path
        self.to_path=to_path
        self.delete_org=delete_org_file
        if not os.path.exists(to_path):
            os.mkdir(to_path)

    def CreateFolder(self,filepath):
        if not os.path.exists(filepath):
                os.mkdir(filepath)

    def MoveData(self):
        labe_set=self.from_path + "/labels"
        data_set=self.from_path + "/tasks"
        ls_label_file = os.listdir(labe_set)
        for i_label_file in ls_label_file:
            print(i_label_file)
            idx = i_label_file.split('task')[-1]
            from_label_task = labe_set +"/"+i_label_file
            from_data_task = data_set+"/"+"task"+idx
            to_label_task =self.to_path +"/labels"
            self.CreateFolder(to_label_task)
            to_data_task = self.to_path +"/datas"
            self.CreateFolder(to_data_task)
            to_label_task =self.to_path +"/labels/task"+idx
            to_data_task = self.to_path +"/datas/task"+idx
            shutil.copytree(from_label_task,to_label_task)
            shutil.copytree(from_data_task,to_data_task)
            if self.delete_org: 
                 shutil.rmtree(from_label_task)
                 shutil.rmtree(from_data_task)



