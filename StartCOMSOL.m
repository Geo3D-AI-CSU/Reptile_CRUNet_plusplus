addpath 'D:\comsol\COMSOL62\Multiphysics\mli'
% cd('F:\dlz\matlab2')
cd('E:\matlab2')
mphstart(2036)
import com.comsol.model.util.*
ModelUtil.showProgress(true)
n_samples_each_task=16;
pathdir = 'E:\matlab2';
maindir = strcat(pathdir, '\tasks\');
savedir=strcat(pathdir,'\labels');
savedir_img=strcat(pathdir,'\img\tasks');
mkdir(savedir);
mkdir(savedir_img);
subdir  = dir( maindir );
ModelGenerator(n_samples_each_task,maindir,subdir,false)