function [] = ModelGenerator(n_samples_each_task,maindir,subdir,if_delete_pic)
fid=fopen("matlab_log.txt",'a');
for j = 1 : length( subdir )
    if( isequal( subdir( j ).name, '.' )||...
        isequal( subdir( j ).name, '..')||...
        ~subdir( j ).isdir)               % 如果不是目录则跳过
        continue;
    end
    task_no = strsplit(subdir( j ).name,'task');
    task_num = task_no(2);
    savedir_pic=strcat("img\tasks\task",task_num);
    mkdir(savedir_pic);
    subdirpath = fullfile( maindir, subdir( j ).name);
    for i = 0:(n_samples_each_task-1)
        sample_num=i;        
        pro_data_path = fullfile( subdirpath, strcat('property',num2str(sample_num),'.csv'));
        well_data_path = fullfile( subdirpath, strcat('well',num2str(sample_num),'.csv'));
        well_loc=csvread( well_data_path);
        x_well=num2str(well_loc(1),4);
        y_well=num2str(well_loc(2),4);
        x_well=strcat(x_well,'[m]');
        y_well=strcat(y_well,'[m]');
        perm_pic_from = fullfile( subdirpath, strcat('perm',num2str(sample_num),'.png'));
        poro_pic_from = fullfile( subdirpath, strcat('poro',num2str(sample_num),'.png'));
        poro_pic_to = fullfile( savedir_pic, strcat('poro',num2str(sample_num),'.png'));
        perm_pic_to = fullfile( savedir_pic, strcat('perm',num2str(sample_num),'.png'));
        try
            BuildSim(pro_data_path,x_well,y_well,task_num,sample_num);
        catch
            log=strcat(subdir(j).name,"_s_",num2str(sample_num));
            fprintf(fid,log);
            disp(log)
            continue;
        end
        copyfile(poro_pic_from,poro_pic_to);
        copyfile(perm_pic_from,perm_pic_to);
        if if_delete_pic
            delete(poro_pic_from);
            delete(perm_pic_from);
        end
    end
end
 