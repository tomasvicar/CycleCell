clc;clear all;close all

dapi_names=subdir('../gt/fulldapires*');
dapi_names={dapi_names(:).name};

qpi_names=cellfun(@(x) strrep(strrep(x,'fulldapires','fullqpires'),'Clipped-DAPI','Compensated phase-pgpum2'),dapi_names,'UniformOutput',false);


results_path='../l2_5_old';

mses_dapi=[];
mses_qpi=[];

for k=1:length(dapi_names)
    disp([num2str(k) '/' num2str(length(dapi_names)) ])
    dapi_name=dapi_names{k};
    qpi_name=qpi_names{k};
    
    
    dapi_name_res=strrep(dapi_name,'..\gt',results_path);
    qpi_name_res=strrep(qpi_name,'..\gt',results_path);
    
    
    dapi=imread(dapi_name);
    qpi=imread(qpi_name);
    
    dapi_res=imread(dapi_name_res);
    qpi_res=imread(qpi_name_res);
    
    
    mses_dapi=[mses_dapi mean(mean((dapi-dapi_res).^2))];
    mses_qpi=[mses_qpi mean(mean((qpi-qpi_res).^2))];
    
end

mean(mses_dapi)
mean(mses_qpi)