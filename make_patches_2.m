clc;clear all;close all
kolik=20000;
odkud_kam=[10 300];
sizes=[128 128];

slozky={'../data_orig/DU145_st'};
names_qpi={};
for k = slozky
    listing=subdir([k{1} '/Compensated phase-pgp*.tiff']);
    names_qpi=[names_qpi {listing(:).name}];
end
names_dapi={};
for name=names_qpi
    names_dapi=[names_dapi strrep(name{1},'Compensated phase-pgpum2','Clipped-DAPI')];
end

names_qpi=names_qpi(1:5);
names_dapi=names_dapi(1:5);

img_size=[600 600];
for k=1:kolik
    k
    snimek_num=randi(length(names_qpi));
    time=randi(odkud_kam);
    pos_x=randi(img_size(1)-sizes(1)+1);
    pos_y=randi(img_size(2)-sizes(2)+1);
    rotation=randi(4);
    flip1=rand>0.5;
    flip2=rand>0.5;
    mult1=0.8+0.4*rand;
    mult2=0.8+0.4*rand;
    
    qpi = single(imread(names_qpi{snimek_num},time,'PixelRegion',{[pos_x pos_x+sizes(1)-1],[pos_y pos_y+sizes(2)-1]}));
    
    dapi = single(imread(names_dapi{snimek_num},time,'PixelRegion',{[pos_x pos_x+sizes(1)-1],[pos_y pos_y+sizes(2)-1]}));
    
    qpi=mat2gray(qpi*mult1,[-0.1 2.4]);
    dapi=mat2gray(dapi*mult2,[0 1200]);
    qpi=single(qpi);
    dapi=single(dapi);
    
    if flip1
         dapi=flipud(dapi);
         qpi=flipud(qpi);
    end
    if flip2
        dapi=fliplr(dapi);
        qpi=fliplr(qpi);
    end
    dapi=rot90(dapi,rotation);
    qpi=rot90(qpi,rotation);
    
%     imshow(qpi)
%     figure()
%     imshow(dapi)
    
%     disp(size(qpi))
    
    imwrite_single(qpi,['../data_patch_15/qpi_' num2str(k,'%07.f') '.tif'])
    imwrite_single(dapi,['../data_patch_15/dapi_' num2str(k,'%07.f') '.tif'])
    
end

% aaaa=imread(['../data_patch/qpi_' num2str(k-1,'%07.f') '.tif']);


