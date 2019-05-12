clc;clear all;close all
kolik=20000;
odkud_kam=[10 300];
sizes=[128 128];


results_path='../l2_5_old';
% results_path='../data_patch_test67';

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

names_qpi=names_qpi(6:7);
names_dapi=names_dapi(6:7);


vahokno=2*ones(128);
vahokno=conv2(vahokno,ones(2*28+1(sum(ones(2*28+1))),'same');
vahokno=vahokno-1;
vahokno(vahokno<0.01)=0.01;

img_num=0;
for k=1:length(names_qpi)
    
    name_qpi=names_qpi{k};
    name_dapi=names_dapi{k};
    
    for kk=odkud_kam(1):odkud_kam(2)
        
           
        poskladany_qpi=zeros(600);
        poskladany_dapi=zeros(600);
        podelit=zeros(600);
        
        
        img_num=img_num+1;
        
        patch_num=0;
        
        pos_start=1:100:550;
        pos_start(2:end)=pos_start(2:end)-28;
%         pos_end=pos_start+127;
        
        for x=pos_start
            xx=x+127;
             for y=pos_start
                yy=y+127;
                patch_num=patch_num+1;
                
%                 qpi = single(imread(names_qpi{k},kk,'PixelRegion',{[x xx],[y yy]}));
%     
%                 dapi = single(imread(names_dapi{k},kk,'PixelRegion',{[x xx],[y yy]}));
                
                qpi=imread([results_path '/qpi_' num2str(img_num,'%07.f') '_' num2str(patch_num,'%03.f') '.tif']);
                dapi=imread([results_path '/dapi_' num2str(img_num,'%07.f') '_' num2str(patch_num,'%03.f') '.tif']);
                
                
                poskladany_qpi(x:xx,y:yy)=poskladany_qpi(x:xx,y:yy)+qpi.*vahokno;
                poskladany_dapi(x:xx,y:yy)=poskladany_dapi(x:xx,y:yy)+dapi.*vahokno;
                podelit(x:xx,y:yy)=podelit(x:xx,y:yy)+vahokno;
                
                

%                 qpi=mat2gray(qpi,[-0.1 2.4]);
%                 dapi=mat2gray(dapi,[0 1200]);
%                 qpi=single(qpi);
%                 dapi=single(dapi);
%                 
                
                
                
             end
        
        end
        
        poskladany_qpi=poskladany_qpi./podelit;
        poskladany_dapi=poskladany_dapi./podelit;
        
        
        name_qpi_tmp=split(name_qpi,'\');
        name_qpi_tmp=name_qpi_tmp{end};
        
        name_dapi_tmp=split(name_dapi,'\');
        name_dapi_tmp=name_dapi_tmp{end};
        
        imwrite(poskladany_qpi,[results_path '/fullqpires_' num2str(kk,'%03.f') '_' name_qpi_tmp])
        imwrite(poskladany_dapi,[results_path '/fulldapires_' num2str(kk,'%03.f')  '_'  name_dapi_tmp])
        
        
%         imwrite(poskladany_qpi,['../gt' '/fullqpires_' num2str(kk,'%03.f') '_' name_qpi_tmp])
%         imwrite(poskladany_dapi,['../gt' '/fulldapires_' num2str(kk,'%03.f')  '_'  name_dapi_tmp])
        
%         imshow(poskladany_dapi,[])
        
    end
    
end
