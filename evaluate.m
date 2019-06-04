clc;clear all;close all

dapi_names=subdir('../gt_5/fulldapires*');
dapi_names={dapi_names(:).name};

qpi_names=cellfun(@(x) strrep(strrep(x,'fulldapires','fullqpires'),'Clipped-DAPI','Compensated phase-pgpum2'),dapi_names,'UniformOutput',false);


% results_path='../l2_5_old';

% results_paths={'../l2_0','../l2_0_last',...
%     '../l2_u','../l2_u_last',...
%     '../l22_u','../l22_u_last',...
%     '../l2_01','../l2_01_last',...
%     '../l2_1','../l2_1_last',...
%     '../l2_10','../l2_10_last',...
%     '../l2_100','../l2_100_last',...
%     '../l22_01','../l22_01_alltrain','../l22_01_last','../l22_01_alltrain_last',...
%     '../l22_1','../l22_1_alltrain','../l22_1_last','../l22_1_alltrain_last',...
%     '../l22_10','../l22_10_last','../l22_10_last',...
%     '../l22_100','../l22_100_alltrain','../l22_100_last','../l22_100_alltrain_last'...
%     };

% results_paths={'../l2_100_last'};

% results_paths={'../l2_0_last' };

% results_paths={'../l2_0'};

results_paths={'../l22_u_last'};

% results_paths={'../l22_1_alltrain_last'};

% results_paths={'../l22_1_last'};

% results_paths={'../l22_10_last'};


res_dapi=[];
res_qpi=[];


for results_path=results_paths
    results_path=results_path{1};





mses_dapi=[];
mses_qpi=[];


for k=1:length(dapi_names)
    

    
    disp([num2str(k) '/' num2str(length(dapi_names)) ])
    dapi_name=dapi_names{k};
    qpi_name=qpi_names{k};
    
    
    
    dapi_name_res=strrep(dapi_name,'..\gt_5',results_path);
    qpi_name_res=strrep(qpi_name,'..\gt_5',results_path);
    
    
    dapi=imread(dapi_name);
    qpi=imread(qpi_name);
    
    
    
    dapi_res=imread(dapi_name_res);
    qpi_res=imread(qpi_name_res);
    
    
    dapi_res=imread(dapi_name_res);
    qpi_res=imread(qpi_name_res);
    
    dapi_name_res_fake=replace(dapi_name_res,'fulldapires','fulldapiresfake');
    qpi_name_res_fake=replace(qpi_name_res,'fullqpires','fullqpiresfake');
    
    dapi_fake=imread(dapi_name_res_fake);
    qpi_fake=imread(qpi_name_res_fake);
    
    
    
    imshow(cat(1,cat(2,dapi,dapi_res),cat(2,qpi,qpi_res)),[0,1]);

    drawnow;
    

%     if k==5
% %         pos=[145 129];
%         pos=[176 338];
%         
%         n=results_path(4:end);
%         
%         imwrite(uint8(dapi(pos(1):pos(1)+127,pos(2):pos(2)+127)*255*1.2),['../vysledkove_obr/gt_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8(qpi(pos(1):pos(1)+127,pos(2):pos(2)+127)*255),['../vysledkove_obr/gt_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         imwrite(uint8(dapi_res(pos(1):pos(1)+127,pos(2):pos(2)+127)*255*1.2),['../vysledkove_obr/res_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8(qpi_res(pos(1):pos(1)+127,pos(2):pos(2)+127)*255),['../vysledkove_obr/res_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         
%         imwrite(uint8(dapi_fake(pos(1):pos(1)+127,pos(2):pos(2)+127)*255*1.2),['../vysledkove_obr/res_fake_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8(qpi_fake(pos(1):pos(1)+127,pos(2):pos(2)+127)*255),['../vysledkove_obr/res_fake_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         
%         drawnow;
%         dfsdf=fdafds
%     end
    








    if k==5
%         pos=[145 129];
%         pos=[64 135];%cele1
        pos=[147 279];
        
        n=results_path(4:end);
        
%         w_s=300;

        w_s=100;
        
        minus=0.90;
        krat=9;
        
        tmp=(dapi(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255;
        imwrite(uint8(cat(3,zeros(size(tmp)),zeros(size(tmp)),tmp)),['../vysledkove_obr/gt_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8((dapi(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255),['../vysledkove_obr/gt_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        imwrite(uint8(qpi(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*255),['../vysledkove_obr/gt_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        
        tmp=(dapi_res(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255;
        imwrite(uint8(cat(3,zeros(size(tmp)),zeros(size(tmp)),tmp)),['../vysledkove_obr/res_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8((dapi_res(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255),['../vysledkove_obr/res_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        imwrite(uint8(qpi_res(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*255),['../vysledkove_obr/res_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        
        tmp=(dapi_fake(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255;
        imwrite(uint8(cat(3,zeros(size(tmp)),zeros(size(tmp)),tmp)),['../vysledkove_obr/res_fake_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8((dapi_fake(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255),['../vysledkove_obr/res_fake_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        imwrite(uint8(qpi_fake(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*255),['../vysledkove_obr/res_fake_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        
        
        tmp=cat(3,qpi,qpi,qpi)*0.9;
        qqq=(dapi_res*9-0.9);
        qqq(qqq<0)=0;
        tmp(:,:,3)=tmp(:,:,3)+qqq;
        
        imwrite(uint8(tmp(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s,:)*255),['../vysledkove_obr/color_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        
        
        tmp=cat(3,qpi,qpi,qpi)*0.9;
        qqq=(dapi*9-0.9);
        qqq(qqq<0)=0;
        tmp(:,:,3)=tmp(:,:,3)+qqq;
        
        imwrite(uint8(tmp(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s,:)*255),['../vysledkove_obr/color_gt_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
        
        
        
        
        drawnow;
        dfsdf=fdafds
    end

    
    


%     if k==5
%         pos=[145 129];
        
%    if k==31
%         pos=[176 338];
        
%    if k==25
%         pos=[365 282];
% %         
%         n=results_path(4:end);
%         
%         w_s=127;
%         
%         minus=0.50;
%         krat=6;
%         
%         tmp=(dapi(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255;
%         imwrite(uint8(cat(3,zeros(size(tmp)),zeros(size(tmp)),tmp)),['../vysledkove_obr/gt_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
% %         imwrite(uint8((dapi(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255),['../vysledkove_obr/gt_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8(qpi(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*255),['../vysledkove_obr/gt_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         tmp=(dapi_res(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255;
%         imwrite(uint8(cat(3,zeros(size(tmp)),zeros(size(tmp)),tmp)),['../vysledkove_obr/res_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
% %         imwrite(uint8((dapi_res(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255),['../vysledkove_obr/res_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8(qpi_res(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*255),['../vysledkove_obr/res_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         tmp=(dapi_fake(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255;
%         imwrite(uint8(cat(3,zeros(size(tmp)),zeros(size(tmp)),tmp)),['../vysledkove_obr/res_fake_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
% %         imwrite(uint8((dapi_fake(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*krat-minus)*255),['../vysledkove_obr/res_fake_dapi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         imwrite(uint8(qpi_fake(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s)*255),['../vysledkove_obr/res_fake_qpi_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         
%         tmp=cat(3,qpi,qpi,qpi);
%         tmp(:,:,3)=tmp(:,:,3)+(dapi_res*krat-minus);
%         
%         imwrite(uint8(tmp(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s,:)*255),['../vysledkove_obr/color_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         
%         tmp=cat(3,qpi,qpi,qpi);
%         tmp(:,:,3)=tmp(:,:,3)+(dapi*3-0.3);
%         
%         imwrite(uint8(tmp(pos(1):pos(1)+w_s,pos(2):pos(2)+w_s,:)*255),['../vysledkove_obr/color_gt_' num2str(pos(1)) '_' num2str(pos(2)) '_'  num2str(k) '_' n '.png'])
%         
%         
%         
%         
%         drawnow;
%         dfsdf=fdafds
%     end






    
    mses_dapi=[mses_dapi mean(mean((dapi-dapi_res).^2))];
    mses_qpi=[mses_qpi mean(mean((qpi-qpi_res).^2))];
    
end

mean(mses_dapi)
mean(mses_qpi)

res_dapi=[res_dapi mean(mses_dapi)];
res_qpi=[res_qpi mean(mses_qpi)];


end