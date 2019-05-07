from data_loader import DataLoader
from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
from unet import Unet
from resnetlike128_pad import ResnetLike128
import os
from tifffile import imsave

#import visdom     #python -m visdom.server
#viz=visdom.Visdom()





iterace=5000
init_lr = 1e-3
batch = 32 
path_to_data_train='../data_patch_15'
path_to_data_test='../data_patch_test67'
n_critic=5
lam=10
lam2=100#jak moc cycle


fol='../l2_100'
try:
    os.mkdir(fol)
except:
    print('folder exits')



def grad_pen(batch,lam,true_im,fake_im,D):
    eps = torch.rand(batch, 1).cuda()
    eps = eps.expand(batch, int(true_im.nelement()/batch)).contiguous().view(s[0],s[1],s[2],s[3])
    interpolates = eps * true_im+ ((1 - eps) * fake_im)
    interpolates=interpolates.detach()
    interpolates.requires_grad=True
    out_interpolates=D(interpolates)
    gradients = autograd.grad(outputs=out_interpolates, inputs=interpolates,grad_outputs=torch.ones(out_interpolates.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)*lam
    
    return gradient_penalty.mean()
    


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']




if __name__ == '__main__':
    
    
    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
    
    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
    
    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
    
    
    
    
    unet_qpi2dapi=Unet(feature_scale=8).cuda()
    unet_dapi2qpi=Unet(feature_scale=8).cuda()
    
    
    D_dapi=ResnetLike128(K=16).cuda()
    D_qpi=ResnetLike128(K=16).cuda()
    
    
    optimizer_unet_qpi2dapi = optim.Adam(unet_qpi2dapi.parameters(),lr = init_lr ,betas=(0.5, 0.9))
    optimizer_unet_dapi2qpi = optim.Adam(unet_dapi2qpi.parameters(),lr = init_lr ,betas=(0.5, 0.9))
    
    optimizer_D_dapi = optim.Adam(D_dapi.parameters(),lr = init_lr ,betas=(0.5, 0.9))
    optimizer_D_qpi = optim.Adam(D_qpi.parameters(),lr = init_lr ,betas=(0.5, 0.9))

    
    scheduler_unet_qpi2dapi=optim.lr_scheduler.StepLR(optimizer_unet_qpi2dapi, 2000, gamma=0.1, last_epoch=-1)
    scheduler_unet_dapi2qpi=optim.lr_scheduler.StepLR(optimizer_unet_dapi2qpi, 2000, gamma=0.1, last_epoch=-1)
    scheduler_D_dapi=optim.lr_scheduler.StepLR(optimizer_D_dapi, 2000, gamma=0.1, last_epoch=-1)
    scheduler_D_qpi=optim.lr_scheduler.StepLR(optimizer_D_qpi, 2000, gamma=0.1, last_epoch=-1)
    
    def l1_loss(input, target):
        return torch.mean(torch.abs(input - target))
    def l2_loss(input, target):
            return torch.mean((input - target)**2)
    
    
    def inf_train_gen():
        while True:
            for qpi,dapi,name_qpi,name_dapi in trainloader:
                # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
                yield qpi,dapi,name_qpi,name_dapi
                
    gen = inf_train_gen()
    
    
#    trainloader=iter(trainloader)


    valid_l2_qpi=[0]
    valid_l2_dapi=[0]
    valid_iters=[0]
    
    best_model_score=999999
    best_model_qpi2dapi=0
    best_model_dapi2qpi=0
    stop=0
    itt=0
    while itt<iterace and stop==0:
        
        itt=itt+1
        scheduler_unet_qpi2dapi=optim.lr_scheduler.StepLR(optimizer_unet_qpi2dapi, 2000, gamma=0.1, last_epoch=-1)
        scheduler_unet_dapi2qpi=optim.lr_scheduler.StepLR(optimizer_unet_dapi2qpi, 2000, gamma=0.1, last_epoch=-1)
        scheduler_D_dapi=optim.lr_scheduler.StepLR(optimizer_D_dapi, 2000, gamma=0.1, last_epoch=-1)
        scheduler_D_qpi=optim.lr_scheduler.StepLR(optimizer_D_qpi, 2000, gamma=0.1, last_epoch=-1)
        
        
        
        unet_qpi2dapi.train()
        D_dapi.train()
        unet_dapi2qpi.train()
        D_qpi.train()
        
        for p in unet_qpi2dapi.parameters():  
            p.requires_grad = False
        for p in unet_dapi2qpi.parameters():  
            p.requires_grad = False
        for p in D_dapi.parameters():  
            p.requires_grad = True
        for p in D_qpi.parameters():  
            p.requires_grad = True
    
        for t in range(n_critic):
            
            (qpi,dapi,name_qpi,name_dapi)=next(gen)
        
            qpi = qpi.cuda(0)
            dapi = dapi.cuda(0)
            
            
            

            qpi.requires_grad=False
            optimizer_D_dapi.zero_grad() 
            
            fake_dapi=unet_qpi2dapi(qpi)
            fake_dapi=fake_dapi.detach()
            fake_dapi.requires_grad=True
            
            s=dapi.size()
            dapi.requires_grad=True
    
            out_dapi_real=-D_dapi(dapi).mean()
            out_dapi_real.backward()
            out_dapi_fake=D_dapi(fake_dapi).mean()
            out_dapi_fake.backward()
            
            dapi = dapi.detach()
            fake_dapi = fake_dapi.detach()
            dapi.requires_grad=True
            fake_dapi.requires_grad=True
            
            gradient_penalty=grad_pen(batch,lam,dapi,fake_dapi,D_dapi)
            gradient_penalty.backward()
            
            loss_D_dapi=out_dapi_fake+out_dapi_real+gradient_penalty
            w_loss_D_dapi=out_dapi_fake+out_dapi_real
            
            optimizer_D_dapi.step()
            
            
            
            

            dapi=dapi.detach()
            dapi.requires_grad=False
            qpi=qpi.detach()
            qpi.requires_grad=True
            optimizer_D_qpi.zero_grad() 
            
            fake_qpi=unet_dapi2qpi(dapi)
            fake_qpi=fake_qpi.detach()
            fake_qpi.requires_grad=True
            
            s=qpi.size()
    
            out_qpi_real=-D_qpi(qpi).mean()
            out_qpi_real.backward()
            out_qpi_fake=D_qpi(fake_qpi).mean()
            out_qpi_fake.backward()
            
            qpi = qpi.detach()
            fake_qpi = fake_qpi.detach()
            qpi.requires_grad=True
            fake_qpi.requires_grad=True
            
            gradient_penalty=grad_pen(batch,lam,qpi,fake_qpi,D_qpi)
            gradient_penalty.backward()
            
            loss_D_qpi=out_qpi_fake+out_qpi_real+gradient_penalty
            w_loss_D_qpi=out_qpi_fake+out_qpi_real
            
            optimizer_D_qpi.step()
            
            
            
            
            
            
            
        for p in D_dapi.parameters(): 
            p.requires_grad = False
        for p in D_qpi.parameters(): 
            p.requires_grad = False
            
        for p in unet_qpi2dapi.parameters():  
            p.requires_grad = True
        for p in unet_dapi2qpi.parameters():  
            p.requires_grad = True    
            
            
        optimizer_unet_qpi2dapi.zero_grad() 
        optimizer_unet_dapi2qpi.zero_grad()     
        
        (qpi,dapi,name_qpi,name_dapi)=next(gen)
        

        qpi = qpi.cuda(0)
        dapi = dapi.cuda(0)
        
        qpi.requires_grad=True
        dapi.requires_grad=True
        
        
        fake_dapi=unet_qpi2dapi(qpi)
        fake_fake_qpi=unet_dapi2qpi(fake_dapi)
        loss_G_dapi = -D_dapi(fake_dapi).mean()
        loss_cycle_qpi=l1_loss(fake_fake_qpi,qpi)
        
        fake_qpi=unet_dapi2qpi(dapi)
        fake_fake_dapi=unet_qpi2dapi(fake_qpi)
        loss_G_qpi = -D_qpi(fake_qpi).mean()
        loss_cycle_dapi=l1_loss(fake_fake_dapi,dapi)
        
        loss=loss_G_dapi+loss_G_qpi+lam2*loss_cycle_dapi+lam2*loss_cycle_qpi
#            
        loss.backward()
        optimizer_unet_qpi2dapi.step()
        optimizer_unet_dapi2qpi.step()


        
        if itt%5==0:
            from_dapi_example=np.concatenate((dapi[0,0,:,:].data.cpu().numpy(),fake_qpi[0,0,:,:].data.cpu().numpy(),fake_fake_dapi[0,0,:,:].data.cpu().numpy()),axis=1)
            from_qpi_example=np.concatenate((qpi[0,0,:,:].data.cpu().numpy(),fake_dapi[0,0,:,:].data.cpu().numpy(),fake_fake_qpi[0,0,:,:].data.cpu().numpy()),axis=1)
            print(str(itt) +'_'+ str(get_lr(optimizer_unet_qpi2dapi)) +'_valid_loss' + str(valid_l2_dapi[-1]))
            plt.plot(valid_iters,valid_l2_qpi)
            plt.show()
            plt.plot(valid_iters,valid_l2_dapi)
            plt.show()
            plt.imshow(from_dapi_example,vmin=-0.5, vmax=0.5)
#            plt.savefig(fol+'/result1'+ str(itt).zfill(7) +'.png', format='png', dpi=200,bbox_inches='tight')
            plt.show()
            plt.imshow(from_qpi_example,vmin=-0.5, vmax=0.5)
#            plt.savefig(fol+'/result2'+ str(itt).zfill(7) +'.png', format='png', dpi=200,bbox_inches='tight')
            plt.show()
            
          
            
        if itt%200==0:  
            valid_l2_qpi_tmp=[]
            valid_l2_dapi_tmp=[]
            for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(validloader):
                unet_qpi2dapi.eval()
                unet_dapi2qpi.eval()
                qpi = qpi.cuda(0)
                dapi = dapi.cuda(0)
                
                
                fake_dapi=unet_qpi2dapi(qpi)
                l2_dapi=l2_loss(fake_dapi,dapi)
                
                
                fake_qpi=unet_dapi2qpi(dapi)
                l2_qpi=l2_loss(fake_qpi,qpi)
                
                
                valid_l2_qpi_tmp.append(l2_qpi.detach().cpu().numpy())
                valid_l2_dapi_tmp.append(l2_dapi.detach().cpu().numpy())
                
                
                if it%5==0:
                    print('test' + str(it))
                    example=np.concatenate((qpi[0,0,:,:].data.cpu().numpy(),fake_qpi[0,0,:,:].data.cpu().numpy()),axis=1)
                    plt.imshow(example,vmin=-0.5, vmax=0.5)
                    plt.savefig(fol+'/validresult'+ str(itt).zfill(7) + str(it) +'_qpi.png', format='png', dpi=200,bbox_inches='tight')
                    plt.show()  
                    example=np.concatenate((dapi[0,0,:,:].data.cpu().numpy(),fake_dapi[0,0,:,:].data.cpu().numpy()),axis=1)
                    plt.imshow(example,vmin=-0.5, vmax=0.5)
                    plt.savefig(fol+'/validresult'+ str(itt).zfill(7) + str(it) +'_dapi.png', format='png', dpi=200,bbox_inches='tight')
                    plt.show()
                    
                    
            tmp=np.mean(valid_l2_qpi_tmp)
            valid_l2_qpi.append(tmp)
            tmp=np.mean(valid_l2_dapi_tmp)
            valid_l2_dapi.append(tmp)
            valid_iters.append(itt)
            
            
            unet_qpi2dapi_name=fol +'/unet_qpi2dapi_'+ str(itt) +'_'+ str(get_lr(optimizer_unet_qpi2dapi)) +'_valid_loss' + str(valid_l2_dapi[-1]) +'.pt'
            torch.save(unet_qpi2dapi,unet_qpi2dapi_name)
            unet_dapi2qpi_name=fol +'/unet_dapi2qpi_'+ str(itt) +'_'+ str(get_lr(optimizer_unet_qpi2dapi)) +'_valid_loss' + str(valid_l2_dapi[-1]) +'.pt'
            torch.save(unet_dapi2qpi,unet_dapi2qpi_name)
            if best_model_score>tmp:
                best_model_score=tmp
                best_model_qpi2dapi=unet_qpi2dapi_name
                best_model_dapi2qpi=unet_dapi2qpi_name
    
            
    
    unet_qpi2dapi=torch.load(best_model_qpi2dapi)
    unet_dapi2qpi=torch.load(best_model_dapi2qpi)
    for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(testloader):
        unet_qpi2dapi.eval()
        unet_dapi2qpi.eval()
        qpi = qpi.cuda(0)
        dapi = dapi.cuda(0)
        
        
        fake_dapi=unet_qpi2dapi(qpi)
        
        
        fake_qpi=unet_dapi2qpi(dapi)
        
        fake_qpi=fake_qpi[0,0,:,:].data.cpu().numpy()
        fake_qpi=(fake_qpi+0.5)
        name=name_qpi[0]
        tmp=name.split('\\')
        imsave(fol + '/' + tmp[-1],fake_qpi)
            
        fake_dapi=fake_dapi[0,0,:,:].data.cpu().numpy()
        fake_dapi=(fake_dapi+0.5)
        name=name_dapi[0]
        tmp=name.split('\\')
        imsave(fol + '/' + tmp[-1],fake_dapi)
        

    
    
    
    
    