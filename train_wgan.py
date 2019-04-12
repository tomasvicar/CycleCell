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

#import visdom     #python -m visdom.server
#viz=visdom.Visdom()



iterace=9999999
init_lr = 1e-4
batch = 32 
path_to_data='../data_patch1'
n_critic=5
lam=10




if __name__ == '__main__':
    
    
    loader = DataLoader(split='train',path_to_data=path_to_data,paired=False)
    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
    
    loader = DataLoader(split='test',path_to_data=path_to_data,paired=True)
    testloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=False,drop_last=False,pin_memory=False)
    
    
    unet_qpi2dapi=Unet(feature_scale=8).cuda()
    
    D_dapi=ResnetLike128(K=16).cuda()
    
    
    optimizer_unet_qpi2dapi = optim.Adam(unet_qpi2dapi.parameters(),lr = init_lr , betas=(0.5, 0.9))
    
    optimizer_D_dapi = optim.Adam(D_dapi.parameters(),lr = init_lr ,betas=(0.5, 0.9))

    
    
    l2_loss = nn.MSELoss()
#    l1_loss = nn.L1Loss()
    def l1_loss(input, target):
        return torch.mean(torch.abs(input - target)) / input.data.nelement()
    
    
    def inf_train_gen():
        while True:
            for images, target in trainloader:
                # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
                yield images, target
                
    gen = inf_train_gen()
    
    
#    trainloader=iter(trainloader)


    
    stop=0
    itt=0
    while itt<iterace and stop==0:
        
        itt=itt+1
        
        
        
        unet_qpi2dapi.train()
        D_dapi.train()
        
        for p in unet_qpi2dapi.parameters():  
            p.requires_grad = False
    
        for t in range(n_critic):
            
            (qpi,dapi)=next(gen)
        
            qpi = qpi.cuda(0)
            dapi = dapi.cuda(0)
            
            qpi.requires_grad=False
            
            
            optimizer_D_dapi.zero_grad() 
            for p in D_dapi.parameters():  
                p.requires_grad = True
            
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
            
            
            eps = torch.rand(batch, 1).cuda()
            eps = eps.expand(batch, int(dapi.nelement()/batch)).contiguous().view(s[0],s[1],s[2],s[3])
            interpolates = eps * dapi + ((1 - eps) * fake_dapi)
            interpolates=interpolates.detach()
            interpolates.requires_grad=True
            out_interpolates=D_dapi(interpolates)
            gradients = autograd.grad(outputs=out_interpolates, inputs=interpolates,grad_outputs=torch.ones(out_interpolates.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)*lam
            gradient_penalty=gradient_penalty.mean()
            gradient_penalty.backward()
            
            
            loss=out_dapi_fake+out_dapi_real+gradient_penalty
            w_loss=out_dapi_fake+out_dapi_real
            
            
            
            optimizer_D_dapi.step()
            
        for p in D_dapi.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in unet_qpi2dapi.parameters():  
            p.requires_grad = True
            
        optimizer_unet_qpi2dapi.zero_grad() 
            
        
        (qpi,dapi)=next(gen)
        

        qpi = qpi.cuda(0)
        dapi = dapi.cuda(0)
        
        qpi.requires_grad=True
        dapi.requires_grad=True
        
        
        fake_dapi=unet_qpi2dapi(qpi)
        
        loss_G = -D_dapi(fake_dapi).mean()
        
#            
        loss_G.backward()
        optimizer_unet_qpi2dapi.step()



        
        if itt%5==0:
            from_dapi_example=np.concatenate((dapi[0,0,:,:].data.cpu().numpy(),fake_dapi[0,0,:,:].data.cpu().numpy(),qpi[0,0,:,:].data.cpu().numpy()),axis=1)
            print(itt)
            plt.imshow(from_dapi_example,vmin=-0.5, vmax=0.5)
            plt.savefig('../tmp/result'+ str(itt).zfill(7) +'.png', format='png', dpi=200,bbox_inches='tight')
            plt.show()
        

    
    
    
    
    