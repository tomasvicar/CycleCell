from data_loader import DataLoader
from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from unet import Unet
from resnetlike128 import ResnetLike128

import visdom     #python -m visdom.server
viz=visdom.Visdom()



iterace=9999999
init_lr = 0.0001
batch = 16 
path_to_data='../data_patch1'
lam=10



if __name__ == '__main__':
    
    
    loader = DataLoader(split='train',path_to_data=path_to_data,paired=False)
    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
    
    loader = DataLoader(split='test',path_to_data=path_to_data,paired=True)
    testloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=False,drop_last=False,pin_memory=False)
    
    
    unet_qpi2dapi=Unet(feature_scale=8).cuda()
    unet_dapi2qpi=Unet(feature_scale=8).cuda()
    
    
    D_dapi=ResnetLike128(K=16).cuda()
    D_qpi=ResnetLike128(K=16).cuda()
    
    
    optimizer_unet_qpi2dapi = optim.Adam(unet_qpi2dapi.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    optimizer_unet_dapi2qpi = optim.Adam(unet_dapi2qpi.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    
    optimizer_D_dapi = optim.Adam(D_dapi.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    optimizer_D_qpi = optim.Adam(D_qpi.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    
    
    l2_loss = nn.MSELoss()
#    l1_loss = nn.L1Loss()
    def l1_loss(input, target):
        return torch.mean(torch.abs(input - target)) / input.data.nelement()
    
    
    stop=0
    itt=0
    while itt<iterace and stop==0:
        for it,(qpi,dapi) in enumerate(trainloader):
            itt=itt+1
            
            unet_qpi2dapi.train()
            unet_dapi2qpi.train()
            D_dapi.train()
            D_qpi.train()
            
            
        
    
            qpi = Variable(qpi.cuda(0))
            dapi = Variable(dapi.cuda(0))
            
            
            fake_dapi=unet_qpi2dapi(qpi)
            fake_qpi=unet_dapi2qpi(dapi)
            
            
            fake_fake_dapi=unet_qpi2dapi(fake_qpi)
            fake_fake_qpi=unet_dapi2qpi(fake_dapi)
            
            
            qpi_fake_qpi=torch.cat((qpi,fake_qpi),dim=0)
            dapi_fake_dapi=torch.cat((dapi,fake_dapi),dim=0)
            gt_qpi=Variable(torch.cat((torch.ones(batch,dtype=torch.float32),torch.zeros(batch,dtype=torch.float32)),dim=0).cuda(0))
            gt_dapi=Variable(torch.cat((torch.ones(batch,dtype=torch.float32),torch.zeros(batch,dtype=torch.float32)),dim=0).cuda(0))
            
            
            
            c_qpi=F.sigmoid(D_qpi(qpi_fake_qpi))
            c_dapi=F.sigmoid(D_dapi(dapi_fake_dapi))
            c_qpi=torch.squeeze(c_qpi)
            c_dapi=torch.squeeze(c_dapi)
            

            ones=Variable(torch.ones(batch,dtype=torch.float32).cuda(0))


            loss_G=l2_loss(c_dapi[batch:],ones)+l2_loss(c_qpi[batch:],ones)+lam*l1_loss(qpi,fake_fake_qpi)+lam*l1_loss(dapi,fake_fake_dapi)
            
            loss_D=l2_loss(c_qpi,gt_qpi)+l2_loss(c_dapi,gt_dapi)
            
            
#            
            for param in D_dapi.parameters():
                param.requires_grad = False
            for param in D_qpi.parameters():
                param.requires_grad = False
            optimizer_unet_qpi2dapi.zero_grad() 
            optimizer_unet_dapi2qpi.zero_grad() 
            loss_G.backward(retain_graph=True)
            optimizer_unet_qpi2dapi.step()
            optimizer_unet_dapi2qpi.step()
    
    
    
            for param in D_dapi.parameters():
                param.requires_grad = True
            for param in D_qpi.parameters():
                param.requires_grad = True
            optimizer_D_dapi.zero_grad() 
            optimizer_D_qpi.zero_grad() 
            loss_D.backward()
            optimizer_D_dapi.step()
            optimizer_D_qpi.step()
            
            from_dapi_example=np.concatenate((dapi[0,0,:,:].data.cpu().numpy(),fake_qpi[0,0,:,:].data.cpu().numpy(),fake_fake_dapi[0,0,:,:].data.cpu().numpy()),axis=1)
            from_qpi_example=np.concatenate((qpi[0,0,:,:].data.cpu().numpy(),fake_dapi[0,0,:,:].data.cpu().numpy(),fake_fake_qpi[0,0,:,:].data.cpu().numpy()),axis=1)
            print(itt)
            plt.imshow(from_dapi_example,vmin=-0.5, vmax=0.5)
            plt.show()
            plt.imshow(from_qpi_example,vmin=-0.5, vmax=0.5)
            plt.show()
            

    
    
    
    
    