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
#
#import visdom     #python -m visdom.server
#viz=visdom.Visdom()



iterace=9999999
init_lr = 0.0001
batch = 16 
path_to_data='../data_patch1'
lam=10



if __name__ == '__main__':
    
    
    loader = DataLoader(split='train',path_to_data=path_to_data,paired=True)
    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
    
    loader = DataLoader(split='test',path_to_data=path_to_data,paired=True)
    testloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=False,drop_last=False,pin_memory=False)
    
    
    unet_qpi2dapi=Unet(feature_scale=8).cuda()

    
    
    optimizer_unet_qpi2dapi = optim.Adam(unet_qpi2dapi.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)

    
    
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
            
            
        
    
            qpi = Variable(qpi.cuda(0))
            dapi = Variable(dapi.cuda(0))
            
            
            fake_dapi=unet_qpi2dapi(qpi)
            
            loss=l2_loss(dapi,fake_dapi)
            
#           
                
            optimizer_unet_qpi2dapi.zero_grad() 
            loss.backward()
            optimizer_unet_qpi2dapi.step()
            
            if itt%10==0:
                from_dapi_example=np.concatenate((dapi[0,0,:,:].data.cpu().numpy(),fake_dapi[0,0,:,:].data.cpu().numpy(),qpi[0,0,:,:].data.cpu().numpy()),axis=1)
                print(itt)
                plt.imshow(from_dapi_example,vmin=-0.5, vmax=0.5)
                plt.show()

            

    
    
    
    
    