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

#
#path_to_data_train='../data_patch_15'
#path_to_data_test='../data_patch_test67'
#batch = 32 
#
#
#fol='../l2_1_last'
#
#best_model_qpi2dapi='../l2_1/unet_qpi2dapi_5000_0.001_valid_loss0.0009746385.pt'
#best_model_dapi2qpi='../l2_1/unet_dapi2qpi_5000_0.001_valid_loss0.0009746385.pt'
#
#try:
#    os.mkdir(fol)
#except:
#    print('folder exits')
#    
#    
#    
#
#
#if __name__ == '__main__':
#    
#    
#    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
#    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
#    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
#    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
#    
#    
#
#    unet_qpi2dapi=torch.load(best_model_qpi2dapi)
#    unet_dapi2qpi=torch.load(best_model_dapi2qpi)
#    for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(testloader):
#        unet_qpi2dapi.eval()
#        unet_dapi2qpi.eval()
#        qpi = qpi.cuda(0)
#        dapi = dapi.cuda(0)
#        
#        
#        fake_dapi=unet_qpi2dapi(qpi)
#        
#        
#        fake_qpi=unet_dapi2qpi(dapi)
#        
#        fake_qpi=fake_qpi[0,0,:,:].data.cpu().numpy()
#        fake_qpi=(fake_qpi+0.5)
#        name=name_qpi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_qpi)
#            
#        fake_dapi=fake_dapi[0,0,:,:].data.cpu().numpy()
#        fake_dapi=(fake_dapi+0.5)
#        name=name_dapi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_dapi)
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#path_to_data_train='../data_patch_15'
#path_to_data_test='../data_patch_test67'
#batch = 32 
#
#
#fol='../l2_5_last'
#
#best_model_qpi2dapi='../l2_5/unet_qpi2dapi_5000_0.001_valid_loss0.0024112582.pt'
#best_model_dapi2qpi='../l2_5/unet_dapi2qpi_5000_0.001_valid_loss0.0024112582.pt'
#
#try:
#    os.mkdir(fol)
#except:
#    print('folder exits')
#    
#    
#    
#
#
#if __name__ == '__main__':
#    
#    
#    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
#    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
#    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
#    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
#    
#    
#
#    unet_qpi2dapi=torch.load(best_model_qpi2dapi)
#    unet_dapi2qpi=torch.load(best_model_dapi2qpi)
#    for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(testloader):
#        unet_qpi2dapi.eval()
#        unet_dapi2qpi.eval()
#        qpi = qpi.cuda(0)
#        dapi = dapi.cuda(0)
#        
#        
#        fake_dapi=unet_qpi2dapi(qpi)
#        
#        
#        fake_qpi=unet_dapi2qpi(dapi)
#        
#        fake_qpi=fake_qpi[0,0,:,:].data.cpu().numpy()
#        fake_qpi=(fake_qpi+0.5)
#        name=name_qpi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_qpi)
#            
#        fake_dapi=fake_dapi[0,0,:,:].data.cpu().numpy()
#        fake_dapi=(fake_dapi+0.5)
#        name=name_dapi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_dapi)
#    
#
#
#
#
#
#
#
#
#
#path_to_data_train='../data_patch_15'
#path_to_data_test='../data_patch_test67'
#batch = 32 
#
#
#fol='../l2_01_last'
#
#best_model_qpi2dapi='../l2_01/unet_qpi2dapi_5000_0.001_valid_loss0.0022650734.pt'
#best_model_dapi2qpi='../l2_01/unet_dapi2qpi_5000_0.001_valid_loss0.0022650734.pt'
#
#try:
#    os.mkdir(fol)
#except:
#    print('folder exits')
#    
#    
#    
#
#
#if __name__ == '__main__':
#    
#    
#    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
#    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
#    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
#    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
#    
#    
#
#    unet_qpi2dapi=torch.load(best_model_qpi2dapi)
#    unet_dapi2qpi=torch.load(best_model_dapi2qpi)
#    for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(testloader):
#        unet_qpi2dapi.eval()
#        unet_dapi2qpi.eval()
#        qpi = qpi.cuda(0)
#        dapi = dapi.cuda(0)
#        
#        
#        fake_dapi=unet_qpi2dapi(qpi)
#        
#        
#        fake_qpi=unet_dapi2qpi(dapi)
#        
#        fake_qpi=fake_qpi[0,0,:,:].data.cpu().numpy()
#        fake_qpi=(fake_qpi+0.5)
#        name=name_qpi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_qpi)
#            
#        fake_dapi=fake_dapi[0,0,:,:].data.cpu().numpy()
#        fake_dapi=(fake_dapi+0.5)
#        name=name_dapi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_dapi)
#
#
#
#
#
#
#
#path_to_data_train='../data_patch_15'
#path_to_data_test='../data_patch_test67'
#batch = 32 
#
#
#fol='../l2_10_last'
#
#best_model_qpi2dapi='../l2_10/unet_qpi2dapi_5000_0.001_valid_loss0.0014496235.pt'
#best_model_dapi2qpi='../l2_10/unet_dapi2qpi_5000_0.001_valid_loss0.0014496235.pt'
#
#try:
#    os.mkdir(fol)
#except:
#    print('folder exits')
#    
#    
#    
#
#
#if __name__ == '__main__':
#    
#    
#    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
#    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
#    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
#    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
#    
#    
#
#    unet_qpi2dapi=torch.load(best_model_qpi2dapi)
#    unet_dapi2qpi=torch.load(best_model_dapi2qpi)
#    for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(testloader):
#        unet_qpi2dapi.eval()
#        unet_dapi2qpi.eval()
#        qpi = qpi.cuda(0)
#        dapi = dapi.cuda(0)
#        
#        
#        fake_dapi=unet_qpi2dapi(qpi)
#        
#        
#        fake_qpi=unet_dapi2qpi(dapi)
#        
#        fake_qpi=fake_qpi[0,0,:,:].data.cpu().numpy()
#        fake_qpi=(fake_qpi+0.5)
#        name=name_qpi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_qpi)
#            
#        fake_dapi=fake_dapi[0,0,:,:].data.cpu().numpy()
#        fake_dapi=(fake_dapi+0.5)
#        name=name_dapi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_dapi)
#        
#        
#        
#        
#
#
#path_to_data_train='../data_patch_15'
#path_to_data_test='../data_patch_test67'
#batch = 32 
#
#
#fol='../l2_50_last'
#
#best_model_qpi2dapi='../l2_50/unet_qpi2dapi_5000_0.001_valid_loss0.0007543122.pt'
#best_model_dapi2qpi='../l2_50/unet_dapi2qpi_5000_0.001_valid_loss0.0007543122.pt'
#
#try:
#    os.mkdir(fol)
#except:
#    print('folder exits')
#    
#    
#    
#
#
#if __name__ == '__main__':
#    
#    
#    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
#    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
#    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
#    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
#    
#    
#
#    unet_qpi2dapi=torch.load(best_model_qpi2dapi)
#    unet_dapi2qpi=torch.load(best_model_dapi2qpi)
#    for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(testloader):
#        unet_qpi2dapi.eval()
#        unet_dapi2qpi.eval()
#        qpi = qpi.cuda(0)
#        dapi = dapi.cuda(0)
#        
#        
#        fake_dapi=unet_qpi2dapi(qpi)
#        
#        
#        fake_qpi=unet_dapi2qpi(dapi)
#        
#        fake_qpi=fake_qpi[0,0,:,:].data.cpu().numpy()
#        fake_qpi=(fake_qpi+0.5)
#        name=name_qpi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_qpi)
#            
#        fake_dapi=fake_dapi[0,0,:,:].data.cpu().numpy()
#        fake_dapi=(fake_dapi+0.5)
#        name=name_dapi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_dapi)
#        
#        
#        
#        
#        
#        
#        
#
#path_to_data_train='../data_patch_15'
#path_to_data_test='../data_patch_test67'
#batch = 32 
#
#
#fol='../l2_100_last'
#
#best_model_qpi2dapi='../l2_100/unet_qpi2dapi_5000_0.001_valid_loss0.00081495807.pt'
#best_model_dapi2qpi='../l2_100/unet_dapi2qpi_5000_0.001_valid_loss0.00081495807.pt'
#
#try:
#    os.mkdir(fol)
#except:
#    print('folder exits')
#    
#    
#    
#
#
#if __name__ == '__main__':
#    
#    
#    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
#    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
#    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
#    
#    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
#    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
#    
#    
#
#    unet_qpi2dapi=torch.load(best_model_qpi2dapi)
#    unet_dapi2qpi=torch.load(best_model_dapi2qpi)
#    for it,(qpi,dapi,name_qpi,name_dapi) in enumerate(testloader):
#        unet_qpi2dapi.eval()
#        unet_dapi2qpi.eval()
#        qpi = qpi.cuda(0)
#        dapi = dapi.cuda(0)
#        
#        
#        fake_dapi=unet_qpi2dapi(qpi)
#        
#        
#        fake_qpi=unet_dapi2qpi(dapi)
#        
#        fake_qpi=fake_qpi[0,0,:,:].data.cpu().numpy()
#        fake_qpi=(fake_qpi+0.5)
#        name=name_qpi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_qpi)
#            
#        fake_dapi=fake_dapi[0,0,:,:].data.cpu().numpy()
#        fake_dapi=(fake_dapi+0.5)
#        name=name_dapi[0]
#        tmp=name.split('\\')
#        imsave(fol + '/' + tmp[-1],fake_dapi)









path_to_data_train='../data_patch_15'
path_to_data_test='../data_patch_test67'
batch = 32 


fol='../l2_0_last'

best_model_qpi2dapi='../l2_0/unet_qpi2dapi_5000_0.001_valid_loss0.0015981033.pt'
best_model_dapi2qpi='../l2_0/unet_dapi2qpi_5000_0.001_valid_loss0.0015981033.pt'

try:
    os.mkdir(fol)
except:
    print('folder exits')
    
    
    


if __name__ == '__main__':
    
    
    loader = DataLoader(split='train',path_to_data=path_to_data_train,paired=False)
    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
    
    loader = DataLoader(split='valid',path_to_data=path_to_data_train,paired=True)
    validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
    
    loader = DataLoader(split='test',path_to_data=path_to_data_test,paired=True)
    testloader= data.DataLoader(loader, batch_size=1, num_workers=1, shuffle=False,drop_last=False,pin_memory=False)
    
    

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











