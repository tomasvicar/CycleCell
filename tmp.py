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



iterace=14000
init_lr = 1e-2
batch = 8
n_critic=5
lam=10
alpha=0
beta=1
save_dir='D:\jakubicek\spectral_CT_results\en_pure_unet'

try:
    os.mkdir(save_dir)
except:
    pass


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    
#    for start in range(0,13,2):
    for start in range(1):
        
#        test_ids=np.arange((start),(start+2))
#        valid_ids=np.arange((start+2),(start+4))
#        train_ids=np.arange((start+4),(start+14))
#        test_ids=test_ids%14+1
#        valid_ids=valid_ids%14+1
#        train_ids=train_ids%14+1
        
        train_ids=np.arange(1,9)
        valid_ids=np.arange(9,11)
        test_ids=np.arange(11,15)
        
        print(train_ids)
        print(valid_ids)
        print(test_ids)
        
        
        loader = DataLoader(split='train',path_to_data='D:\jakubicek\spectral_CT_data',ids=train_ids)
        trainloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=True,drop_last=True,pin_memory=False)
        
        loader = DataLoader(split='valid',path_to_data='D:\jakubicek\spectral_CT_data',ids=valid_ids)
        validloader= data.DataLoader(loader, batch_size=batch, num_workers=4, shuffle=False,drop_last=False,pin_memory=False)
        
        loader = DataLoader(split='test',path_to_data='D:\jakubicek\spectral_CT_data',ids=test_ids)
        testloader= data.DataLoader(loader, batch_size=1, num_workers=0, shuffle=False,drop_last=False,pin_memory=False)
        
        
        unet=Unet(filters = [8, 16, 32, 32]).cuda()
        
        D=ResnetLike128(K=16).cuda()
        
        
        optimizer_unet = optim.Adam(unet.parameters(),lr = init_lr , betas=(0.5, 0.9))
        
        optimizer_D = optim.Adam(D.parameters(),lr = init_lr ,betas=(0.5, 0.9))
        
        scheduler1=optim.lr_scheduler.StepLR(optimizer_unet, 6000, gamma=0.1, last_epoch=-1)
        scheduler2=optim.lr_scheduler.StepLR(optimizer_D, 6000, gamma=0.1, last_epoch=-1)
        
        
        #    l2_loss = nn.MSELoss()
        #    l1_loss = nn.L1Loss()
        def l1_loss(input, target):
            return torch.mean(torch.abs(input - target))
        
        def l2_loss(input, target):
            return torch.mean((input - target)**2) 
        
        def inf_train_gen():
            while True:
                for in_images,out_images,pat in trainloader:
                    # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
                    yield in_images,out_images,pat
                    
        gen = inf_train_gen()
        
        
        
        
        best_model=0
        best_model_score=100000
        train_loss=[]
        test_loss=[0.1]
        train_iters=[]
        test_iters=[0]
        stop=0
        itt=0
        start_sl=0
        
        
        stop=0
        itt=0
        while itt<iterace and stop==0:
            
            itt=itt+1
            
            scheduler1.step()
            scheduler2.step()
            
            unet.train()
            D.train()
            
            for p in unet.parameters():  
                p.requires_grad = False
                
            for p in D.parameters():  
                p.requires_grad = True
                
            if alpha>0:
                for t in range(n_critic):
                    
                    (in_images,out_images,pat)=next(gen)
                
                    in_images = in_images.cuda(0)
                    out_images = out_images.cuda(0)
                    
                    in_images.requires_grad=False
                    
                    
                    optimizer_D.zero_grad() 
                    
                    fake_out_images=unet(in_images)
                    fake_out_images=fake_out_images.detach()
                    fake_out_images=torch.cat((fake_out_images,in_images),dim=1)
                    fake_out_images.requires_grad=True
                    
                    
                    
                    
                    out_images=torch.cat((out_images,in_images),dim=1)
                    out_images.requires_grad=True
                    
                    s=out_images.size()
                    
            
                    out_real=-D(out_images).mean()
                    out_real.backward()
                    out_fake=D(fake_out_images).mean()
                    out_fake.backward()
                    
                    
                    out_images = out_images.detach()
                    fake_out_images = fake_out_images.detach()
                    out_images.requires_grad=True
                    fake_out_images.requires_grad=True
                    
                    
                    eps = torch.rand(batch, 1).cuda()
                    eps = eps.expand(batch, int(out_images.nelement()/batch)).contiguous().view(s[0],s[1],s[2],s[3])
                    interpolates = eps * out_images + ((1 - eps) * fake_out_images)
                    interpolates=interpolates.detach()
                    interpolates.requires_grad=True
                    out_interpolates=D(interpolates)
                    gradients = autograd.grad(outputs=out_interpolates, inputs=interpolates,grad_outputs=torch.ones(out_interpolates.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)*lam
                    gradient_penalty=gradient_penalty.mean()
                    gradient_penalty.backward()
                    
                    
                    loss=out_fake+out_real+gradient_penalty
                    w_loss=out_fake+out_real
                    
                    
                    
                    
                    optimizer_D.step()
                
            for p in D.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in unet.parameters():  
                p.requires_grad = True
                
            optimizer_unet.zero_grad() 
                
            
            (in_images,out_images,pat)=next(gen)
            
        
            in_images = in_images.cuda(0)
            out_images = out_images.cuda(0)
            
            in_images.requires_grad=True
            out_images.requires_grad=True
            
            
            fake_out_images=unet(in_images)
            
            l2_l=l2_loss(fake_out_images,out_images)
            
            fake_out_images=torch.cat((fake_out_images,in_images),dim=1)
            if alpha>0:
                loss_G = -D(fake_out_images).mean()
            else:
                loss_G=0
            
           
            
            train_loss.append(l2_l.detach().cpu().numpy())
            train_iters.append(itt)
            
            loss_G=alpha*loss_G+beta*l2_l
        #            
            loss_G.backward()
            optimizer_unet.step()
            
            
            
            if itt%20==0:
                print(str(itt) +'_'+ str(get_lr(optimizer_unet)) +'_train_loss:' +str(train_loss[-1])+'_test_loss:' + str(test_loss[-1]))
                example=np.concatenate((in_images[0,0,:,:].data.cpu().numpy(),fake_out_images[0,0,:,:].data.cpu().numpy(),out_images[0,0,:,:].data.cpu().numpy()),axis=1)
                plt.imshow(example,vmin=-0.5, vmax=0)
                plt.show()
                example=np.abs(fake_out_images[0,0,:,:].data.cpu().numpy()-out_images[0,0,:,:].data.cpu().numpy())
                plt.imshow(example,vmin=0, vmax=0.5)
                plt.show()
                
                print(str(itt) +'_'+ str(get_lr(optimizer_unet)) +'_train_loss:' +str(train_loss[-1])+'_test_loss:' + str(test_loss[-1]))
                example=np.concatenate((in_images[0,0,:,:].data.cpu().numpy(),fake_out_images[0,1,:,:].data.cpu().numpy(),out_images[0,1,:,:].data.cpu().numpy()),axis=1)
                plt.imshow(example,vmin=-0.5, vmax=0)
                plt.show()
                example=np.abs(fake_out_images[0,1,:,:].data.cpu().numpy()-out_images[0,1,:,:].data.cpu().numpy())
                plt.imshow(example,vmin=0, vmax=0.5)
                plt.show()
                
                print(str(itt) +'_'+ str(get_lr(optimizer_unet)) +'_train_loss:' +str(train_loss[-1])+'_test_loss:' + str(test_loss[-1]))
                example=np.concatenate((in_images[0,0,:,:].data.cpu().numpy(),fake_out_images[0,2,:,:].data.cpu().numpy(),out_images[0,2,:,:].data.cpu().numpy()),axis=1)
                plt.imshow(example,vmin=-0.5, vmax=0)
                plt.show()
                example=np.abs(fake_out_images[0,2,:,:].data.cpu().numpy()-out_images[0,2,:,:].data.cpu().numpy())
                plt.imshow(example,vmin=0, vmax=0.5)
                plt.show()
                
                
                plt.plot(train_iters,train_loss)
                plt.plot(test_iters,test_loss)
                plt.ylim([0,0.001])
                plt.show()
                
                
                
        
            if itt%200==0:  
                test_loss_tmp=[]
                for it,(in_images,out_images,pat) in enumerate(validloader):
                    unet.eval()
                    in_images = in_images.cuda(0)
                    out_images = out_images.cuda(0)
                    
                    
                    fake_out_images=unet(in_images)
                    
                    l2_l=l2_loss(fake_out_images,out_images)
                    
                    test_loss_tmp.append(l2_l.detach().cpu().numpy())
                    
                    if it%20==0:
                        print('test' + str(it))
                        example=np.concatenate((in_images[0,0,:,:].data.cpu().numpy(),fake_out_images[0,0,:,:].data.cpu().numpy(),out_images[0,0,:,:].data.cpu().numpy()),axis=1)
                        plt.imshow(example,vmin=-0.5, vmax=0)
                        plt.show()
                        example=np.abs(fake_out_images[0,0,:,:].data.cpu().numpy()-out_images[0,0,:,:].data.cpu().numpy())
                        plt.imshow(example,vmin=0, vmax=0.5)
                        plt.show()
                        
                tmp=np.mean(test_loss_tmp)
                test_loss.append(tmp)
                test_iters.append(itt)
                
                unet_name=save_dir +'/unet_'+ str(itt) +'_'+ str(get_lr(optimizer_unet)) +'_train_loss' +str(train_loss[-1])+'_test_loss' + str(test_loss[-1]) +'.pt'
                torch.save(unet,unet_name)
                if best_model_score>tmp:
                    best_model_score=tmp
                    best_model=unet_name
        
        
        unet=torch.load(best_model)
        for it,(in_images,out_images,pat) in enumerate(testloader):
            unet.eval()
            in_images = Variable(in_images.cuda(0))
            out_images = Variable(out_images.cuda(0))
            
            fake_out=unet(in_images)
            for k in range(fake_out.size()[1]):
                img=fake_out[0,k,:,:].data.cpu().numpy()
                img=(img+0.5)*(2**12)
                name=pat[k][0]
                name=name.replace('D:\jakubicek\spectral_CT_data',save_dir)
                imsave(name,img)
    
    