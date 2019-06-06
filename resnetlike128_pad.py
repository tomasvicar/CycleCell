import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Conv2BnRelu(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=1):
        super().__init__()
    
        self.conv=nn.Conv2d(in_size, out_size,filter_size,stride,pad)
#        self.bn=nn.BatchNor2m2d(out_size,momentum=0.1)
#        self.bn=nn.InstanceNorm2d(out_size)

#        dov=0.1
#        self.do=nn.Sequential(nn.Dropout(dov),nn.Dropout2d(dov))

    def forward(self, inputs):
        outputs = self.conv(inputs)
#        outputs = self.bn(outputs)          
        outputs=F.relu(outputs)
#        outputs = self.do(outputs)

        return outputs
    
    
class ResnetLike128(nn.Module):
    def __init__(self,K=16):
        super().__init__()
        
        
        
        self.initc=Conv2BnRelu(1, K)##128
        
        self.c1=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))
        
        self.c2=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))

        self.p1=nn.MaxPool2d(2, stride=2)#64
        
        
        K=K*2
        
        self.c11=Conv2BnRelu(K//2, K,filter_size=1)
        
        self.c3=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))
        
        self.c4=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))
        
        self.p2=nn.MaxPool2d(2, stride=2)##32
        
        
        K=K*2
        
        self.c22=Conv2BnRelu(K//2, K,filter_size=1)
        
        self.c5=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))
        
        self.c6=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))
        
        self.p3=nn.MaxPool2d(2, stride=2)##8
        
        
        K=K*2
        
        self.c33=Conv2BnRelu(K//2, K,filter_size=1)
        
        self.c7=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))#
        
        self.c8=nn.Sequential(Conv2BnRelu(K, K))#
        
        self.p4=nn.MaxPool2d(2, stride=2)##4
        
        
        K=K*2
        
        self.c44=Conv2BnRelu(K//2, K,filter_size=1)
        
        self.c9=nn.Sequential(Conv2BnRelu(K, K),Conv2BnRelu(K, K))#
        
        self.c10=nn.Sequential(Conv2BnRelu(K, K))#
        
        self.p5=torch.nn.AdaptiveAvgPool2d(1)##1
        
        
        
        


        self.finalc=nn.Conv2d(K, 1,1)
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
        
    

    
    
    def sum_in(self,input1,input2):
        
        return input1 + input2
    
    
    def forward(self, x):
        
        
        x=self.initc(x)
        
        y=self.c1(x)
        x=self.sum_in(x,y)
        
        y=self.c2(x)
        x=self.sum_in(x,y)
        
        x=self.p1(x)
        
        
        x=self.c11(x)
        
        y=self.c3(x)
        x=self.sum_in(x,y)
        
        y=self.c4(x)
        x=self.sum_in(x,y)
        
        x=self.p2(x)
        
        x=self.c22(x)
        
        
        y=self.c5(x)
        x=self.sum_in(x,y)
        
        y=self.c6(x)
        x=self.sum_in(x,y)
        
        x=self.p3(x)       
        
        x=self.c33(x)

        y=self.c7(x)
        x=self.sum_in(x,y)
        
        y=self.c8(x)
        x=self.sum_in(x,y)
        
        x=self.p4(x) 
        
        
        
        x=self.c44(x)

        y=self.c9(x)
        x=self.sum_in(x,y)
        
        y=self.c10(x)
        x=self.sum_in(x,y)
        
        x=self.p5(x) 
        
        
        
        
        x=self.finalc(x)
        
        return x
        
