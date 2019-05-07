import numpy as np
from torch.utils import data
import torch
import os
from scipy import misc
import matplotlib.pyplot as plt




class DataLoader(data.Dataset):
    def __init__(self, split="train",path_to_data='',paired=False):
        self.split=split
        self.paired=paired
        self.path_to_data=path_to_data
        
        
        
        self.file_names=[]
        for root, dirs, files in os.walk(self.path_to_data):
            for name in files:
                if name.endswith(".tif") and name.startswith("qpi"):
                    self.file_names.append(root+'\\'+name)
                    
        st0 = np.random.get_state()
        np.random.seed(0)
        
        
        if split!="test":
            r=np.random.choice(len(self.file_names),size=19000,replace=False)
            rr=np.arange(len(self.file_names))
            rr=np.delete(rr,r)
            
            
            np.random.set_state(st0)
        
        
        if split=="train":
            self.file_names=[self.file_names[i] for i in r]
            
        if split=="valid":
            self.file_names=[self.file_names[i] for i in rr]
        
        self.num_of_img=len(self.file_names)
        
        print(self.num_of_img)
        
    def __len__(self):
        return self.num_of_img
    
    
    
    def __getitem__(self, index):
        if self.paired:
            index_dapi=index
        else:
            index_dapi=np.random.randint(0,self.num_of_img)
        index_qpi=index
        
        name_qpi=self.file_names[index_qpi]
        name_dapi=self.file_names[index_dapi]
        name_dapi=name_dapi.replace('qpi','dapi')
        
        qpi=misc.imread(name_qpi)-0.5
        dapi=misc.imread(name_dapi)-0.5
        
        qpi=np.expand_dims(qpi,2)
        dapi=np.expand_dims(dapi,2)
        
        qpi=np.transpose(qpi,(2, 0, 1))
        qpi=torch.from_numpy(qpi)
        
        dapi=np.transpose(dapi,(2, 0, 1))
        dapi=torch.from_numpy(dapi)
        
        return qpi,dapi,name_qpi,name_dapi
        
        
        