import os
import numpy as np
from utils.dataprocessor import getTrainData
from torch.utils.data import DataLoader, SequentialSampler
from utils.bertWhiteness import BertWhitening,sents2vecs


def get_kernel_bias(config,tokenizer,model,test_dataloader,datastore_path):
        #compute/load kernel, bias
    if not os.path.exists(datastore_path):
        os.mkdir(datastore_path)
    kernel_path=datastore_path+'/kernel.npy'
    bias_path=datastore_path+'/bais.npy'
    
    if not os.path.exists(kernel_path) or os.path.exists(bias_path):
        train_data=getTrainData(tokenizer,config['model']['model_name'],config['data']['data_path'])
        train_sampler = SequentialSampler(train_data) 
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['knnTest']['train_datastore_size'])
        
        train_vecs,_=sents2vecs(train_dataloader,model)
        test_vecs,_=sents2vecs(test_dataloader,model)
        kernel, bias=BertWhitening([train_vecs,test_vecs])
        np.save(kernel_path,kernel,allow_pickle=True, fix_imports=True)
        np.save(bias_path,bias,allow_pickle=True, fix_imports=True)
    else:
        kernel=np.load(kernel_path, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
        bias=np.load(bias_path, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    
    return kernel,bias