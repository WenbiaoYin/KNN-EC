import os
import json
import torch
import pickle
import pandas as pd
from torch.utils.data import TensorDataset

emotionNum2ekmanNum=json.load(open('/path_to_store_your/emotionNum2ekmanNum.json', 'r'))

def getData(tokenizer,file_name):
    
    data=pd.read_csv(file_name,sep='\t',header=None)
    
    data=data[data[1]!='27'] #Remove neutral labels
    data=data[[len(label.split(','))==1 for label in data[1].tolist()]] #Remove mutil labels
    
    sents=[tokenizer(sent.lower(),padding='max_length',truncation=True,max_length=128) for sent in data[0].tolist()]
    sents_input_ids=torch.tensor([temp["input_ids"] for temp in sents])
    sents_attn_masks=torch.tensor([temp["attention_mask"] for temp in sents])
    
    labels=torch.tensor([emotionNum2ekmanNum[label] for label in data[1].tolist()])
    
    dataset=TensorDataset(sents_input_ids,sents_attn_masks,labels)
    
    return dataset

def getTrainData(tokenizer,bert_name,data_path):
    if not os.path.exists(data_path+ "/%s"%(bert_name.split('/')[-1])):
        os.makedirs( data_path+"/%s"%(bert_name.split('/')[-1]))
    
    feature_file = data_path+"/%s/train_features.pkl"%(bert_name.split('/')[-1])
    if os.path.exists(feature_file):
        train_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        train_dataset = getData(tokenizer,data_path+'/train.tsv')
        with open(feature_file, 'wb') as w:
            pickle.dump(train_dataset, w)
    return train_dataset

def getDevData(tokenizer,bert_name,data_path):
    feature_file = data_path+"/%s/dev_features.pkl"%(bert_name.split('/')[-1])
    if os.path.exists(feature_file):
        dev_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        dev_dataset = getData(tokenizer,data_path+'/dev.tsv')
        with open(feature_file, 'wb') as w:
            pickle.dump(dev_dataset, w)
    return dev_dataset


def getTestData(tokenizer,bert_name,data_path):
    feature_file = data_path+"/%s/test_features.pkl"%(bert_name.split('/')[-1])
    if os.path.exists(feature_file):
        test_dataset = pickle.load(open(feature_file, 'rb'))
    else:
        test_dataset = getData(tokenizer,data_path+'/test.tsv')
        with open(feature_file, 'wb') as w:
            pickle.dump(test_dataset, w)
    return test_dataset