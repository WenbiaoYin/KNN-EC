import os
import faiss
import numpy as np
from utils.dataprocessor import getTrainData
from torch.utils.data import DataLoader, SequentialSampler
from utils.bertWhiteness import transform_and_normalize

def buildDatastore(config,datastore_path,tokenizer,model,kernel,bias):
    if os.path.exists(datastore_path):
        index_embed=faiss.read_index(datastore_path)
        train_labels=np.load(config['knnTest']['train_labels'], mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    else:
        train_data=getTrainData(tokenizer,config['model']['model_name'],config['data']['data_path'])
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['knnTest']['train_datastore_size'])
        
        index_embed = faiss.IndexFlatL2(config['knnTest']['n_components'])
        
        model.eval()
        train_labels=[]
        for batch in train_dataloader:
            b_input_ids, b_input_mask,b_labels = batch[0].cuda(),batch[1].cuda(),batch[2]
            outputs = model(b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask)
            embedd=outputs[1].cpu().detach().numpy()
            vecs=transform_and_normalize(embedd, kernel[:,:config['knnTest']['n_components']], bias)
            index_embed.add(vecs.astype('float32'))
            train_labels.append(b_labels)
        train_labels=np.array([label.item() for labels in train_labels for label in labels])
        faiss.write_index(index_embed,datastore_path)
        np.save(config['knnTest']['train_labels'],train_labels,allow_pickle=True, fix_imports=True)
    return index_embed,train_labels