import os
import yaml
import time
import torch
import faiss 
from tqdm import tqdm
from pathlib import Path
import numpy as np
from utils.bertWhiteness import BertWhitening,transform_and_normalize,neg_softmax,softmax
from models.model import  BertForSequenceClassification
from sklearn.metrics import classification_report,f1_score,accuracy_score
from utils.dataprocessor import getTrainData,getTestData,getDevData
from transformers import BertTokenizer,get_linear_schedule_with_warmup
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler

def train(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']
    
    model_name=config['model']['model_name'] # model_list=['bert-base-uncased','roberta-base','roberta-large']
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_data=getTrainData(tokenizer,model_name,config['data']['data_path'])
    dev_data=getDevData(tokenizer,model_name,config['data']['data_path'])
    
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels = config['model']['num_classes'], 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.cuda()
    
    
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['training']['train_batch_size'])
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=config['training']['dev_batch_size'])   
    
    optimizer = torch.optim.AdamW(model.parameters(),
                  lr = eval(config['training']['learning_rate']), # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    total_steps = len(train_dataloader) * config['training']['num_train_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = config['training']['warmup_prop'], # Default value in run_glue.py
                                                num_training_steps = total_steps
                )
    
    
    
    #train model
    for epoch in range(config['training']['num_train_epochs']):
        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            
            total_loss,step=0,0
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{config['training']['num_train_epochs']}")
                b_input_ids, b_input_mask,b_labels = batch[0].cuda(),batch[1].cuda(),batch[2].long().cuda()
                model.zero_grad()        

                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
                
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                predict=np.argmax(outputs[1].detach().cpu().numpy(), axis=1)
                step+=1
                tepoch.set_postfix(average_loss=total_loss/step,loss=loss.item(),f1=f1_score(batch[2].flatten(),predict.flatten(),average='weighted') ,accuracy='{:.3f}'.format(accuracy_score(batch[2].flatten(),predict.flatten())))
                time.sleep(0.0001)
        
        #eval model        
        model.eval() 
        true_labels,predict_labels=[],[]
        for batch in dev_dataloader:
            
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            logits = outputs[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predict_labels.append(np.argmax(logits, axis=1).flatten())
            true_labels.append(label_ids.flatten())
        
            
        true_labels=[y for x in true_labels for y in x]
        predict_labels=[y for x in predict_labels for y in x]
        print(classification_report(true_labels,predict_labels,digits=4))
        f1=f1_score(true_labels,predict_labels,average='macro')
    
    if config['training']['save_model']:
        torch.save(model, "/path_to_store_checkpoint/{}_{}.pt".format(model_name,f1))

def test(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']
    
    model_name=config['model']['model_name'] # model_list=['bert-base-uncased','roberta-base','roberta-large']
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model=torch.load(config['testing']['model_path'])
    model.cuda()
    
    test_data=getTestData(tokenizer,model_name,config['data']['data_path'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config['training']['test_batch_size'])
    
    #eval model        
    model.eval() 
    true_labels,predict_labels=[],[]
    for batch in test_dataloader:
        
        batch = tuple(t.cuda() for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predict_labels.append(np.argmax(logits, axis=1).flatten())
        true_labels.append(label_ids.flatten())
    
        
    true_labels=[y for x in true_labels for y in x]
    predict_labels=[y for x in predict_labels for y in x]
    print(classification_report(true_labels,predict_labels,digits=4))



def knnTest(config):
    np.random.seed(config['general']['seed'])
    torch.manual_seed(config['general']['seed'])
    torch.cuda.manual_seed_all(config['general']['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpu_ids']
    
    model_name=config['model']['model_name']
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model=torch.load(config['testing']['model_path'])
    model.cuda()
    
    datastore_path=config['knnTest']['datastore_path']+str(config['knnTest']['n_components'])+'.bin'
    if os.path.exists(datastore_path):
        index_embed=faiss.read_index(datastore_path)
        train_labels=np.load(config['knnTest']['train_labels'], mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
    else:
        train_data=getTrainData(tokenizer,model_name,config['data']['data_path'])
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
            embedd=outputs[1]
            kernel, bias =BertWhitening(embedd,config['knnTest']['n_components'])
            vecs=transform_and_normalize(embedd, kernel, bias)
            index_embed.add(vecs.cpu().detach().numpy())
            train_labels.append(b_labels)
        train_labels=np.array([label.item() for labels in train_labels for label in labels])
        faiss.write_index(index_embed,datastore_path)
        np.save(config['knnTest']['train_labels'],train_labels,allow_pickle=True, fix_imports=True)
    
    
    test_data=getTestData(tokenizer,model_name,config['data']['data_path'])
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config['knnTest']['test_batch_size'])
    
    
    k = config['knnTest']['k']   # we want to see k nearest neighbors
    alpha=config['knnTest']['alpha']
    
    true_labels,predict_labels=[],[]
    bert_labels,knn_labels=[],[]
    model.eval()
    for batch in test_dataloader:
        b_input_ids, b_input_mask,b_labels = batch[0].cuda(),batch[1].cuda(),batch[2]#.long().cuda()
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask)
        
        embedd=outputs[1]
        
        if config['knnTest']['n_components']==768:
            vecs=embedd
        else:
            kernel, bias =BertWhitening(embedd,config['knnTest']['n_components'])
            vecs=transform_and_normalize(embedd, kernel, bias)
        
        
        D, I = index_embed.search(vecs.cpu().detach().numpy(), k) 
        Distance=np.tile(neg_softmax(D,t=config['knnTest']['temperature']).reshape(-1,k,1), (1,6))
        Index_label=np.eye(6)[train_labels[I]]
        
        Knn_res=np.sum(Distance*Index_label, axis=1)
        bert_res=softmax(outputs[0].cpu().detach().numpy())
        res=(alpha*bert_res+(1-alpha)*Knn_res)
        
        predict_labels.append(np.argmax(res, axis=1).flatten())
        bert_labels.append(np.argmax(bert_res, axis=1).flatten())
        knn_labels.append(np.argmax(Knn_res, axis=1).flatten())
        true_labels.append(b_labels.flatten())

    true_labels=[y for x in true_labels for y in x]
    predict_labels=[y for x in predict_labels for y in x]
    knn_labels=[y for x in knn_labels for y in x]
    bert_labels=[y for x in bert_labels for y in x]
    
    print(classification_report(true_labels,predict_labels,digits=4))


def main():
    project_root: Path = Path(__file__).parent
    with open(str(project_root / "config.yml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    knnTest(config)

if __name__ == '__main__':
    main()
    