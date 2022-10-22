import numpy as np

def BertWhitening(vecs):
    """
    compute kernel and bias; Final transformation:y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1/np.sqrt(s)))
    return W, -mu

def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return normalize(vecs)

def neg_softmax(x,t=1):
    return np.exp(-x*t) / np.sum(np.exp(-x*t), axis=-1, keepdims=True)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sents2vecs(dataloader,model):
    model.eval()
    labels=[]
    vec=[]
    for batch in dataloader:
        b_input_ids, b_input_mask,b_labels = batch[0].cuda(),batch[1].cuda(),batch[2]
        outputs = model(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask)
        embedd=outputs[1].detach().cpu().numpy()
        vec.append(embedd)
        labels.append(b_labels)
    
    labels=np.array([t.item() for label in labels for t in label])
    vecs=np.vstack(vec)
    
    return vecs,labels