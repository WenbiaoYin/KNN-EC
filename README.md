# KNN-EC

Code for EMNLP2022 paper: "**Efficient Nearest Neighbor Emotion Classification with BERT-whitening**"

## Abstract

Retrieval-based methods have been proven effective in many NLP tasks. Previous methods use representations from the pre-trained model for similarity search directly. However, the sentence representations from the pre-trained model like BERT perform poorly in retrieving semantically similar sentences, resulting in poor performance of the retrieval-based methods. In this paper, we propose KNN-EC, a simple and efficient non-parametric emotion classification (EC) method using nearest neighbor retrieval. We use BERT-whitening to get better sentence semantics, ensuring that nearest neighbor retrieval works. Meanwhile, BERT-whitening can also reduce memory storage of datastore and accelerate retrieval speed, solving the efficiency problem of the previous methods. KNN-EC average improves the pre-trained model by 1.17 F1-macro on two emotion classification datasets. 

## Overview

An overview of the proposed KNN-EC. The datastore stores the hidden representations of each sentence in the training data as keys and their corresponding labels as values. We use whitening transformation to enhance the isotropy of sentence representations and dimension reduction to optimize memory storage and accelerate retrieval speed. In inference, we use the whitening transformation on the test sentence's representation to retrieve the k nearest neighbors from the datastore. We interpolate the model and kNN distributions with a hyper-parameter $\lambda$ as the final distribution.

![model.png](https://github.com/WenbiaoYin/KNN-EC/blob/master/png/model.png?raw=true)

## Step of KNN-EC

### Step 1: Train your model and store the model's checkpoint

### Step 2: Use the model's checkpoint to get the sentences‘ hidden embedding in the training set and then use the whitening transformation on them and then save them on datastore.

<img src="https://github.com/WenbiaoYin/KNN-EC/blob/master/png/BERT-whitening.jpg?raw=true" alt="BERT-whitening.jpg" style="zoom:80%;" />

* Core code

```python
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
```

### Step 3: . In inference, we use the whitening transformation on the test sentence's representation to retrieve the k nearest neighbors from the datastore.

 We interpolate the model and kNN distributions with a hyper-parameter $\lambda$ as the final distribution.



## Visualization

Visualization of sentence representations on GoEmotions.

 (a) uses sentence representations from the fine-tuned RoBERTa without BERT-whitening, while (b) uses RoBERTa with BERT-whitening.

![Visualization.png](https://github.com/WenbiaoYin/KNN-EC/blob/master/png/Visualization.png?raw=true)



## Citation

```
@inproceedings{
  title={Efficient Nearest Neighbor Emotion Classification with BERT-whitening},
  author={Yin, Wenbiao and Shang, Lin},

}
```

