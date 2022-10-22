from models.model import  BertForSequenceClassification,RobertaForSequenceClassification
from transformers import BertTokenizer,RobertaTokenizer


def get_model(config):
    model_name=config['model']['model_name']
    if model_name.split('-')[0]=='bert':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = config['model']['num_classes'], 
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
    
    elif  model_name.split('-')[0]=='roberta':
        tokenizer=RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
                model_name, 
                num_labels = config['model']['num_classes'], 
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
            )
    else:
        raise IndexError("Please enter correct model name!")
    
    return model,tokenizer
    
    