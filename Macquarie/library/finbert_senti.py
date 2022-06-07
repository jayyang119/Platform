import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig
from progress.bar import ChargingBar

#Class from the FinBERT documentations
class BertClassification(nn.Module):
    def __init__(self, weight_path, num_labels=2, vocab="base-cased"):
        super(BertClassification, self).__init__()
        self.num_labels = num_labels
        self.vocab = vocab 
        if self.vocab == "base-cased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=28996, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        elif self.vocab == "base-uncased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
            
        elif self.vocab == "finance-cased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=28573, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        elif self.vocab =="finance-uncased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=30873, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, graphEmbeddings=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
       
        logits = self.classifier(pooled_output)
            
        return logits

# FinBERT NLP function to assign the sentiment
def get_finbert_sentiments(sentences):
    
    bar_name = 'Getting Sentiments...'
    bar = ChargingBar("{0:<38}".format(bar_name), max = len(sentences))

    sentiments = []

    labels = {0:'neutral', 1:'positive',2:'negative'}
    num_labels= len(labels)
    vocab = 'finance-uncased'
    vocab_path = r'./analyst_tone/vocab'
    pretrained_weights_path = r'./analyst_tone/pretrained_weights' # this is pre-trained FinBERT weights
    fine_tuned_weight_path = r'./analyst_tone/fine_tuned.pth'      # this is fine-tuned FinBERT weights
    max_seq_length=512
    device='cuda:1'
    
    model = BertClassification(weight_path=pretrained_weights_path, num_labels=num_labels, vocab=vocab)
    
    model.load_state_dict(torch.load(fine_tuned_weight_path, map_location=torch.device('cpu')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True, do_basic_tokenize=True)

    model.eval()
    
    for sent in sentences: 
        tokenized_sent = tokenizer.tokenize(sent)
        if len(tokenized_sent) > max_seq_length:
            tokenized_sent = tokenized_sent[:max_seq_length]
        
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_sent)
        mask_input = [1]*len(ids_review)        
        padding = [0] * (max_seq_length - len(ids_review))
        ids_review += padding
        mask_input += padding
        input_type = [0]*max_seq_length
        
        input_ids = torch.tensor(ids_review).to(device).reshape(-1, max_seq_length)
        attention_mask =  torch.tensor(mask_input).to(device).reshape(-1, max_seq_length)
        token_type_ids = torch.tensor(input_type).to(device).reshape(-1, max_seq_length)
        
        with torch.set_grad_enabled(False):
            outputs = model(input_ids, token_type_ids, attention_mask)
            outputs = F.softmax(outputs,dim=1)
            sentiments.append(labels[torch.argmax(outputs).item()])
        
        bar.next()
        time.sleep(0.000001)
    
    bar.finish()

    return sentiments
