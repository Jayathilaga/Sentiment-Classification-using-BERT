from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel
import torch
import sys
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 300
sentiment = ['1star', '2star', '3star', '4star', '5star']
def data_preprocessing(text):
    print(text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def bert_preprocessing(data):
    input_ids = []
    attention_masks = []
    for i in data:
        encoded_input = tokenizer.encode_plus(text=data_preprocessing(i),add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True,return_attention_mask=True)
        input_ids.append(encoded_input.get('input_ids'))
        attention_masks.append(encoded_input.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks


class Classifier_For_Bert(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Classifier_For_Bert, self).__init__()
        input, hidden, output = 768, 150, 5
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(input, hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, output)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)        
        hs = outputs[0][:, 0, :]
        logits = self.classifier(hs)
        return logits

if __name__ == '__main__':
    model = Classifier_For_Bert(freeze_bert=False)
    model.load_state_dict(torch.load("sentiment_model_4"))
    model.eval()
    review_text = sys.argv[1]
    input_ids,attention_mask = bert_preprocessing([review_text])
    output = model(input_ids, attention_mask)
    prediction = torch.argmax(output, dim=1).flatten()
    print("\n\n=============================================================================================")
    print(f'Input Review: {review_text}')
    print(f'Predicted Sentiment: {sentiment[prediction]}')
    print("=============================================================================================")