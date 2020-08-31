import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb


if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))
#torch.cuda.empty_cache()

st = 'Data Files\\Clean files\\CSV\\articles-training-bypublisher.csv'
train = pd.read_csv(st,nrows=50)

st = 'Data Files\\Clean files\\CSV\\articles-validation-bypublisher.csv'
test = pd.read_csv(st,nrows=10)

train.pop('Unnamed: 0')
test.pop('Unnamed: 0')

train.head()
print(train['truth'].value_counts())
test.head()
print(test['truth'].value_counts())

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertForSequenceClassification, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights).cuda()
#model = model_class.from_pretrained(pretrained_weights)

def createID(arg):
    if arg == 'train':
        tokenized = train['content'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        for x in range(len(tokenized)):
            y = len(tokenized[x])
            if(y>512):
                start = tokenized[x][: 129]
                end = tokenized[x][y-383 :]
                temp = start + end
                tokenized[x] = temp
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        #input_ids = torch.tensor(padded).to(torch.int64)
        #attention_mask = torch.tensor(attention_mask)
        input_ids = torch.tensor(padded).to(torch.int64).cuda()
        attention_mask = torch.tensor(attention_mask).cuda()
        labels = torch.tensor(train['truth']).unsqueeze(0).cuda()
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask,labels=labels)
        #features = last_hidden_states[0][:,0,:].cpu().numpy()
        #features = last_hidden_states[0][:,0,:].numpy()
        #return features
        return last_hidden_states

    elif arg == 'test':
        tokenized = test['content'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        for x in range(len(tokenized)):
            y = len(tokenized[x])
            if(y>512):
                start = tokenized[x][: 129]
                end = tokenized[x][y-383 :]
                temp = start + end
                tokenized[x] = temp
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        #input_ids = torch.tensor(padded).to(torch.int64)
        #attention_mask = torch.tensor(attention_mask)
        input_ids = torch.tensor(padded).to(torch.int64).cuda()
        attention_mask = torch.tensor(attention_mask).cuda()
        labels = torch.tensor(test['truth']).unsqueeze(0).cuda()
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask,labels=labels)
        #features = last_hidden_states[0][:,0,:].cpu().numpy()
        #features = last_hidden_states[0][:,0,:].numpy()
        #return features
        return last_hidden_states


#train_features , train_labels = createID('train'), train['truth']
#test_features, test_labels = createID('test'), test['truth']
train_model = createID('train')
#test_model = createID('test')

loss, logits = train_model[:2]
print(loss)
print(logits)
print(len(logits))

model.eval(test['content'])