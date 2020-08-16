import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb


#http://jalammar.github.io/illustrated-bert/
#http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
#https://github.com/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb


#https://medium.com/ai%C2%B3-theory-practice-business/use-gpu-in-your-pytorch-code-676a67faed09
if torch.cuda.is_available():
  print(torch.cuda.get_device_name(0))


#test data 
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
df = df[:4000]
print(df[1].value_counts())
print(df[1][1])


model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = model_class.from_pretrained(pretrained_weights).cuda()
model = model_class.from_pretrained(pretrained_weights)


tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print(tokenized[0])

#add padding to test data 
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print("padded: ",np.array(padded).shape)


attention_mask = np.where(padded != 0, 1, 0)
print("mask: ",attention_mask.shape)

#https://stackoverflow.com/questions/56360644/pytorch-runtimeerror-expected-tensor-for-argument-1-indices-to-have-scalar-t
#input_ids = torch.tensor(padded).to(torch.int64).cuda()
#attention_mask = torch.tensor(attention_mask).cuda()
input_ids = torch.tensor(padded).to(torch.int64)
attention_mask = torch.tensor(attention_mask)


with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

#https://stackoverflow.com/questions/57832423/convert-cuda-tensor-to-numpy
#features = last_hidden_states[0][:,0,:].cpu().numpy()
features = last_hidden_states[0][:,0,:].numpy()

labels = df[1]


train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
parameters = {'C': np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(max_iter= 3000), parameters)
grid_search.fit(train_features, train_labels) 

print('best parameters: ', grid_search.best_params_)
print('best scores: ', grid_search.best_score_)



lr_clf = LogisticRegression(max_iter= 1000)
lr_clf.fit(train_features, train_labels)

print (lr_clf.score(test_features, test_labels))

#overall links used to do stuff throughout
#debug and other links found between code 
