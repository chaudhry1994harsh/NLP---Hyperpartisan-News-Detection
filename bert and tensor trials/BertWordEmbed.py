import json
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb


st = 'Data Files\\Clean files\\Json\\articles-validation-bypublisher.json'
f = open(st,'r') 
data = json.load(f) 
f.close()

#https://www.geeksforgeeks.org/python-pandas-dataframe/
#https://stackoverflow.com/questions/21104592/json-to-pandas-dataframe
truth = []
content = []
ids = []
#i = 0
for elements in data:
    truth.append(elements['truth'])
    content.append(elements['content'])
    ids.append(elements['id'])
df = pd.DataFrame({'id':ids,'truth':truth,'content':content,})

df.head()
print(df['truth'].value_counts())
df['truth']
type(df)
#print(df['truth'][4])
#print(df['content'][4])

a=0
for x in df['content']:
    if(a<len(x.split())):
        a = len(x.split())
print(a)


model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = model_class.from_pretrained(pretrained_weights).cuda()
model = model_class.from_pretrained(pretrained_weights)

tokenized = df['content'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

tokenized
print(tokenized[1])
tokenized.head()
type(tokenized)
type(tokenized[1])
#https://stackoverflow.com/questions/402504/how-to-determine-a-python-variables-type

min=99999
max=0
avg=0
for x in tokenized:
    if(max<len(x)):
        max = len(x)
    if(min>len(x)):
        min = len(x)
    avg = avg + len(x)
print("min: ",min)
print("max: ",max)
print("avg: ",(avg/len(tokenized)))

#https://www.geeksforgeeks.org/python-truncate-a-list/
#https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification
#https://arxiv.org/pdf/1905.05583.pdf
outofPLACE = []
for x in range(len(tokenized)):
    y = len(tokenized[x])
    if(y>512):
        outofPLACE.append(x)
        start = tokenized[x][: 129]
        #print("start sz:",len(start), " ", start[0])
        end = tokenized[x][y-383 :]
        #print("end sz:",len(end), " ", end[len(end)-1])
        temp = start + end
        #print("temp sz:", len(x), ", start:",x[0],", end:",x[len(x)-1] )
        tokenized[x] = temp

print("done")

min=99999
max=0
avg=0
for x in tokenized:
    if(max<len(x)):
        max = len(x)
    if(min>len(x)):
        min = len(x)
    avg = avg + len(x)
print("min: ",min)
print("max: ",max)
print("avg: ",(avg/len(tokenized)))

#https://huggingface.co/transformers/main_classes/tokenizer.html
#https://huggingface.co/transformers/preprocessing.html
len(outofPLACE)
outofPLACE[0]
tokenized[outofPLACE[0]]
tokenizer.decode(tokenized[outofPLACE[0]])




#test vales for bypublisher files
'''
#f = open('Data Files\\Clean files\\Json\\articles-validation-bypublisher.json','r') 
#f = open('Data Files\\Clean files\\Json\\articles-training-bypublisher.json','r')
data = json.load(f) 
f.close()
truth = []
content = []
ids = []
for elements in data:
    truth.append(elements['truth'])
    content.append(elements['content'])
    ids.append(elements['id'])
df = pd.DataFrame({'id':ids,'truth':truth,'content':content})
print(df['truth'].value_counts())
print(df.head())'''

#https://arxiv.org/pdf/1905.05583.pdf
