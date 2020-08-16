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


st = 'Clean Data Files\\clnSMALL.json'
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
    truth.append(elements['hyperpartsan'])
    content.append(elements['content'])
    ids.append(elements['_id'])
    '''x = elements['content']
    damn = x.split()
    if len(damn) > 510:
        i = i +1
        nue = ""
        for jam in range(0,128):
            nue = nue + " " + damn[jam]
        for jam in range(len(damn)-382,len(damn)):
            nue = nue + " " + damn[jam]
        nue.strip()
        content.append(nue)
    else:
        content.append(x)
print("i: ",i)
print(len(truth))
print(len(content))'''
df = pd.DataFrame({'content':content,'truth':truth})

print(df['truth'].value_counts())
#print(df['truth'][4])
#print(df['content'][4])

a=0
for x in df['content']:
    if(a<len(x.split())):
        a = len(x.split())
print(a)

'''cnt = 0
lol = 0
cnter = 0
total = 0
for x in df['content']:
    damn = x.split()
    cnt = cnt + len(damn)
    if len(damn) > 510:
        cnter = cnter + 1
        nue = ""
        for jam in range(0,128):
            nue = nue + " " + damn[jam]
        for jam in range(len(damn)-382,len(damn)):
            nue = nue + " " + damn[jam]
        nue.strip()
        lol = lol + len(nue.split())
        total = total + len(nue.split())
        print(len(nue.split()))
        df['content'].replace({x:nue},inplace=True)
    else:
        total = total + len(damn)

print (cnt)
cnt = cnt/645
print (cnt)
print(lol)
print (lol/cnter)
print(cnter)
print(total/645)'''




model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = model_class.from_pretrained(pretrained_weights).cuda()
model = model_class.from_pretrained(pretrained_weights)

tokenized = df['content'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))






f = open('Clean Data Files\\clnLARGE.json','r') 
data = json.load(f) 
f.close()
truth = []
content = []
ids = []
for elements in data:
    truth.append(elements['hyperpartsan'])
    content.append(elements['content'])
    ids.append(elements['_id'])
df = pd.DataFrame({'id':ids,'content':content,'truth':truth})
print(df['id'][15])
print(df['truth'].value_counts())
print(df['content'][15])
print(df.head())

#https://arxiv.org/pdf/1905.05583.pdf