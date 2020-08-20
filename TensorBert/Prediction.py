from tensorflow.keras.models import load_model
from Init import BertModelLayer
import pandas as pd
from bert.tokenization.bert_tokenization import FullTokenizer
import os
import numpy as np
from tqdm import tqdm

bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir = os.path.join("model/", bert_model_name)
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
print('Loading the model')
model= load_model('saved_models/part10000.h5', custom_objects={'BertModelLayer': BertModelLayer})
print('Model is loaded')
print(model.summary())
test=pd.read_csv('articles-validation-bypublisher.csv')
conlist=test['content'].tolist()
idlist=test['id'].tolist()
print('conlist', len(conlist),'idlist',len(idlist))

resList=[]

for id, content in tqdm(zip(idlist, conlist)):
    pred_tokens = tokenizer.tokenize(content)
    pred_tokens = ["[CLS]"] + pred_tokens + ["[SEP]"]
    pred_token_ids = list(tokenizer.convert_tokens_to_ids(pred_tokens))
    if len(pred_token_ids) >= 512:
        pred_token_ids = pred_token_ids[:512]
    else:
        pred_token_ids = pred_token_ids + [0] * (512 - len(pred_token_ids))

    pred_token_ids = [pred_token_ids]

    pred_token_ids = np.array(pred_token_ids)

    predictions = model.predict(pred_token_ids).argmax(axis=-1)
    pred = 'true' if predictions[0] else 'false'
    resList.append(str(id) + ' ' + pred)

print('Writing into the file')
ResultFile=open('result.txt','w')
for element in resList:
     ResultFile.write(element)
     ResultFile.write('\n')
ResultFile.close()
print('Everything is done')