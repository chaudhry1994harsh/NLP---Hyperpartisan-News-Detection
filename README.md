# SemEval 2019 Hyperpartisan-News-Detection

[Task]: Given a news article text, decide whether it follows a hyperpartisan argumentation, i.e., whether it exhibits blind, prejudiced, or unreasoning allegiance to one party, faction, cause, or person.

## Dataset
2 datasets were provided:
* **By Article** : It is labeled through crowdsourcing on an article basis. The data contains only articles for which a consensus among the crowdsourcing workers existed. It contains a total of 645 articles. Of these, 238 (37%) are hyperpartisan and 407 (63%) are not. The test set contains 628 articles.
* **By Publisher** : It contains a total of 750,000 articles, half of which (375,000) are hyperpartisan and half of which are not. Half of the articles that are hyperpartisan (187,500) are on the left side of the political spectrum, half are on the right side. This data is split into a training set (80%, 600,000 articles) and a validation set (20%, 150,000 articles). The test set contains 4000 articles.
  
## Training
Primarily BERT language model was used to make classifiers.
In [Tensorbert] folder, tensorflow was used to make corresponding BERT layer in  the model but in [distilBERT TF2] folder pytorch transformer's BERT layer was used. 

## Evaluation
THe models were evaluated on TIRA platform. The test sets were not available directly and could only be tested via TIRA.

### Results on 'By Article' dataset
| Model|Accuracy | Precision | Recall | F-score |
| ------ | ------ | ------ | ------ |
| DistilBert-base acc | 0.718 | 0.920 | 0.477 | 0.628 |
|Bert-base | .590  | 0.551 | 0.974 | 0.704 |
|DistilBert-base loss | 0.724  | 0.917 | 0.493 | 0.641 |

### Results on 'By Publisher' dataset
| Model|Accuracy | Precision | Recall | F-score |
| ------ | ------ | ------ | ------ |
|DistilBert-base  | 0.521 | 0.561 | 0.193 | 0.287 |
|Tiny bert | 0.646 | 0.610 | 0.804 | 0.694 |





[Task]: <https://pan.webis.de/semeval19/semeval19-web/>
[Tensorbert]: <https://github.com/chaudhry1994harsh/NLP---Hyperpartisan-News-Detection/tree/master/TensorBert>
[distilBERT TF2]: <https://github.com/chaudhry1994harsh/NLP---Hyperpartisan-News-Detection/tree/master/distilBERT%20TF2>
