#!/usr/bin/env python

"""Random baseline for the PAN19 hyperpartisan news detection task"""
# Version: 2018-09-24

# Parameters:
# --inputDataset=<directory>
#   Directory that contains the articles XML file with the articles for which a prediction should be made.
# --outputDir=<directory>
#   Directory to which the predictions will be written. Will be created if it does not exist.

from __future__ import division

import os
import getopt
import sys
from xml.etree import ElementTree
import random
from tensorflow.keras.models import load_model
from bert import BertModelLayer
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
from bs4 import BeautifulSoup
from tqdm import tqdm
random.seed(42)
runOutputFileName = "prediction.txt"


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir=", "modelType="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:m", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"
    modelType = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        elif opt in ("-m", "--modelType"):
            modelType = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    if modelType == "undefined":
        sys.exit("modeltype is undefined. Use option -m or --modelType.")
    elif modelType == "1":
        modelType = 'byarticle.h5'
    elif modelType == "2":
        modelType = 'bertFull_10000.h5' # by publisher full 768 bert model
    elif modelType == "3":
        modelType = 'tinybert300000.h5' # by publisher on tiny bert
    else :
        sys.exit("modeltype is undefined. Use option -m or --modelType.")

    return (inputDataset, outputDir, modelType)


########## SAX ##########

def clean(article):
    soup = BeautifulSoup(article, "html.parser")
    text = soup.get_text()
    text = text.strip()
    return text


def predict(article,tokenizer, model):
    article = clean(article)

    pred_tokens = tokenizer.tokenize(article)
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
    return pred

def element_to_string(element):
    s = element.text or ""
    for sub_element in element:
        s += ElementTree.tostring(sub_element, encoding='unicode')
    s += element.tail
    return s

########## MAIN ##########


def main(inputDataset, outputDir, modelType):
    print(inputDataset, outputDir, 'modelType : ',modelType)
    """Main method of this module."""
    print('--------------loading model----------------------')
    tokenizer = FullTokenizer(vocab_file=("/home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/vocab.txt"))
    model = load_model('/home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/models/'+modelType,
                       custom_objects={'BertModelLayer': BertModelLayer})
    print('------------------Model Loaded---------------------')
    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                tree = ElementTree.parse(inputDataset+ "/" +file)
                root = tree.getroot()
                print('Total articles ', len(root))
                for article in tqdm(root.iter('article')):
                    articleID = article.attrib['id']
                    content = element_to_string(article)
                    prediction = predict(content,tokenizer, model)
                    outFile.write(articleID + " " + prediction + "\n")

    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

