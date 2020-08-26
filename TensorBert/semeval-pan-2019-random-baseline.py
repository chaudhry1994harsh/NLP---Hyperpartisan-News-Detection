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
import xml.sax
import random
from tensorflow.keras.models import load_model
from bert import BertModelLayer
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
from bs4 import BeautifulSoup

random.seed(42)
runOutputFileName = "prediction.txt"


def parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
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

    return (inputDataset, outputDir)


########## SAX ##########

def clean(article):
    soup = BeautifulSoup(article, "html.parser")
    text = soup.get_text()
    text = text.strip()
    return text


def predict(article):
    article = clean(article)
    tokenizer = FullTokenizer(vocab_file=("/home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/vocab.txt"))
    print('Loading the model')
    model = load_model('/home/zenith-kaju/NLP---Hyperpartisan-News-Detection/TensorBert/models/byarticle.h5', custom_objects={'BertModelLayer': BertModelLayer})

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

class HyperpartisanNewsRandomPredictor(xml.sax.ContentHandler):
    def __init__(self, outFile):
        xml.sax.ContentHandler.__init__(self)
        self.outFile = outFile

    def startElement(self, name, attrs):
        if name == "article":
            articleId = attrs.getValue("id") # id of the article for which hyperpartisanship should be predicted
            content = attrs.getValue("content")
            prediction = predict(content)
            #prediction = random.choice(["true", "false"]) # random prediction
            #confidence = random.random() # random confidence value for prediction
            # output format per line: "<article id> <prediction>[ <confidence>]"
            #   - prediction is either "true" (hyperpartisan) or "false" (not hyperpartisan)
            #   - confidence is an optional value to describe the confidence of the predictor in the prediction---the higher, the more confident
            self.outFile.write(articleId + " " + prediction+ "\n")


########## MAIN ##########


def main(inputDataset, outputDir):
    """Main method of this module."""

    with open(outputDir + "/" + runOutputFileName, 'w') as outFile:
        for file in os.listdir(inputDataset):
            if file.endswith(".xml"):
                with open(inputDataset + "/" + file) as inputRunFile:
                    xml.sax.parse(inputRunFile, HyperpartisanNewsRandomPredictor(outFile))


    print("The predictions have been written to the output folder.")


if __name__ == '__main__':
    main(*parse_options())

