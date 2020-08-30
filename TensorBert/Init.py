"""
Task : Given a news article text, decide whether it follows a
hyperpartisan argumentation, i.e., whether it exhibits blind,
prejudiced, or unreasoning allegiance to one party, faction, cause,
or person.
https://pan.webis.de/semeval19/semeval19-web/

This file creates or loads a BERT model based on Hyperpartisan
Dataset and save it in the file system for later use. The
functions train the model on the Hyperpartisan dataset and gives out
training and test accuracies.

BERT
https://github.com/google-research/bert

BERT for Tensorflow
https://github.com/kpe/bert-for-tf2
https://www.curiousily.com/posts/intent-recognition-with-bert-using-keras-and-tensorflow-2/
https://www.tensorflow.org/guide/keras/save_and_serialize
https://machinelearningspace.com/sentiment-analysis-tensorflow/
https://keras.io/api/models/model/

Input Data : A preprocessed JSON file is given as input to this file.
The Hyperpartisan Dataset XML files were parsed and converted into
JSON files for easy processing of data using pandas dataframe

"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from tensorflow.keras.models import load_model
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import confusion_matrix, classification_report
from PartisanDetectionData import PartisanDetectionData
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def getInputData(filename):
    """
    convert JSON file data into pandas dataframe
    :param filename: file containing the dataset
    :return: pandas dataframe
    """
    return pd.read_csv(filename)


def create_model(max_seq_len, bert_ckpt_file,bert_config_file, classes):
    """
    Create a keras model based on bert model checkpoint file, config
    file and target classes.
    Return a keras model which includes a BERT layer
    """
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)


    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)

    return model

def run():
    """ Function to create or load a NN model and train over the hyperpartisan datasets"""
    train = getInputData('trainmain_10000.csv')
    test = getInputData('testground_10000.csv')

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    bert_model_name="uncased_L-12_H-768_A-12"

    bert_ckpt_dir = os.path.join("model/", bert_model_name)
    bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
    bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))

    classes = train['truth'].unique().tolist()

    data = PartisanDetectionData(train, test, tokenizer, classes, max_seq_len=512)
    with strategy.scope():
        model = create_model(data.max_seq_len, bert_ckpt_file,bert_config_file, classes)
        print(model.summary())
        model.compile(
          optimizer=keras.optimizers.Adam(1e-5),
          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )
    model_dir= 'saved_models/'
    model_name = 'tensorflow_10000.h5'
    if not os.path.isfile(model_dir+model_name):
        history = model.fit(
          x=data.train_x,
          y=data.train_y,
          validation_split=0.1,
          batch_size=64,
          shuffle=True,
          epochs=5
        )
        model.save(model_dir+model_name)
    else:
        model = load_model(model_dir+model_name, custom_objects={'BertModelLayer': BertModelLayer})

    _, train_acc = model.evaluate(data.train_x, data.train_y)
    _, test_acc = model.evaluate(data.test_x, data.test_y)

    print("train acc", train_acc)
    print("test acc", test_acc)
    y_pred = model.predict(data.test_x).argmax(axis=-1)

    with open(model_dir+model_name.split()[0]+'.txt', 'w') as fh:
        all_lines = fh.write(classification_report(data.test_y.tolist(), y_pred, target_names=['0','1']))


if __name__ == '__main__':
    run()