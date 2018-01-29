""" The preprocessor script for extracting the data from the csv files and loading it up
    into numpy arrays
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from GeneralConfiguration import generalConf
from matplotlib import pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS


def main(_):
    print("Preprocessing the data . . .")

    # bring the files for processing
    train_file = generalConf.TRAIN_DATA
    test_file = generalConf.TEST_DATA

    # load the csv as data frames
    train_dataframe = pd.read_csv(train_file)
    test_dataframe = pd.read_csv(test_file)

    # carve out the text from the dataframe
    train_text = list(train_dataframe["comment_text"])
    test_text = list(test_dataframe["comment_text"])

    # obtain all the labels from the train_dataframe
    labels = np.array(train_dataframe[
                          ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]],
                      dtype=np.float32)

    # convert the train_text and test_text into coded sequences
    if FLAGS.show_stats:
        word_lengths = list(map(len, map(lambda x: x.split(), train_text)))
        plt.figure().suptitle("Word Length v/s sent number")
        plt.plot(word_lengths)
        plt.show()

    # From the above figure (saved in common/visualizations) it can be observed that
    # max length of 800 is a good idea

    # Actual text preprocessing begins :) ->
    processor = tf.contrib.learn.preprocessing.VocabularyProcessor(generalConf.MAX_WORD_LENGTH)

    # fit the vocabulary on the
    processor.fit(train_text)
    encoded_train_text = np.array(list(map(list, processor.transform(train_text))), dtype=np.int32)

    print(encoded_train_text.shape, encoded_train_text[:30])
    # print(max(map(len, encoded_train_text)))


if __name__ == '__main__':
    # define the command line arguments
    flags.DEFINE_boolean("show_stats", True, "To view various charts of the data")

    tf.app.run(main)
