""" The preprocessor script for extracting the data from the csv files and loading it up
    into numpy arrays
"""
import os
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from CSVReader import read_csv_files
from GeneralConfiguration import generalConf
from Pickler import pickle_it

flags = tf.app.flags
FLAGS = flags.FLAGS


def main(_):
    print("Preprocessing the data . . .")

    train_text, test_text, labels = read_csv_files()

    # convert the train_text and test_text into coded sequences
    print("displaying the sentences' word lengths...")
    if FLAGS.show_stats:
        word_lengths_train = list(map(len, map(lambda x: x.split(), train_text)))
        plt.figure().suptitle("Word Length v/s sent number")
        plt.plot(word_lengths_train)
        plt.show()
        word_lengths_test = list(map(len, map(lambda x: x.split(), test_text)))
        plt.figure().suptitle("Word Length v/s sent number")
        plt.plot(word_lengths_test)
        plt.show()

    # From the above figure (saved in common/visualizations) it can be observed that
    # max length of 800 is a good idea

    # Actual text preprocessing begins :) ->
    processor = tf.contrib.learn.preprocessing.VocabularyProcessor(generalConf.MAX_WORD_LENGTH)

    # fit the vocabulary on the
    print("Generating the vocabulary for the data ...")
    processor.fit(train_text + test_text)

    # Trim the vocabulary to minimum and maximum frequencies.
    processor.vocabulary_.trim(min_frequency = generalConf.MIN_WORD_FREQ,
                               max_frequency=generalConf.MAX_WORD_FREQ)

    print("Encoding the training data ...")
    encoded_train_text = np.array(list(map(list, processor.transform(train_text))), dtype=np.int32)
    print("Encoding the test data ...")
    encoded_test_text = np.array(list(map(list, processor.transform(test_text))), dtype=np.int32)

    # pickle the processed data
    data_obj = {
        "train_data": encoded_train_text,
        "train_labels": labels,
        "test_data": encoded_test_text,
        "vocabulary_processor": processor
    }

    # define the path for saving the preprocessed data
    save_path = os.path.join(generalConf.DATA_PATH, "plug_and_play.pickle")

    # pickle the file
    print("Saving the processed data ...")
    pickle_it(data_obj, save_path)

    print("Preprocessing complete ...")


if __name__ == '__main__':
    # define the command line arguments
    flags.DEFINE_boolean("show_stats", True, "To view various charts of the data")

    tf.app.run(main)
