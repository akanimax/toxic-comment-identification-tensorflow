""" The endpoint script for training the model.
"""

import tensorflow as tf

from common.GeneralConfiguration import generalConf
from common.Pickler import unpickle_it
from common.DataPartitioner import *
from ModelConfiguration import modelConf

flags = tf.app.flags
FLAGS = flags.FLAGS


def setup_data():
    """
    Small helper method for setting up the data
    :return: train_X, train_Y, dev_X, dev_Y, test_X, vocabulary_proc
    """
    # load the pickled data
    print("Loading the pickled data ...")
    data_obj = unpickle_it(generalConf.PICKLE_FILE)

    train_data = data_obj['train_data']
    train_labels = data_obj['train_labels']
    test_data = data_obj['test_data']

    vocabulary_obj = data_obj['vocabulary_processor']

    # shuffle the training data
    train_data, train_labels = synch_random_shuffle_non_np(train_data, train_labels)

    # split the dataset into training and development set
    train_x, train_y, dev_x, dev_y = split_train_dev(train_data,
                                                     train_labels,
                                                     train_percentage=modelConf.TRAINING_PARTITION)

    # return the whole data
    return train_x, train_y, dev_x, dev_y, test_data, vocabulary_obj


def main(_):
    """ Function for the main script
    :param _: nothing. ignore
    :return: None
    """

    # obtain the data
    train_x, train_y, dev_x, dev_y, test_x, vocab_proc = setup_data()

    # print data related shape information
    print("Training_data information: ", train_x.shape, train_y.shape)
    print("Development data information: ", dev_x.shape, dev_y.shape)
    print("Test_data information: ", test_x.shape)

    # // TODO implement the remaining training script


if __name__ == '__main__':
    # define the command-line arguments for the script

    flags.DEFINE_integer("epochs", 12, help="The number of epochs for training the Network")

    # call the main function
    tf.app.run(main)
