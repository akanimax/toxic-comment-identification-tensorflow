""" The endpoint script for training the model.
"""

import tensorflow as tf
import os

from common.GeneralConfiguration import generalConf
from common.Pickler import unpickle_it
from ResNet.GraphCreator import create_graph
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


def train_network(network, optimizer, epochs, dropout_prob, batch_size, model_dir):
    """
    The function for training the network.
    :param batch_size: The size of mini-batches for training the network
    :param dropout_prob: The dropout probability for the final dense layers
    :param epochs: Number of epochs to train for
    :param network: The computation graph following the interface names
    :param optimizer: The optimizer (Tensorflow train Optimizer) to be used
    :param model_dir: The directory where the model is to be saved
    :return: None
    """
    # extract the required tensors from the network
    loss = network.get_tensor_by_name("Loss/loss:0")

    # setup the training optimizer for the graph
    with network.as_default():
        # add the training optimizer
        with tf.name_scope("Trainer"):
            train_step = optimizer.minimize(loss)

        # add the general errands
        with tf.name_scope("Errands"):
            init = tf.global_variables_initializer()
            all_sums = tf.summary.merge_all()

    # start the training Session
    with tf.Session(graph=network) as sess:
        # setup the tensorboard writer
        tensorboard_writer = tf.summary.FileWriter(model_dir, graph=sess.graph, filename_suffix=".bot")

        # setup the model saver
        saver = tf.train.Saver(max_to_keep=3)

        # load the previously saved network
        if os.path.isfile(os.path.join(model_dir, "checkpoint")):
            # restore the model
            saver.restore(sess, model_dir)

        else:
            # initialize the network
            sess.run(init)

        # start the training loop
        for epoch in range(epochs):
            pass


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

    # create the arguments for the create_graph method
    seq_length = train_x.shape[-1]
    num_classes = train_y.shape[-1]
    vocab_size = len(vocab_proc.vocabulary_.__dict__['_mapping'])
    print("Vocabulary Size: ", vocab_size)

    emb_size = modelConf.EMB_SIZE
    network_depth = FLAGS.network_depth
    fc_layer_width = FLAGS.fc_layer_width
    fc_layer_depth = FLAGS.fc_layer_depth

    # generate the graph
    network_graph = create_graph(seq_length, num_classes, vocab_size, emb_size,
                                 network_depth, fc_layer_width, fc_layer_depth)

    print("Computation graph generated: ", network_graph)


if __name__ == '__main__':
    # define the command-line arguments for the script

    flags.DEFINE_integer("epochs", 12, help="The number of epochs for training the Network")
    flags.DEFINE_integer("network_depth", 0, help="The Depth of the Network (No. of residual blocks)")
    flags.DEFINE_integer("fc_layer_width", 512, help="The width of the final dense layers")
    flags.DEFINE_integer("fc_layer_depth", 1, help="The number of dense layer in the end")

    # call the main function
    tf.app.run(main)
