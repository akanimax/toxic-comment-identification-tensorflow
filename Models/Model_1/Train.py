""" The endpoint script for training the model.
"""

import tensorflow as tf
import os
import timeit  # for calculating the computational time
import numpy as np

# for running the script on command-line terminal
import sys

sys.path.append("../")

from common.GeneralConfiguration import generalConf
from common.Pickler import unpickle_it
from ResNet.GraphCreator import create_graph
from common.DataPartitioner import *
from Model_1.ResNet.ModelConfiguration import modelConf

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


def train_network(network, optimizer, epochs, train_x, train_y, dev_x, dev_y,
                  dropout_prob, batch_size, model_dir):
    """
    The function for training the network.
    :param dev_y: dev set data labels
    :param dev_x: dev set data
    :param train_y: train set data labels
    :param train_x: train set data
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
    accuracy = network.get_tensor_by_name("Accuracy/accuracy:0")

    # input placeholder for computational feeding
    input_x = network.get_tensor_by_name("Inputs/Input_sequences:0")
    input_y = network.get_tensor_by_name("Inputs/Sequence_labels:0")
    dropout_keep_prob = network.get_tensor_by_name("Inputs/drop_out_keep_probability:0")
    dropout_mode = network.get_tensor_by_name("Inputs/drop_out_mode:0")
    batch_norm_mode = network.get_tensor_by_name("Inputs/batchNorm_mode:0")

    # setup the training optimizer for the graph
    with network.as_default():
        # add the training optimizer
        with tf.name_scope("Trainer"):
            train_step = optimizer.minimize(loss)

        # add the general errands
        with tf.name_scope("Errands"):
            init = tf.global_variables_initializer()
            all_sums = tf.summary.merge_all()

    print("\nStarting the Network Training ...")
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
            # record the time
            start_time = timeit.default_timer()

            # loop over the mini_batches in the data
            # run through the batches of the data:
            total_train_examples = train_x.shape[0]
            losses = []
            accuracies = []
            summaries = None
            for batch in range(int((total_train_examples / batch_size) + 0.5)):
                start = batch * batch_size
                end = start + batch_size

                # extract the relevant data:
                batch_data_x = train_x[start: end]
                batch_data_y = train_y[start: end]

                # run the training step and calculate the loss and accuracy
                _, cost, acc, summaries = sess.run([train_step, loss, accuracy, all_sums],
                                                   feed_dict={
                                                       input_x: batch_data_x,
                                                       input_y: batch_data_y,
                                                       dropout_keep_prob: dropout_prob,
                                                       dropout_mode: True,
                                                       batch_norm_mode: True
                                                   })

                # append the acc to the accuracies list
                accuracies.append(acc)

                # append the loss to the losses list
                losses.append(cost)

            stop_time = timeit.default_timer()

            print("\nepoch = ", epoch + 1, "time taken = ", (stop_time - start_time))

            avg_acc = np.mean(accuracies)
            avg_loss = np.mean(losses)
            print("Average epoch loss = ", avg_loss)
            print("Average epoch input accuracy = ", avg_acc)

            # evaluate the accuracy for the dev set
            dev_acc = sess.run(accuracy,
                               feed_dict={
                                   input_x: dev_x,
                                   input_y: dev_y,
                                   dropout_mode: False,
                                   dropout_keep_prob: dropout_prob,
                                   batch_norm_mode: False
                               })
            print("dev_accuracy = ", dev_acc)

            # write the summaries to the file_system
            summaries.value.add(tag="%strain_accuracy" % "Accuracy/", simple_value=avg_acc)
            summaries.value.add(tag="%sdev_accuracy" % "Accuracy/", simple_value=dev_acc)
            summaries.value.add(tag="%sloss" % "Loss/", simple_value=avg_loss)
            tensorboard_writer.add_summary(summaries, epoch + 1)

            # save the model after every epoch
            saver.save(sess, os.path.join(model_dir, "Model"), global_step=(epoch + 10))
    print("Training complete ...")


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

    # generate the model name based on the configuration
    model_name = ("Model-" + str(FLAGS.network_depth) + "-deep-" +
                  str(FLAGS.fc_layer_width) + "-flw-" +
                  str(FLAGS.fc_layer_depth) + "-fld-" +
                  str(FLAGS.epochs) + "-epochs-" +
                  str(FLAGS.learning_rate) + "-learning_rate-" +
                  str(FLAGS.dropout_prob) + "-regularization_lambda-" +
                  str(FLAGS.batch_size) + "-batch_size")

    # model_save directory
    model_save_dir = os.path.join("Trained_Models", model_name)

    # call the training method
    train_network(
        network_graph,
        tf.train.AdamOptimizer(),
        FLAGS.epochs,
        train_x, train_y, dev_x, dev_y,
        FLAGS.dropout_prob,
        FLAGS.batch_size,
        model_save_dir
    )


if __name__ == '__main__':
    # define the command-line arguments for the script

    flags.DEFINE_integer("epochs", 12, "The number of epochs for training the Network")
    flags.DEFINE_integer("network_depth", 0,"The Depth of the Network (No. of residual blocks)")
    flags.DEFINE_integer("fc_layer_width", 512, "The width of the final dense layers")
    flags.DEFINE_integer("fc_layer_depth", 1, "The number of dense layer in the end")
    flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for training")
    flags.DEFINE_float("dropout_prob", 0.5, "Dropout probability for the final dense layers")
    flags.DEFINE_integer("batch_size", 128, "Batch size for sgd")

    # call the main function
    tf.app.run(main)
