""" This module has the functionality for creating the 1D Convolutional Neural Network.
    The network has the architecture defined as:
"""

import tensorflow as tf

from Model_1.ResNet.ModelConfiguration import modelConf


# =============================================================================================
# helper methods for the graph creation
# =============================================================================================
def residual_block_with_parallel_conv(num_id, inp, filter_size, residual_block_channels, bn_mode):
    """
    The function for adding a residual block to the network
    consists of conv -> bn -> Relu -> conv -> bn -> Relu -> conv -> bn -> |
                  |---------------------------------------> conv -> bn -> Relu -> output
    :param num_id: The id for name_scope
    :param inp: input tensor (pipeline)
    :param filter_size: size of the 1d convolution window
    :param residual_block_channels: Dimension of the residual block
    :param bn_mode: for controlling the batch normalization mode
    :return: output tensor
    """
    with tf.variable_scope("Residual_Parallel_Block" + str(num_id)):
        x = inp  # store for residual connection

        # normal connections ->
        x1 = tf.layers.conv1d(x, residual_block_channels, filter_size,
                              strides=1, padding="SAME", name="conv1")
        x1 = tf.layers.batch_normalization(x1, training=bn_mode, name="bn1")
        x1 = tf.nn.relu(x1)

        x1 = tf.layers.conv1d(x1, residual_block_channels, filter_size,
                              strides=1, padding="SAME", name="conv2")
        x1 = tf.layers.batch_normalization(x1, training=bn_mode, name="bn2")
        x1 = tf.nn.relu(x1)

        x1 = tf.layers.conv1d(x1, residual_block_channels, filter_size,
                              strides=1, padding="SAME", name="conv3")
        x1 = tf.layers.batch_normalization(x1, training=bn_mode, name="bn3")

        # parallel residual connection ->
        x2 = tf.layers.conv1d(x, residual_block_channels, filter_size,
                              strides=1, padding="SAME", name="parallel_conv")
        x2 = tf.layers.batch_normalization(x2, training=bn_mode, name="parallel_bn")

        # add the two parallel paths
        y = tf.add(x1, x2, name="block_output")

    # add the histogram summaries for these weights and biases
    with tf.variable_scope("Residual_Parallel_Block" + str(num_id), reuse=True):
        for lay in range(1, 4):
            weight = tf.get_variable("conv" + str(lay) + "/kernel")
            bias = tf.get_variable("conv" + str(lay) + "/bias")

            # add histogram summary for these weights and biases
            tf.summary.histogram("conv" + str(lay) + "/kernel", weight)
            tf.summary.histogram("conv" + str(lay) + "/bias", bias)

        # add the histogram summary for the parallel connection
        par_weight = tf.get_variable("parallel_conv/kernel")
        par_bias = tf.get_variable("parallel_conv/bias")

        tf.summary.histogram("parallel_conv/kernel", par_weight)
        tf.summary.histogram("parallel_conv/bias", par_bias)

    # return the computed output
    return y


def pooling2x(inp):
    """
    Function for performing half pooling of the data
    :param inp: Input tensor
    :return: pooled output tensor
    """
    # pool to half dimensions
    with tf.name_scope("2xPooling"):
        y = tf.layers.max_pooling1d(inp, 2, 2)

    # return the output
    return y


# =============================================================================================


# =============================================================================================
# The graph creator method
# =============================================================================================
def create_graph(sequence_length, num_classes, vocab_size, emb_size, network_depth, max_depth=9,
                 fc_layer_width=512, fc_layer_depth=3):
    """ Function for creating the Graph object that is going to be trained

    :return: computation graph object
    """

    my_graph = tf.Graph()

    # bring this new graph into scope in order to modify it
    print("Constructing computational graph ...")
    print("===========================================================================================")
    with my_graph.as_default():
        # define the placeholders for the computation graph
        with tf.name_scope("Inputs"):
            input_x = tf.placeholder(tf.int32, shape=(None, sequence_length), name="Input_sequences")
            input_y = tf.placeholder(tf.float32, shape=(None, num_classes), name="Sequence_labels")

            # placeholder for applying dropout to the final classifying layers
            dropout_keep_prob = tf.placeholder(tf.float32, shape=None, name="drop_out_keep_probability")
            dropout_mode = tf.placeholder(tf.bool, shape=None, name="drop_out_mode")

            # placeholder for controlling the batch normalization
            batch_norm_mode = tf.placeholder(tf.bool, shape=None, name="batchNorm_mode")

        print("Defined Inputs to graph: ", input_x, input_y)
        print("Input Training parameters: ", dropout_keep_prob, dropout_mode, batch_norm_mode)

        # define the embedding layer
        with tf.variable_scope("Embedding"):
            embeddings = tf.get_variable("Embedding_Matrix", shape=(vocab_size, emb_size), dtype=tf.float32)
            embedded_x = tf.nn.embedding_lookup(embeddings, input_x, name="Embedded_input")

        print("Embedded inputs: ", embedded_x)

        # create the computational network
        # validate the network_depth
        network_depth = max(0, network_depth)  # lower limit clipping
        network_depth = min(max_depth, network_depth)  # upper limit clipping

        # start the network pipeline
        filter_size = modelConf.FILTER_SIZE
        channel_width = embedded_x.shape.as_list()[-1]  # input channel width
        x = embedded_x
        for lay in range(network_depth):
            # update the channel width
            channel_width = 2 * channel_width
            x = residual_block_with_parallel_conv((lay + 1), x, filter_size, channel_width, batch_norm_mode)
            x = pooling2x(x)

        x = tf.identity(x, "last_conv_output")

        print("Final output from convolutional pipeline: ", x)

        # validate the fc_layer_depth
        fc_layer_depth = max(0, fc_layer_depth)

        # validate the fc_layer_width
        fc_layer_width = max(1, fc_layer_width)
        # width cannot be zero or -ve

        # define the final fully connected layers with the last one being for classification
        y = tf.layers.flatten(x, name="flatten_output")  # start the dense layer computations
        for fc_lay in range(fc_layer_depth):
            lay_no = str(fc_lay + 1)
            with tf.variable_scope("Dense_Dropout" + str(lay_no)):
                # apply fully connected layer
                y = tf.layers.dense(y, fc_layer_width, activation=tf.nn.relu,
                                    name="Dense_Layer_" + lay_no)
                # apply dropout
                y = tf.layers.dropout(y, rate=dropout_keep_prob, training=dropout_mode,
                                      name="Dropout" + lay_no)

            # add summaries for the dense layers
            with tf.variable_scope("Dense_Dropout" + str(lay_no), reuse=True):
                # add the histogram summary for the dense layers
                tensor_name = "Dense_Layer_" + lay_no
                tf.summary.histogram(tensor_name + "/kernel", tf.get_variable(tensor_name + "/kernel"))
                tf.summary.histogram(tensor_name + "/bias", tf.get_variable(tensor_name + "/bias"))

        # define the final classification fully connected layer
        y = tf.layers.batch_normalization(y, training=batch_norm_mode, name="final_batch_normalization")
        raw_preds = tf.layers.dense(y, num_classes, name="Raw_Predictions")

        # define the sigmoid predictions
        with tf.name_scope("Predictions"):
            preds = tf.nn.sigmoid(raw_preds, name="preds")

        print("Predictions obtained at the end: ", preds)

        # define the classification loss
        with tf.name_scope("Loss"):
            # note that it is raw_preds that we pass here
            loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=input_y,
                                                        logits=raw_preds), name="loss")

        # define the accuracy metric
        with tf.name_scope("Accuracy"):
            correct = tf.equal(input_y, tf.round(preds, name="round_probabilities"))
            all_labs_true = tf.reduce_min(tf.cast(correct, tf.float32, name="correct_calculation"), axis=-1)

            accuracy = tf.reduce_mean(all_labs_true, name="accuracy")
            # accuracy is in fraction (not %age)

    print("===========================================================================================")
    print("Graph construction complete ...")

    return my_graph

# =============================================================================================
