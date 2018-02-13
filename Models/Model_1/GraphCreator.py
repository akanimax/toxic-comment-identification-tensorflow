""" This module has the functionality for creating the 1D Convolutional Neural Network.
    The network has the architecture defined as:
    conv -> pool -> conv ... so on.
"""

import tensorflow as tf


def create_graph():
    """ Function for creating the Graph object that is going to be trained

    :return: computation graph object
    """

    my_graph = tf.Graph()

    # bring this new graph into scope in order to modify it
    with my_graph.as_default():
        pass

    return 42 # the secret answer to everything
