""" Test Cases for the GraphCreator Module
"""

import unittest
from common.GeneralConfiguration import generalConf
from ResNet.GraphCreator import *

# temp mock variables
vocab_size = 10000
emb_size = 16


class TestGraphCreator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.seq_length, cls.num_classes = generalConf.MAX_WORD_LENGTH, generalConf.NUM_CLASSES
        cls.graph = create_graph(cls.seq_length, cls.num_classes, vocab_size, emb_size)

    def test_one_hot_encoding(self):
        # test the one_hot encoded version
        one_hot = self.graph.get_tensor_by_name("OneHot_Encoding/one_hot_encoded_labels:0")

        # test if the one hot encoding shape is correct
        self.assertEqual(one_hot.shape.as_list(), [None, self.num_classes])

    def test_embeddings(self):
        # test the one_hot encoded version
        embedded_inp = self.graph.get_tensor_by_name("Embedding/Embedded_input:0")

        # test if the one hot encoding shape is correct
        self.assertEqual(embedded_inp.shape.as_list(), [None, self.seq_length, emb_size])

    def test_residual_block_with_parallel_conv(self):
        # create a mock input
        inp = tf.placeholder(tf.float32, shape=(None, 800, 16), name="mock_input")

        # get output from the method
        output = residual_block_with_parallel_conv(1, inp, 5, 32, True)

        # check the tensor shape of the output
        self.assertEqual(output.shape.as_list(), [None, 800, 32])

    def test_pooling2x(self):
        # create a mock input
        inp = tf.placeholder(tf.float32, shape=(None, 25, 16), name="mock_input")

        # get output from the method
        output = pooling2x(inp)

        # check the tensor shape of the output (for odd number it rounds down)
        self.assertEqual(output.shape.as_list(), [None, 12, 16])