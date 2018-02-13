import unittest
from Model_1.GraphCreator import create_graph


class TestGraphCreator(unittest.TestCase):

    def test_create_graph(self):
        """ Test case for the create_graph method
        :return: None
        """

        secret_value = create_graph()
        self.assertEqual(secret_value, 42, "The Answer to universe, life, everything")
