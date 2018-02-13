""" The main script that combines all the tests for this package in a suite and runs it
"""

import unittest
from test_GraphCreator import TestGraphCreator


def create_suite():
    # create the suite object
    suite = unittest.TestSuite()

    # add all the tests in the suite
    suite.addTest(unittest.makeSuite(TestGraphCreator))

    # finally return the generated suite object
    return suite


# the main test runner script
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = create_suite()

    # run all the tests in the test suite
    runner.run(test_suite)