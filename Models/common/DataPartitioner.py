""" This module has functionality that can partition the data into train,
    dev and test splits.
"""
import numpy as np


# function to perform synchronous random shuffling of the training data
def synch_random_shuffle_non_np(X, Y):
    """
        ** This function takes in the parameters that are non numpy compliant dtypes such as list, tuple, etc.
        Although this function works on numpy arrays as well, this is not as performant enough
        @param
        X, Y => The data to be shuffled
        @return => The shuffled data
    """
    combined = list(zip(X, Y))

    # shuffle the combined list in place
    np.random.shuffle(combined)

    # extract the data back from the combined list
    X[:], Y[:] = zip(*combined)

    # return the shuffled data:
    return X, Y


# function to split the data into train - dev sets:
def split_train_dev(X, Y, train_percentage):
    '''
        function to split the given data into two small datasets (train - dev)
        @param
        X, Y => the data to be split
        (** Make sure the train dimension is the first one)
        train_percentage => the percentage which should be in the training set.
        (**this should be in 100% not decimal)
        @return => train_X, train_Y, test_X, test_Y
    '''
    m_examples = len(X)
    assert train_percentage <= 100, "Train percentage cannot be greater than 100! NOOB!"
    partition_point = int((m_examples * (float(train_percentage) / 100)) + 0.5)  # 0.5 is added for rounding

    # construct the train_X, train_Y, test_X, test_Y sets:
    train_X = X[: partition_point]
    train_Y = Y[: partition_point]
    test_X = X[partition_point:]
    test_Y = Y[partition_point:]

    assert len(train_X) + len(test_X) == m_examples, "Something wrong in X splitting"
    assert len(train_Y) + len(test_Y) == m_examples, "Something wrong in Y splitting"

    # return the constructed sets
    return train_X, train_Y, test_X, test_Y
