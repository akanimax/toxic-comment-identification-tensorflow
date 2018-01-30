""" Module for reading the csv files and returning the plain data from pandas dataframes
"""

import pandas as pd
import numpy as np

from GeneralConfiguration import generalConf


def read_csv_files():
    """
    The function for reading the csv file file
    :return: The train_text, test_text and the labels for training
    """

    # read the csv files and get all the text
    # bring the files for processing
    train_file = generalConf.TRAIN_DATA
    test_file = generalConf.TEST_DATA

    # load the csv as data frames
    print("Data_Paths: ", train_file, test_file)
    print("Reading the data from the csv_files ...")
    train_dataframe = pd.read_csv(train_file)
    test_dataframe = pd.read_csv(test_file)

    # carve out the text from the dataframe
    train_text = list(train_dataframe["comment_text"])
    test_text = list(test_dataframe["comment_text"])

    # obtain all the labels from the train_dataframe
    labels = np.array(train_dataframe[
                          ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]],
                      dtype=np.float32)

    return train_text, test_text, labels