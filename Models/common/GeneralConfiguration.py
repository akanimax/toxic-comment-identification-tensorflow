""" The module storing the general configuration information for the code
"""
from easydict import EasyDict as edict
import os

generalConf = edict({})

# Set the paths required for setting up and preprocessing the data
generalConf.DATA_PATH = os.path.join(os.path.abspath("."), "../../Data")
generalConf.TRAIN_DATA = os.path.join(generalConf.DATA_PATH, "train.csv")
generalConf.TEST_DATA = os.path.join(generalConf.DATA_PATH, "test.csv")
generalConf.SAMPLE_SUBMISSION = os.path.join(generalConf.DATA_PATH, "sample_submission.csv")

# Set the constants for the data
generalConf.MAX_WORD_LENGTH = 800 # derived from the common/visualizations/word_lengths.png plot


if __name__ == '__main__':
    print("Loading the General Configuration ...")
    print("\n\nGENERAL CONFIGURATION:\n")
    for key, value in generalConf.items():
        print(key, "->", value)