""" The module storing the Model specific configuration
"""
from easydict import EasyDict as eDict

modelConf = eDict({})


# Set the constants for the Model
modelConf.FILTER_SIZE = 5 # all 1D convolutional filters have a width of 5
modelConf.TRAINING_PARTITION = 99 # percentage for training data. rest is the validation set


if __name__ == '__main__':
    print("Loading the General Configuration ...")
    print("\n\nGENERAL CONFIGURATION:\n")
    for key, value in modelConf.items():
        print(key, "->", value)