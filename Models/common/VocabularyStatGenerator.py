""" The module that generates the max and min frequency counts for the words in the total vocabulary.
"""
import collections
import matplotlib.pyplot as plt
import tensorflow as tf

from CSVReader import read_csv_files


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]  # start with this list.
    count.extend(collections.Counter(words).most_common(n_words - 1))  # this is inplace. i.e. has a side effect

    dictionary = dict()  # initialize the dictionary to empty one
    # fill this dictionary with the most frequent words
    for word, _ in count:
        dictionary[word] = len(dictionary)

    # loop to replace all the rare words by the UNK token
    data = list()  # start with empty list
    unk_count = 0  # counter for keeping track of the unknown words
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count  # replace the earlier -1 by the so calculated unknown count

    print("Total rare words replaced: ", unk_count)  # log the total replaced rare words

    # construct the reverse dictionary for the original dictionary
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # return all the relevant stuff
    return data, count, dictionary, reversed_dictionary


def main(_):
    # read the csv files to get the data
    train_text, test_text, _ = read_csv_files()

    # concatenate the lists to view the total text
    total_text = train_text + test_text

    # obtain the histogram of the words in the
    word_lines = []  # initialize to empty list
    for line in total_text:
        word_lines += (list(map(lambda x: x.lower(), line.strip().split())))

    _, cnt, _, _ = build_dataset(word_lines, len(word_lines))

    print(cnt[:10], len(cnt))

    # extract counts from the cnt
    plt.figure().suptitle("Word frequency v/s word index")
    plt.plot(list(map(lambda x: x[1], cnt))[500:])  # don't use the UNK token
    plt.show()


if __name__ == '__main__':
    tf.app.run(main)

"""
    Notes: 
    The Word frequency can be trimmed in to the range of 4000 - 40. This is a good measure to use the 
    Words in the values.
    
    This should produce a vocabulary of around 20000 words
"""