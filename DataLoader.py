import numpy as np
import h5py
import os, os.path
import re
import pandas as pd

from sklearn.datasets import make_blobs
from PIL import Image
from collections import Counter





def gen_cluster(n, m, k):
    '''
    Generate a cluster of n samples with m features and k clusters
    '''

    x, y = make_blobs(n_samples=n, centers=k, n_features=m)

    return x, y

def file_count(DIR):
    '''
    Count the number of files in a directory
    '''

    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


def h5_loader(DIR):
    '''
    Load the data in the h5 file, convert to numpy arrays and return the data
    '''

    with h5py.File(DIR, 'r') as hf:
        train = hf.get('train')
        x_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        x_te = test.get('data')[:]
        y_te = test.get('target')[:]

    A = np.concatenate((np.array(x_tr),np.array(x_te)))
    b = np.concatenate((np.array(y_tr),np.array(y_te)))
    return A, b

def gray_img_np(DIR, ratio):
    '''
    Load the image and convert to a numpy array in black and white (1 color channel) and resize
    '''

    img1 = Image.open(DIR)
    gray_image=img1.convert('L').reduce(ratio)
    return np.asarray(gray_image)

def img_loader(DIR, ratio):
    '''
    Load the image and convert to a numpy array iusing the gray_img_np function
    Differ between the ORL and COIL datasets
    '''
    names = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    x = gray_img_np(DIR + '/' + names[0], ratio).flatten()

    for i in range(1, len(names)):
        # Load imaged based on file name and flatten them
        file_path = DIR + '/' + names[i]
        x = np.vstack((x,  gray_img_np(file_path, ratio).flatten()))


    # Classify the images based on the file name
    if DIR == 'ORL':
        y = ORL_classifier(names[0])
        for i in range(1, len(names)):
            classifier  = ORL_classifier(names[i])
            y = np.vstack((y, classifier))
    else:
        y = coil_classifier(names[0])
        for i in range(1, len(names)):
            classifier = coil_classifier(names[i])
            y = np.vstack((y, classifier))
    return x, y

def coil_classifier(file_name):
    '''
    Classify the COIL images based on the file name
    '''

    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    res = temp.match(file_name).groups()
    return np.array([int(res[1])])

def ORL_classifier(file_name):
    '''
    Classify the ORL images based on the file name
    '''
    return np.array([int(file_name.split('_')[1].split('.')[0])])


def text_loader(DIR, title):
    '''
    Load the text file and convert to a numpy array
    '''

    df = df_loader(DIR)
    vocab = df_to_word_list(df, title)

    # Convert the text to a numpy array
    sentence_list = df_to_sentence_list(df, title)
    vocab_to_int_long = word_to_int_vocab(vocab)
    sentence_int = remove_zero_len(vocab_to_int_long, sentence_list)

    # Remove the sentences with length 0
    non_zero_idx = [ii for ii, review in enumerate(sentence_int) if len(review) != 0]
    headlines_ints = [sentence_int[ii] for ii in non_zero_idx]

    seq_length = len(max(sentence_list, key=len))
    features = pad_features(headlines_ints, seq_length)

    return features

def lable_loader(DIR, title):
    '''
    Load the labels from the text file and convert to a coded numpy array
    '''
    df = df_loader(DIR)
    vocab = df_to_word_list(df, title)

    word_list = df_to_sentence_list(df, title)
    vocab_to_int_long = word_to_int_vocab(vocab)
    word_int = remove_zero_len(vocab_to_int_long, word_list)

    non_zero_idx = [ii for ii, word in enumerate(word_int) if len(word) != 0]
    word_ints = [word_int[ii] for ii in non_zero_idx]

    return np.array(word_ints) - np.ones(np.array(word_ints).shape)

def df_loader(DIR):
    '''
    Load the dataframe from the csv file
    '''
    text = pd.read_csv(DIR)
    df = pd.DataFrame(text)

    return df


def df_to_word_list(df, title):
    '''
    Find the vocabulary from the dataframe
    '''
    title_df = df[title].tolist()[:-1]
    words_pattern = '[a-z]+'
    words =''
    for i in range(len(title_df)):
        words += ' ' + title_df[i]

    vocab = re.findall(words_pattern, words, flags=re.IGNORECASE)

    return vocab


def df_to_sentence_list(df, title):
    '''
    Find the sentences from the dataframe
    '''
    title_df = df[title].tolist()[:-1]
    words_pattern = '[a-z]+'
    sentences = []

    for i in range(len(title_df)):
        sentence = re.findall(words_pattern, title_df[i], flags=re.IGNORECASE)
        sentences.append(' '.join(sentence))

    return sentences


def word_to_int_vocab(vocab):
    '''
    Encode the vocabulary into integers
    '''
    counts = Counter(vocab)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    return vocab_to_int


def remove_zero_len(vocab, sentences):
    '''
    Remove the sentences with length 0
    '''
    sentences_ints = []

    for line in range(len(sentences)):
        sentences_ints.append([vocab[word] for word in sentences[line].split()])

    return sentences_ints


def pad_features(text_ints, seq_length):
    '''
    Pad the features in the dataframe to the same length
    '''

    features = np.zeros((len(text_ints), seq_length), dtype=int)

    for i, row in enumerate(text_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features
