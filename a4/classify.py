"""
classify.py
"""
import configparser
from TwitterAPI import TwitterAPI
import sys
import requests
from collections import Counter
import re
from itertools import product
from collections import defaultdict
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import pickle


twittesFile = 'tweetsData.pkl'

def get_census_names():
    """ Fetch a list of common male/female names from the census.
    For ambiguous names, we select the more frequent gender."""
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])
    return male_names, female_names

# Read twitter which collected in collect.py
def get_twitter(filename):
    """ Load Twitters from files
    Returns:
      An instance of TwitterAPI.
    """
    if not os.path.isfile(filename):
        print("File %s do not exist, return derectly" %filename)
        return
    else:
        try:
            with open(filename, "rb") as file:
                unpickler = pickle.Unpickler(file)
                tweets = unpickler.load()
        except EOFError:
            return {}

    print(len(tweets))
    return tweets


def tokenize(string, lowercase, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):

    """ split a tweet into tokens. """
    if not string:
        return []
    if lowercase:
        string = string.lower()

    tokens=[]
    if collapse_urls:
        string = re.sub('http\S+', 'THIS_IS_A_URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'THIS_IS_A_MENTION', string)
    if keep_punctuation:
        tokens=string.split()
    else:
        tokens=re.sub('\W+', ' ', string).split()
    if prefix:
        tokens=['%s%s' % (prefix, t) for t in tokens]
    return tokens

def tweet2tokens(tweet, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):
    #print("tweet obj is ", tweet)
    tokens = tokenize(tweet['text'], lowercase, keep_punctuation, None,
                      collapse_urls, collapse_mentions)

    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'], lowercase,
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens

def make_vocabulary(tokens_list):
    vocabulary = defaultdict(lambda: len(vocabulary)) #if term not present, assign next it
    for tokens in tokens_list:
        for token in tokens:
            vocabulary[token]
    print("%d unique terms in vocabulary" %len(vocabulary))
    return vocabulary

def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()

def make_feature_matrix(tweets, tokens_list, vocabulary):
    X=lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j=vocabulary[token]
            X[i,j] += 1
    return X.tocsr()# convert to CSR for more efficient random access.

def get_gender(tweet, male_names, female_names):
    name= get_first_name(tweet)
    if name in female_names:
        return 1
    elif name in male_names:
        return 0
    else:
        return -1

def do_cross_val(X, y, nfolds):
    cv=KFold(len(y), nfolds)
    accuracies=[]

    for train_idx, test_idx in cv:
        clf=LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc=accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg=np.mean(accuracies)
    return avg

def main():
    male_names, female_names = get_census_names()
    tweets=get_twitter(twittesFile)

    #print("tweets is", tweets)

    test_tweet = tweets[1]
    print('test tweet:\n\tscreen_name=%s\n\tname=%s\n\tdescr=%s\n\ttext=%s' %
          (test_tweet['user']['screen_name'],
           test_tweet['user']['name'],
           test_tweet['user']['description'],
           test_tweet['text']))

    #print("tweets is", tweets)
    tokens_list = [tweet2tokens(t, use_descr=True, lowercase=True,
                                keep_punctuation=False, descr_prefix='d=',
                                collapse_urls=True, collapse_mentions=True)
                   for t in tweets]

    vocabulary = make_vocabulary(tokens_list)
    # store these in a sparse matrix
    X = make_feature_matrix(tweets, tokens_list, vocabulary)
    #print('shape of X:', X.shape)
    #print(X[10])

    y = np.array([get_gender(t, male_names, female_names) for t in tweets])
    print('gender labels:', Counter(y))

    print('avg accuracy', do_cross_val(X, y, 5))


if __name__ == '__main__':
    main()