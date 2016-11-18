

"""
collect.py
"""

import re
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import hashlib
import numpy as np
import csv
from datetime import *
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import pickle
from sklearn import neighbors
from zipfile import ZipFile
from sklearn import svm
from pylab import *
from scipy.sparse import csr_matrix
from TwitterAPI import TwitterAPI
from collections import Counter
import networkx as nx
import io
import sys, os
import time
import pickle

consumer_key = 'EqritIkuIQDEFtC8X0tHKCSAw'
consumer_secret = 'XoUbK9DXQpyI9X923fi2cGvQ1pABONUHXVKETmPCpMlc0aebcH'
access_token = '769354220537602048-lt18gDc963UdQGJinrfIYD8pwkmaiHT'
access_token_secret = 'ze4ACglFMf5dYfgdL3LUUepgBymJJbu3OCjjN5AvcZpFG'


def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def getData(twitter):
    #base on one hundreds
    limit=5
    tweets = []
    # Fetching tweets which talking about trump
    f= open('tweetsData.txt', 'wb+')

    for request in robust_request(twitter, 'statuses/filter', {'track': "Donald Trump"}):
    #for request in twitter.request('statuses/filter', {'track': "Donald Trump"}):
        #print(request.keys())
        tweets.append(request)
        if len(tweets) % 100 == 0:
            print('found %d tweets' % len(tweets))
        if len(tweets) >= 100*limit:
                break

    #print("before dump",tweets)
    pickle.dump(tweets,f)
    print(tweets[-1])
    f.close()

    #Debug log
    #f = open('tweetsData.txt', 'rb')
    #out = pickle.load(f)
    #print("after loading")
    #print(out[-1])
    #print(len(out))
    '''
    # Fetching tweets which talking about trump

    tweets = []
    totalTweets = 0

    for request in robust_request(twitter, 'statuses/filter', {'track': "Donald Trump"}):
        tweets.append(request)
        if len(tweets) % 100 == 0:
            saveData(tweets)
            tweets = []
            totalTweets+=100
            print('found %d tweets' % totalTweets)
        if totalTweets >= 100*limit:
                break
    #print(tweets[0])
    f = open('tweetsData.txt', 'rb')

    out = pickle.load(f)
    print("after loading")
    print(len(out))
    '''

if __name__ == '__main__':
    twitter = get_twitter()
    getData(twitter)