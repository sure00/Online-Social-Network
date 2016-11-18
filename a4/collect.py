

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

# append to database
def saveData(tweets):
    """ save the collect data to tweetsData.txt.
    Args:
      twitters .... Collect data from twitter.
    Returns:
      NULL
    """

    f = open('tweetsData.txt', 'wb+')
    tweets = [t for t in tweets if 'user' in t]
    print('fetched %d tweets' % len(tweets))

    pickle.dump(tweets, f)
    f.close()
    print("Data had saved to tweetsData.txt")

def getData(twitter,limit):
    """ Get the twitter data with stream API.
    Args:
      twitter .... A TwitterAPI object.
      limit ... Total twitters that collected
    Returns:
      A TwitterResponse array which collect all the twitters
    """

    #base on one hundreds
    tweets = []

    # Fetching tweets with stream api
    for request in robust_request(twitter, 'statuses/filter', {'track': "Donald Trump"}):
        tweets.append(request)
        if len(tweets) % 100 == 0:
            print('found %d tweets' % len(tweets))
        if len(tweets) >= 100*limit:
                return  tweets

if __name__ == '__main__':
    twitter = get_twitter()
    twitters = getData(twitter, limit=10)
    saveData(twitters)