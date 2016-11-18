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
from urllib.parse  import urlparse

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
    request = robust_request(twitter,'search/tweets', {'q': "Prop64", 'count': 5000})



    #tweets = twitter.request('search/tweets', {'q': "Prop64", 'count': 5000})
    print(request.get_rest_quota())
    obj = request.json()
    print("obj is",obj)

    k = 1
    while (len(request.text) > 0):
            obj = request.json();
            #print("obj is", obj)
            refresh_url = obj["search_metadata"]["refresh_url"]
            max_id_str = obj["search_metadata"]["max_id_str"]
            since_id_str = obj["search_metadata"]["since_id_str"]
            query = urlparse(refresh_url).query.split("&");
            print("since_id is",query[0].split("=")[1],query[1].split("=")[1])
            lastKey = query[0].split("=")[1]
            print("lastKey Added ", lastKey)
            #appendData(obj['statuses'])
            print(obj['statuses'])
            print("no of tweets", len(obj['statuses']))
            if len(obj['statuses']) == 0:
                #print("sleeping ofr 15 minutes at", str(datetime.now()))
                time.sleep(61 * 15)
            else:
                tweets = twitter.request('search/tweets', {'q': query[1].split("=")[1], 'include_entities': 1,
                                                       'since_id': query[0].split("=")[1], 'count': 5000})
            k = k + 1
            if k >= 15:
                break;

if __name__ == '__main__':
    twitter = get_twitter()
    getData(twitter)