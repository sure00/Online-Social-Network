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
from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle
from io import BytesIO, StringIO
from zipfile import ZipFile
import urllib.request
import pandas as pd

twittesFile = 'tweets.pkl'
classifyDataFile = 'classify.pkl'

def getSentimentData():
    # The file is 78M, so this will take a while.
    url = urllib.request.urlopen('http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    tweet_file = zipfile.open('testdata.manual.2009.06.14.csv')
    return tweet_file


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

    return tweets


def tokenize(s,  keep_punctuation, prefix):

    """ split a tweet into tokens. """
    if not s:
        return []

    s = s.lower()

    tokens=[]
    s = re.sub('http\S+', 'THIS_IS_A_URL', s)
    s = re.sub('@\S+', 'THIS_IS_A_MENTION', s)
    if keep_punctuation:
        tokens=s.split()
    else:
        tokens=re.sub('\W+', ' ', s).split()
    if prefix:
        tokens=['%s%s' % (prefix, t) for t in tokens]
    return tokens

def tweet2tokens(tweet, csv, use_descr=True, lowercase=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):
    #print("tweet obj is ", tweet)
    if csv == True:
        text = tweet[5]
    else:
        text =tweet['text']
    tokens = tokenize(text,  keep_punctuation, None,
                      )

    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'],
                               keep_punctuation, descr_prefix,
                               ))
    return tokens

def makeVocab(tokensList):
    vocab = defaultdict(lambda: len(vocab)) #if term not present, assign next it
    for tokens in tokensList:
        for t in tokens:
            vocab[t]
    #print("%d unique terms in vocabulary" %len(vocab))
    return vocab

def getFirstName(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        p = tweet['user']['name'].split()
        if len(p) > 0:
            return p[0].lower()

def makeFeatureMatrix(tweets, tokens_list, vocabulary):
    X=lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j=vocabulary[token]
            X[i,j] += 1
    return X.tocsr()# convert to CSR for more efficient random access.

def getGender(tweet, maleNames, femaleNames):
    name= getFirstName(tweet)
    if name in femaleNames:
        return 1
    elif name in maleNames:
        return 0
    else:
        return -1

def doCrossVal(X, y, nfolds):
    cv=KFold(len(y), nfolds)
    acc=[]

    for train_idx, test_idx in cv:
        clf=LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        tmp=accuracy_score(y[test_idx], predicted)
        acc.append(tmp)
    avg=np.mean(acc)
    return avg


def getLogisClf(C = 1.,penalty = 'l2'):
    return LogisticRegression(C = C,penalty=penalty,random_state=42)

def print_results(data):
        print("\tTweets  aganist Trump\t\t%d" %data[0])
        print("\tNeutral  tweets on Trump\t\t%d" %data[2])
        print("\tSupport  Trump tweets\t\t\t%d" %data[4])

def classifyTweets(predictes):
    Againist=[]
    Neutral=[]
    Support=[]

    for i in range(len(predictes)):
        if predictes[i]==0:
            Againist.append(i)
        elif predictes[i]==2:
            Neutral.append(i)
        elif predictes[i]==4:
            Support.append(i)

    return Againist, Neutral, Support

# append to database
def saveData(data,file):
           """ save the collect data to tweetsData.txt.
           Args:
             twitters .... Collect data from twitter.
           Returns:
             NULL
           """
           f = open(file, 'wb+')
           #tweets = [t for t in tweets if 'user' in t]
           #print('fetched %d tweets' % len(tweets))
           pickle.dump(data, f)
           f.close()
           print("data %s saved successfully" %file)

def main():
    savedData={}
    #Gets tweets
    testTweets = get_twitter(twittesFile)
    if testTweets =={}:
        return

    #Check Sentiment of twittes
    trainingFile = getSentimentData()
    trainingTweets = pd.read_csv(trainingFile,header=None,names=['polarity', 'id', 'date',
                                                       'query', 'user', 'text'])

    # Feturerize
    TrainingLabel = np.array(trainingTweets['polarity'])
    print("TrainingLabel", len(TrainingLabel))

    #Training vectorizer
    training_tokens_list = [tweet2tokens(t, csv=True, use_descr=False, lowercase=True,
                                keep_punctuation=False, descr_prefix='',
                                collapse_urls=True, collapse_mentions=True )
                   for t in trainingTweets.values]
    vocabulary = makeVocab(training_tokens_list)
    print("len vocab is",len(vocabulary))

    #Get feature of text
    test_tokens_list = [tweet2tokens(t,csv=False, use_descr=True, lowercase=True,
                                keep_punctuation=False, descr_prefix='',
                                collapse_urls=True, collapse_mentions=True)
                   for t in testTweets]
    #print("tokens_list is", tokens_list)
    test_vocabulary = makeVocab(test_tokens_list)

    # merge two vocabulay together
    vocabulary.update(test_vocabulary)

    TrainX = makeFeatureMatrix(trainingTweets, training_tokens_list, vocabulary)
    TestX = makeFeatureMatrix(testTweets, test_tokens_list, vocabulary)

    #Fit A logistrcRegression model
    clf_logistic1 = getLogisClf(C=1.0)
    clf_logistic1.fit(TrainX, TrainingLabel)

    # Classify Data
    predictes = clf_logistic1.predict(TestX)
    Againist_idx, Neutral_idx, Support_idx = classifyTweets(predictes)
    print_results(Counter(predictes))

    savedData['Againist_idx']= Againist_idx
    savedData['Neutral_idx'] = Neutral_idx
    savedData['Support_idx'] = Support_idx

    male_names, female_names = get_census_names()

    #print("tweets is", tweets)
    tokens_list = [tweet2tokens(t,csv=False, use_descr=True, lowercase=True,
                                keep_punctuation=False, descr_prefix='d=',
                                collapse_urls=True, collapse_mentions=True)
                   for t in testTweets]

    vocabulary = makeVocab(tokens_list)
    # store these in a sparse matrix
    X = makeFeatureMatrix(testTweets, tokens_list, vocabulary)

    y = np.array([getGender(t, male_names, female_names) for t in testTweets])
    #print('gender labels:', Counter(y))

    clf=LogisticRegression()
    clf.fit(X, y)

    predicted = clf.predict(X[Againist_idx])
    #print("Againist Trump is", Counter(predicted))
    acc = accuracy_score(y[Againist_idx], predicted)
    print("Againist Gender acc is", acc)
    savedData['AgainistGender']=predicted

    predicted = clf.predict(X[Neutral_idx])
    #print("Neutral Trump is", Counter(predicted))
    acc = accuracy_score(y[Neutral_idx], predicted)
    print("Neutral Gender acc is", acc)
    savedData['NeutralGender'] = predicted

    predicted = clf.predict(X[Support_idx])
    #print("Support Trump is", Counter(predicted))
    acc = accuracy_score(y[Support_idx], predicted)
    print("Support Gender acc is", acc)
    savedData['SupportGender'] = predicted

    saveData(savedData, classifyDataFile)



if __name__ == '__main__':
    main()