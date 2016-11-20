import configparser
from TwitterAPI import TwitterAPI
import sys
import requests
from collections import Counter
import re

consumer_key = 'EqritIkuIQDEFtC8X0tHKCSAw'
consumer_secret = 'XoUbK9DXQpyI9X923fi2cGvQ1pABONUHXVKETmPCpMlc0aebcH'
access_token = '769354220537602048-lt18gDc963UdQGJinrfIYD8pwkmaiHT'
access_token_secret = 'ze4ACglFMf5dYfgdL3LUUepgBymJJbu3OCjjN5AvcZpFG'

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

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()

def sample_tweets(twitter, limit, male_names, female_names):
    tweets=[]
    while Treu:
        try:
            # restrict to U.s
            for response in twitter.request('statuses/filter',
                        {'locations':'-124.637,24.548,-66.993,48.9974'}):
                if 'user' in response:
                    name = get_first_name(response)
                    if name in male_names or name in female_names:
                        tweets.append(response)
                        if len(tweets) % 100 == 0:
                            print('found %d tweets' % len(tweets))
                        if len(tweets) >= limit:
                            return tweets
        except:
            print("Unexpected error:", sys.exc_info()[0])
    return tweets

def tokenize(string, lowercase, keep_punctuation, prefix,
                collapse_urls, collapse_mentions):
    """ split a tweet into tokens. """
    if not strings:
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


if __name__ == '__main__':
    twitter = get_twitter()
    male_names, female_names = get_census_names()
    tweets = sample_tweets(twitter, 5000, male_names, female_names)
    test_tweet = tweets[200]
    print('test tweet:\n\tscreen_name=%s\n\tname=%s\n\tdescr=%s\n\ttext=%s' %
          (test_tweet['user']['screen_name'],
           test_tweet['user']['name'],
           test_tweet['user']['description'],
           test_tweet['text']))
    tokenize(test_tweet['user']['description'], lowercase=True,
         keep_punctuation=True, prefix='d=',
         collapse_urls=True, collapse_mentions=True)


