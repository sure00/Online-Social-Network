from collections import defaultdict
from io import  BytesIO
from zipfile import ZipFile
from urllib.request import  urlopen
import  configparser
from TwitterAPI import TwitterAPI
import re
from collections import Counter


url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
zipfile = ZipFile(BytesIO(url.read()))
afinn_file = zipfile.open('AFINN/AFINN-111.txt')

afinn = dict()

for line in afinn_file:
    parts=line.strip().split()
    if len(parts) == 2:
        afinn[parts[0].decode("utf-8")] = int(parts[1])
print('read %d AFINN terms.\nE.g.: %s' % (len(afinn), str(list(afinn.items())[:10])))
print(len(afinn))

def afinn_sentiment(terms, afinn):
    total=0
    for t in terms:
        if t in afinn:
            print('\t%s=%d' %(t, afinn[t]))
            total +=afinn[t]
    return total
doc = "i don't know if this is a scam or if mine was broken".split()
print('AFINN: ', afinn_sentiment(doc, afinn))

#What if mixed sentiment?
doc = "it has a hokey plot that is both to good and bad".split()
print('AFINN: ', afinn_sentiment(doc, afinn))

#Distinguish neutral from pos/neg.
#Return two scores per document
def afinn_sentiment2(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg
doc = "it has a hokey plot that is both to good and bad".split()
print('AFINN:', afinn_sentiment2(doc, afinn, verbose=True))

consumer_key = 'EqritIkuIQDEFtC8X0tHKCSAw'
consumer_secret = 'XoUbK9DXQpyI9X923fi2cGvQ1pABONUHXVKETmPCpMlc0aebcH'
access_token = '769354220537602048-lt18gDc963UdQGJinrfIYD8pwkmaiHT'
access_token_secret = 'ze4ACglFMf5dYfgdL3LUUepgBymJJbu3OCjjN5AvcZpFG'

searchKey='realDonaldTrum'
twitterFile = 'tweetsData.pkl'
userFile='user.pkl'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
twitter = get_twitter()
tweets = []
for r in twitter.request('search/tweets', {'q': 'mcdonalds', 'count': 100}):
    tweets.append(r)
print('read %d mcdonalds tweets' % len(tweets))

def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()

tokens = [tokenize(t['text']) for t in tweets]
print('tokenized, e.g., \n%s\nto\n%s' %
      (tweets[10]['text'], tokens[10]))

positives=[]
negatives=[]

for token_list, tweet in zip(tokens, tweets):
    pos, neg = afinn_sentiment2(token_list,afinn)
    if pos > neg:
        positives.append((tweet['text'], pos, neg))
    elif neg > pos:
        negatives.append((tweet['text'], pos, neg))

# Print top positives:
for tweet, pos, neg in sorted(positives, key=lambda x: x[1], reverse=True):
    print(pos, neg, tweet)

# Which words contribute most to sentiment?
all_counts=Counter()
for tweet in tokens:
    all_counts.update(tweet)
sorted_tokens=sorted(all_counts.items(), key=lambda  x:x[1], reverse=True)
i=0
for token, count in sorted_tokens:
    print('%s count=%d sentiment=%d' % (token, count, afinn[token]))
    i += 1
    if i > 10:
        break