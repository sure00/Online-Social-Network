"""
cluster.py
"""
import pickle
from TwitterAPI import TwitterAPI
import sys, os
import requests
from pprint import pprint
from collections import Counter
import re

def genderList():
    males_url = 'http://www2.census.gov/topics/genealogy/' + \
                '1990surnames/dist.male.first'
    females_url = 'http://www2.census.gov/topics/genealogy/' + \
                  '1990surnames/dist.female.first'

    males = requests.get(males_url).text.split('\n')
    females = requests.get(females_url).text.split('\n')

    males_pct = get_percents(males)
    females_pct = get_percents(females)

    male_names = set([m.split()[0].lower() for m in males if m])
    female_names = set([f.split()[0].lower() for f in females if f])

    male_names = set([m for m in male_names if m not in female_names or
                      males_pct[m] > females_pct[m]])
    female_names= set([f for f in female_names if f not in male_names or
                       females_pct[f]>males_pct[f]])





    #print('%d male and %d female names' % (len(male_names), len(female_names)))
    #print('males:\n' + '\n'.join(list(male_names)[:10]))
    #print('\nfemales:\n' + '\n'.join(list(female_names)[:10]))

    return male_names, female_names

def print_ambiguous_name(male_names, female_names):
    ambiguous = [n for n in male_names if n in female_names]
    print('found %d ambiguous names:\n'% len(ambiguous))
    print('\n'.join(ambiguous[:20]))

def get_percents(name_list):
    return dict([(n.split()[0].lower(), float(n.split()[1])) for n in name_list if n])


def print_genders(tweets):
    counts = Counter(t['gender'] for t in tweets)
    print('%.2f of accounts are labeled with gender' %
          ((counts['male'] + counts['female']) / sum(counts.values())))
    print('gender counts:\n', counts)
    for t in tweets[:20]:
        print(t['gender'], t['user']['name'])



def gender_by_name(tweets):
    #get male and female name from Census
    male_names, female_names = genderList()

    for t in tweets:
        name = t['user']['name']
        t['gender'] = 'unknown'
        if name:
            #remove punctuation
            name_parts=re.findall('\w+',name.split()[0].lower())
            #print("name_parts is", name_parts)
            if len(name_parts) > 0:
                firstName = name_parts[0].lower()
                if firstName in male_names:
                    t['gender']='male'
                    #print("first name %s is male" %firstName)
                elif firstName in female_names:
                    t['gender']='female'
                    #print("first name %s is female" % firstName)
                else:
                    t['gender']='unknown'
                    #print("first name %s is unknow" % firstName)

    print_ambiguous_name(male_names, female_names)

    #print("tweets is", tweets)
    #for t in tweets:
    #    print("final first name %s , gender is %s" %(t['user']['name'],t['gender']))
    #    #print("final first name %s " %(t['user']['name']))

def loadData(filename):
    # The protocol version used is detected automatically, so we do not
    # have to specify it.

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

def counts_to_probs(gender_words):
    '''
    Compute probability of each term according to the frequency in a gender
    :param gender_words:
    :return:
    '''
    total = sum(gender_words.values())
    return dict([(word, count/total)
                 for word, count in gender_words.items()])

def tokenize(s):
    return re.sub('\W+', ' ', s).lower().split() if s else []

def odds_ratios(male_probs, female_probs):
    return dict([(w, female_probs[w]/male_probs[w])
                 for w in
                 set(male_probs.keys()) | set(female_probs.keys())
                 ])

if __name__ == '__main__':
    filename = 'tweetsData.txt'
    tweets = loadData(filename)
    gender_by_name(tweets)
    #print_genders(tweets)

    #male_profiles = [t['user']['description'] for t in tweets
                     #if t['gender'] == 'male']
    #female_profiles = [t['user']['description'] for t in tweets
                       #if t['gender'] == 'female']

    male_profiles = [t['text'] for t in tweets
    if t['gender'] == 'male']
    female_profiles = [t['text'] for t in tweets
    if t['gender'] == 'female']

    print("male profiles is", male_profiles)
    print("female profiles is", female_profiles)

    male_words = Counter()
    female_words = Counter()

    for p in male_profiles:
        male_words.update(Counter(tokenize(p)))

    for p in female_profiles:
        female_words.update(Counter(tokenize(p)))

    diff_counts = dict([(w, female_words[w] - male_words[w])
                        for w in
                        set(female_words.keys() | set(male_words.keys()))
                        ])
    sorted_diffs = sorted(diff_counts.items(), key=lambda  x:x[1])

    male_probs = counts_to_probs(male_words)
    female_probs = counts_to_probs(female_words)

    print('p(w|male)')
    pprint(sorted(male_probs.items(), key=lambda x: -x[1])[:10])

    print('\np(w|female)')
    pprint(sorted(female_probs.items(), key=lambda x: -x[1])[:10])

    #additive smoothing, Add count of 1 for all workds
    all_words=set(male_words) | set(female_words)
    male_words.update(all_words)
    female_words.update(all_words)

    male_probs = counts_to_probs(male_words)
    female_probs = counts_to_probs(female_words)

    ors = odds_ratios(male_probs, female_probs)

    sorted_ors = sorted(ors.items(), key=lambda x: -x[1])

    print('Top Female Terms (OR):')
    pprint(sorted_ors[:20])

    print('\nTop Male Terms (OR):')
    pprint(sorted_ors[-20:])