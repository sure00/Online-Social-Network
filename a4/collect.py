

"""
collect.py
"""

import datetime
import pickle
from TwitterAPI import TwitterAPI
import io
import sys, os
import time

consumer_key = 'EqritIkuIQDEFtC8X0tHKCSAw'
consumer_secret = 'XoUbK9DXQpyI9X923fi2cGvQ1pABONUHXVKETmPCpMlc0aebcH'
access_token = '769354220537602048-lt18gDc963UdQGJinrfIYD8pwkmaiHT'
access_token_secret = 'ze4ACglFMf5dYfgdL3LUUepgBymJJbu3OCjjN5AvcZpFG'

searchKey='Trump'
savedFile = 'tweetsData.pkl'

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
    f = open(savedFile, 'wb+')
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
    for request in robust_request(twitter, 'statuses/filter', {'track': searchKey}):
        tweets.append(request)
        #if len(tweets) % 100 == 0:
        if len(tweets) % 30 == 0:
            print('found %d tweets' % len(tweets))
        #if len(tweets) >= 100*limit:
        if len(tweets) >= 30 * limit:
                return  tweets

# using  to filtering all the friends who have Twitte about trump.
def filterFriends(twitter, friends):
    """ Return a list of Twitter IDs for friends who is also care Trump.
    Args:
        twitter.......The TwitterAPI object
        friendsID... Friends IDS
    Returns:
        A list of friends ID whose Twitt contains Trump.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, friends)[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """

    #request = robust_request(twitter, 'users/lookup', {'user_id': friends})
    #print("request is", request)


    filtedList=[]
    for f in friends:
        #print("friend id is", f)
        #check the account status
        request = robust_request(twitter, 'users/show', {'user_id': f})
        userInfo = [r for r in request]

        # If the accoutn is protected, that means cannot access it. Skip it.
        #print(userInfo[0]['protected'])
        if userInfo[0]['protected'] == True:
            continue
        #print("person info is ", info)

        request = robust_request(twitter, 'statuses/user_timeline', {'user_id': f})

        #check timeline conclude Trump.
        fInfo = [r for r in request if r['text'] and searchKey in r['text']]
        if fInfo != []:
            filtedList.append(f)
            #print(fInfo)

    print("After Filter, there are %d friends have tweet contain Trump", len(filtedList))

    return filtedList


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.
    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    print("screen name is", screen_name)
    respond  = robust_request(twitter, 'friends/ids', {'screen_name': screen_name}, 5 )
    friends  = [r for r in respond]

    filtedFriends=filterFriends(twitter, friends)

    filtedFriends.sort()
    #print(friends)
    return (filtedFriends)

def expandNetWork(twitter, twitters):
    '''Base on the twitter data to expand the network for using detect community algorithm to find the community.
    Args:
        twitters... A list of tweets

    Return:
         A new list of tweets which can construct a network
    '''
    for t in twitters:
        t['user']["friends"] = get_friends(twitter, t['user']["screen_name"])


if __name__ == '__main__':
    twitter = get_twitter()
    twitters = getData(twitter, limit=1)
    #print("Changing from streaming request to REST at %s " %(str(datetime.datetime.now())))
    #time.sleep(61 * 15)
    expandNetWork(twitter, twitters)
    saveData(twitters)