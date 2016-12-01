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

searchKey='@realDonaldTrump'
twitterFile = 'tweets.pkl'
userFile='user.pkl'

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
def saveData(tweets,file):
           """ save the collect data to tweetsData.txt.
           Args:
             twitters .... Collect data from twitter.
           Returns:
             NULL
           """
           f = open(file, 'wb+')
           #tweets = [t for t in tweets if 'user' in t]
           #print('fetched %d tweets' % len(tweets))
           pickle.dump(tweets, f)
           f.close()
           print("Twittes File %s saved successfully" %file)

def getTwittesData(twitter,limit):
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
    #tweets = api.
    while(True):
        for request in robust_request(twitter,'search/tweets', {'q': searchKey, 'count': 4000}):
        #for request in robust_request(twitter, 'statuses/filter', {'track':searchKey}):
            #request = request.json()
            #print(request)
            #if  searchKey not in request['text']:
                #print("sure debug %s" %request['text'])
                #continue
            tweets.append(request)
            #if len(tweets) % 5 == 0:
            if len(tweets) % 100 == 0:
                print('found %d tweets talk about %s' %(len(tweets),searchKey))
            #if len(tweets) >= 5*limit:
            if len(tweets) >= 100 * limit:
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
    friendsTwitts=[]
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
            #print("friend ID %s have %d tweets that mentioned %s" %(f,len(fInfo), searchKey))
            filtedList.append(f)
            friendsTwitts += fInfo

    return filtedList, friendsTwitts


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
    #print("Twitter user screen name is", screen_name)
    respond  = robust_request(twitter, 'friends/ids', {'screen_name': screen_name}, 5 )
    friends  = [r for r in respond][:100]

    filtedFriends, friendsTwitts =filterFriends(twitter, friends)
    print(" %s have %d friends have tweets contain Trump"%(screen_name,len(filtedFriends)))

    filtedFriends.sort()
    #print(friends)
    return filtedFriends,friendsTwitts

def get_friends(twitter, user_id):

    respond  = robust_request(twitter, 'friends/ids', {'user_id': user_id}, 5 )
    friends  = [r for r in respond][:200]

    friends.sort()
    print("User %d have %d friends" %(user_id, len(friends)))
    return (friends)

def getUserData(twitter, twitters):
    user={}
    for t in twitters:
        #user.setdefault(t['user']["id"], []).append(get_friends(twitter,  t['user']["id"]))
        user[t['user']["id"]] = get_friends(twitter, t['user']["id"])

    return user


def findFriendstweets(twitter, twitters):
    '''Base on the twitter data to expand the network for using detect community algorithm to find the community.
    Args:
        twitters... A list of tweets
    Return:
         A new list of tweets which can construct a network
    '''
    friendsTwitts=[]
    for t in twitters:
        t['user']["friends"], friendsTwitts = get_friends(twitter, t['user']["screen_name"])
        #print("user %s tweets is%s" %(t['user']["screen_name"], t))
        #print("friends ", friendsTwitts)
    friendsTwitts.extend(friendsTwitts)
    return friendsTwitts

if __name__ == '__main__':
    print("start collecting")
    twitter = get_twitter()
    print("start to collect twittes")

    #collect twittes data
    twitters = getTwittesData(twitter, limit=15)
    #print(twitters)
    saveData(twitters, twitterFile)

    #get user data
    #print("twitters is ",twitters)
    user = getUserData(twitter, twitters[:5])
    #print("user is", user)
    saveData(user, userFile)
