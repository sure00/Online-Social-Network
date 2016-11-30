"""
sumarize.py

Number of users collected:
Number of messages collected:
Number of communities discovered:
Average number of users per community:
Number of instances per class found:
One example from each class:
"""
import pickle
import os
from collections import Counter

summaryFile = 'summary.txt'
twitterFile = 'tweets.pkl'
userFile='user.pkl'
classifyDataFile = 'classify.pkl'
clusterFile =  'cluster.pkl'

def loadData(filename):
    """ Load twittes which collected in collect period

    Returns:
    return twittes
    """
    if not os.path.isfile(filename):
        print("File %s do not exist, return derectly" %filename)
        return
    else:
        try:
            with open(filename, "rb") as file:
                unpickler = pickle.Unpickler(file)
                data = unpickler.load()
        except EOFError:
            return {}

    #print("Load %d  data "%len(data))
    return data

# append to database
def saveData(data,filePath):
    with open(filePath, "w+",encoding='utf-8') as f:
        for item in data:
                f.write(item)
                f.write('\n')

def main():
    summary=[]

    userInfo = loadData(userFile)
    #print(userInfo)
    total = 0
    for u in userInfo.values():
        if u == []:
            continue
        #print(u)
        #print(len(u))
        total += len(u)

    summary.append('Number of users collected: '+ str(total))
    tweets = loadData(twitterFile)
    summary.append('Number of messages collected: ' + str(len(tweets)))

    components = loadData(clusterFile)
    summary.append('\nNumber of communities discovered: ' + str(len(components)))

    total=0
    for comp in components:
        total+=len(comp)

    summary.append('Average number of users per community:' + str(total/len(components)))

    classify = loadData(classifyDataFile)
   # print(classify)
    #print(classify['Sentiment'])
    summary.append('\nNumber of instances per class found: ')
    summary.append('\tAganist Trump Tweets: ' + str(len(classify['Againist_idx'])))
    summary.append('\tNeutral Trump Tweets: ' + str(len(classify['Neutral_idx'])))
    summary.append('\tSupport Trump Tweets: ' + str(len(classify['Support_idx'])))

    summary.append('\nNumber of instances per class found:  ')
    #print(tweets[2])
    #print(classify['Againist_idx'][2])

    #print(tweets[2]['text'])

    summary.append('\tInstance of Aganist Trump Tweets: ' + str(tweets[(classify['Againist_idx'])[-1]]['text']))
    summary.append('\tInstance of Neutral Trump Tweets: ' + str(tweets[(classify['Neutral_idx'])[-1]]['text']))
    summary.append('\tInstance of Support Trump Tweets: ' + str(tweets[(classify['Support_idx'])[-1]]['text']))

    #print([(classify['Againist_idx'])[1]])
    #print([(classify['Neutral_idx'])[1]])
    #print([(classify['Support_idx'])[1]])

    SupportGender = Counter(classify['SupportGender'])
    #print(SupportGender)
    NeutralGender= Counter(classify['NeutralGender'])
    AgainistGender= Counter(classify['AgainistGender'])

    summary.append('\n\tFor the tweets that Support Trump there are ' + str(SupportGender[1]/sum(SupportGender.values())) + '% femal '
                   + str(SupportGender[0]/sum(SupportGender.values())) + '% male')

    summary.append('\tFor the tweets that Neutral Trump there are ' + str(NeutralGender[1]/sum(NeutralGender.values())) + '% femal '
                   + str(NeutralGender[0]/sum(NeutralGender.values())) + '% male')

    summary.append('\tFor the tweets that Againist Trump there are ' + str(AgainistGender[1]/sum(AgainistGender.values())) + '% femal '
                   + str(AgainistGender[0]/sum(AgainistGender.values())) + '% male')

    saveData(summary,summaryFile)
    print("Summarize Done")


if __name__ == '__main__':
    main()
"""
Number of users collected:
Number of messages collected:
Number of communities discovered:
Average number of users per community:
Number of instances per class found:
One example from each class:
"""