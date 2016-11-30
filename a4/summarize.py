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

    summary.append('\t\nInstance of Aganist Trump Tweets: ' + str(tweets[(classify['Againist_idx'])[0]]['text']))
    summary.append('\t\nInstance of Neutral Trump Tweets: ' + str(tweets[(classify['Neutral_idx'])[0]]['text']))
    summary.append('\t\nInstance of Support Trump Tweets: ' + str(tweets[(classify['Support_idx'])[0]]['text']))

    #print([(classify['Againist_idx'])[1]])
    #print([(classify['Neutral_idx'])[1]])
    #print([(classify['Support_idx'])[1]])

    SupportGender = Counter(classify['SupportGender'])
    NeutralGender= Counter(classify['NeutralGender'])
    AgainistGender= Counter(classify['AgainistGender'])

    summary.append('\n\nThere are ' + str(SupportGender[1]/sum(SupportGender.values())) + '% femal Support Trump and '
                   + str(SupportGender[0]/sum(SupportGender.values())) + '% male')

    #print("Againist_idx",classify['Againist_idx'])
    getFemal =getMale=0
    for index in range(len(classify['SupportGender'])):
        if getFemal == 1 and getMale ==1:
            #print(tweets[classify['Againist_idx'][femaleNamesID]]['user']['name'])
            #print(tweets[classify['Againist_idx'][maleNameId]]['user']['name'])

            summary.append('In Support Trump tweets, one female name is :' +
                           str(tweets[classify['Againist_idx'][femaleNamesID]]['user']['name']) + '\t male name :'
                           + str(tweets[classify['Againist_idx'][maleNameId]]['user']['name']))
            break
        if getFemal ==0 and classify['SupportGender'][index] == 1:
            femaleNamesID= index
            getFemal=1
        if getMale ==0 and classify['SupportGender'][index] == 0:
            maleNameId=index
            getMale =1


    summary.append('\t\nThere are ' + str(NeutralGender[1]/sum(NeutralGender.values())) + '% femal neutral Trump and '
                   + str(NeutralGender[0]/sum(NeutralGender.values())) + '% male')

    #print("Againist_idx",classify['Againist_idx'])
    getFemal =getMale=0
    for index in range(len(classify['NeutralGender'])):
        if getFemal == 1 and getMale ==1:
            #print(tweets[classify['Neutral_idx'][femaleNamesID]]['user']['name'])
            #print(tweets[classify['Neutral_idx'][maleNameId]]['user']['name'])

            summary.append('In Neutral Trump tweets, one female name is :' +
                           str(tweets[classify['Neutral_idx'][femaleNamesID]]['user']['name']) + '\t male name :'
                           + str(tweets[classify['Neutral_idx'][maleNameId]]['user']['name']))
            break
        if getFemal ==0 and classify['NeutralGender'][index] == 1:
            femaleNamesID= index
            getFemal=1
        if getMale ==0 and classify['NeutralGender'][index] == 0:
            maleNameId=index
            getMale =1

    summary.append('\t\nThere are  ' + str(AgainistGender[1]/sum(AgainistGender.values())) + '% femal againist Trump and'
                   + str(AgainistGender[0]/sum(AgainistGender.values())) + '% male')

    getFemal =getMale=0
    for index in range(len(classify['AgainistGender'])):
        if getFemal == 1 and getMale ==1:
            #print(tweets[classify['Againist_idx'][femaleNamesID]]['user']['name'])
            #print(tweets[classify['Againist_idx'][maleNameId]]['user']['name'])

            summary.append('In Againist Trump tweets, one female name is :' +
                           str(tweets[classify['Againist_idx'][femaleNamesID]]['user']['name']) + '\t male name :'
                           + str(tweets[classify['Againist_idx'][maleNameId]]['user']['name']))
            break
        if getFemal ==0 and classify['AgainistGender'][index] == 1:
            femaleNamesID= index
            getFemal=1
        if getMale ==0 and classify['AgainistGender'][index] == 0:
            maleNameId=index
            getMale =1

    saveData(summary,summaryFile)
    print("Summarize Done")


if __name__ == '__main__':
    main()
