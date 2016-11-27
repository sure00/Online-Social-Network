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

summaryFile = 'summary.txt'
twitterFile = 'tweets.pkl'
userFile='user.pkl'
classifyDataFile = 'classify.pkl'

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

    print("Load %d  data "%len(data))
    return data

# append to database
def saveData(data,filePath):
    with open(filePath, "w+") as text_file:
        print("{}".format(data), file=text_file)


def main():
    summary={}
    summary['Number_users_collected']=len(loadData(userFile))
    summary['Number_messages_collected'] =len(loadData(twitterFile))
    #summary['Number_communities_discovered']=
    saveData(summary,summaryFile)


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