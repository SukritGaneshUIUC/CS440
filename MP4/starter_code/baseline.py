"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

import queue

def getHighestValueKey(theDict):
    maxVal = -1
    maxValKey = ''
    for key in theDict:
        if (theDict[key] > maxVal):
            maxVal = theDict[key]
            maxValKey = key

    return maxValKey

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # very simple - dict of dicts
    # dict within dict: frequency for each word
    # also keep of most common overall tags

    frequencyDict = {}
    allTags = {}

    for sentence in train:
        for word_tag in sentence:
            word = word_tag[0]
            tag = word_tag[1]

            # check if word in frequencyDict, retrieve that word's tag dict
            if (word not in frequencyDict):
                currDict = {}
            else:
                currDict = frequencyDict[word]

            # add tag to currentDict (the current word's tag dict)
            if (tag not in currDict):
                currDict[tag] = 1
            else:
                currDict[tag] += 1

            # update the current word's tag dict
            frequencyDict[word] = currDict

            # also make sure to update allTags (keeping track of frequency of all tags)
            if (tag not in allTags):
                allTags[tag] = 1
            else:
                allTags[tag] += 1

    # get the most common tag
    mostCommonTag = getHighestValueKey(allTags)

    # for each word in test set, find most common tag
    tagLabels = []
    for sentence in test:
        currSentenceLabels = []
        for word in sentence:
            # find maximum tag
            if (word not in frequencyDict):
                currSentenceLabels.append((word, mostCommonTag))
            else:
                currSentenceLabels.append((word, getHighestValueKey(frequencyDict[word])))

        tagLabels.append(currSentenceLabels)

    return tagLabels
