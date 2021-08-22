# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import string
import math
import random
import copy
import nltk
from nltk.corpus import stopwords

DEFAULT_STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

UNKNOWN = '28372837287382'

def get_random_string(length):
    letters = string.printable
    return ''.join(random.choice(letters) for i in range(length))

def removeStopWords(theSet, theStopWords=DEFAULT_STOP_WORDS):
    cleanSet = []
    for review in theSet:
        cleanReview = []
        for word in review:
            if (word.lower() not in theStopWords):
                cleanReview.append(word)
        cleanSet.append(cleanReview)
    return cleanSet

def getFrequencyDicts(train_set, train_labels, label_list):
    # Create a templateDict, which contains ALL words in train_set (with frequency 0)
    templateDict = {}
    # templateDict[UNKNOWN] = 0
    for entry in train_set:
        for elem in entry:
            templateDict[elem] = 0
    print('Distinct elements in training set:', len(templateDict))

    # Create the masterDict, which contains frequency of ALL ELEMENTS
    masterDict = copy.deepcopy(templateDict)
    # masterDict[UNKNOWN] = 0
    for entry in train_set:
        for elem in entry:
            masterDict[elem] = masterDict[elem] + 1

    # Now, create individual dicts for each class
    # only contains the words in that specific class
    # we initially assume no unknown elements, so we set currDict[UNKNOWN] = 0
    dicts = []
    for label in label_list:
        currDict = {}
        currDict[UNKNOWN] = 0
        for i in range(len(train_set)):
            for elem in train_set[i]:
                # print('Element:', elem)
                # if that element is in the current train label, add it to the corresponding frequency dict
                if (train_labels[i] == label):
                    if (elem in currDict):
                        currDict[elem] = currDict[elem] + 1
                    else:
                        currDict[elem] = 1
        dicts.append(currDict)

    # print out the size of the dicts
    for d in dicts:
        print('Size:', len(d))

    return masterDict, dicts

def shortenDicts(dicts, shortenLength):
    # # Generate the shortened master dict
    # masterDict = copy.deepcopy(mDict)
    # shortenedMasterDict = {}
    # for i in range(shortenLength):
    #     maxWord = ''
    #     maxFreq = -1
    #     if (not masterDict):    # true if masterDict is empty
    #         break
    #     for word in masterDict:
    #         if (masterDict[word] > maxFreq):
    #             maxFreq = masterDict[word]
    #             maxWord = word
    #     # print(maxWord)
    #     # print(masterDict[maxWord])
    #     shortenedMasterDict[maxWord] = maxFreq
    #     masterDict.pop(maxWord)
    #     # print(len(masterDict))
    # print('Shortened Master Dict Length:', len(shortenedMasterDict))

    # Generate the temporary shortened dicts
    shortenedDicts = []
    for dict in dicts:
        currDict = copy.deepcopy(dict)
        currShortenedDict = {}

        # find the 5000 most common words
        for i in range(shortenLength):
            maxWord = ''
            maxFreq = -1
            if (not currDict):  # true if you've included every word in currDict
                break
            for word in currDict:
                if (currDict[word] > maxFreq):
                    maxFreq = currDict[word]
                    maxWord = word
            currShortenedDict[maxWord] = maxFreq
            currDict.pop(maxWord)

        # add the UNKNOWN key
        if (UNKNOWN not in currShortenedDict):
            currShortenedDict[UNKNOWN] = 0

        # all other words become UNKNOWN (whatever is not in top 5000)
        for word in currDict:
            if (word == UNKNOWN or word not in currShortenedDict):
                currShortenedDict[UNKNOWN] += currDict[word]

        shortenedDicts.append(currShortenedDict)

    return shortenedDicts

    # # Truncate all!
    # shortenedDicts = []
    # for dict in dicts:
    #     shortenedDict = {}
    #     shortenedDict[UNKNOWN] = 0
    #     for word in dict:
    #         if (word not in shortenedMasterDict):
    #             shortenedDict[UNKNOWN] = shortenedDict[UNKNOWN] + dict[word]
    #         else:
    #             shortenedDict[word] = dict[word]
    #
    #     shortenedDicts.append(shortenedDict)
    #
    # return shortenedMasterDict, shortenedDicts

def getWordProbabilities(dicts, smoothing_parameter=1.0):
    probs = []
    # Calculate probabilities of each word FOR EACH dict
    for dict in dicts:
        # Step 1: Calculate total number of elements belonging to current label
        elemCount = 0
        for elem in dict:
            elemCount += dict[elem]

        # Step 2: Calculate probability using laplace smoothing
        a = smoothing_parameter
        d = len(dict)
        N = elemCount

        probDict = {}
        for elem in dict:
            probDict[elem] = (dict[elem] + a) / (N + a * d)

        probs.append(probDict)

    for p in probs:
        print('Length of Probability Dict:', len(p))

    return probs

def classifyWordList(wordList, priorProbabilities, wordProbabilities, classList):
    classProbabilities = []
    for i in range(len(classList)):      # in our case, we only have 2 classes
        currentClass = classList[i]

        currProb = math.log(priorProbabilities[i])   # math.log or simply log ?

        for word in wordList:
            if (word not in wordProbabilities[i]):
                word = UNKNOWN
            currProb += math.log(wordProbabilities[i][word])

        classProbabilities.append(currProb)

    # get index of highest prob, that's the class you want
    # if ('horrible' in wordList and 'terrible' in wordList and 'bad' in wordList and 'slow' in wordList and 'boring' in wordList and 'bland' in wordList):
    #     print(classProbabilities)
    #     print(wordList)
    return classList[classProbabilities.index(max(classProbabilities))], classProbabilities

def classifyWordLists(dev_set, priorProbabilities, wordProbabilities, classList):
    classes = []
    classProbabilities = []

    for elemList in dev_set:
        currClass, currClassProbability = classifyWordList(elemList, priorProbabilities, wordProbabilities, classList)
        classes.append(currClass)
        classProbabilities.append(currClassProbability)

    return classes, classProbabilities

def convertToTuple(the_set):
    newSet = []
    for o in the_set:
        cr = []
        for m in o:
            cr.append(tuple([m]))
        newSet.append(cr)
    return newSet


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8, returnProbs=False):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set

    # Step 0: Initialize all variables
    # global UNKNOWN
    # UNKNOWN = get_random_string(25)
    priorProbabilities = [1.0 - pos_prior, pos_prior]

    # # Step 1: Remove stop words in train_set and dev_set
    # train_set = removeStopWords(train_set)
    # dev_set = removeStopWords(dev_set)

    # test: convert to tuple
    # print(train_set[0])
    # train_set = convertToTuple(train_set)
    # dev_set = convertToTuple(dev_set)
    # print(train_set[0])

    # Step 2: map words to their frequency for each class
    labels = (0, 1)
    print('Generating Frequency Dictionaries:')
    masterDict, dicts = getFrequencyDicts(train_set, train_labels, labels)

    # Step 3: find the 5000 most frequent words (in the master dict)
    # Then, across all the dicts, keep only those words (all other words should be added to UNKNOWN key)
    # the unknown key is randomly generated
    # in essence, vocab is shortened to 5000 words

    # # Use the following code if you WANT to shorten the dicts
    # maxLength = 5000
    # print('Shortening Dictionaries to', maxLength, 'words.')
    # shortenedDicts = shortenDicts(dicts, maxLength)

    # USE THE following code if you DON'T want to shorten the dicts:
    shortenedDicts = dicts
    for d in shortenedDicts:
        if (UNKNOWN not in d):
            d[UNKNOWN] = 0

    # Step 4: For each label, calculate the probability that a specific word will occur
    print('Generating word probabilities.')
    wordProbabilities = getWordProbabilities(shortenedDicts, smoothing_parameter)

    # print('Word Probabilities:\n\n')
    # print(wordProbabilities[0])
    # print()
    # print(wordProbabilities[1])
    # print()

    # Step 5: GO WILD AND CLASSIFY!!!!
    print('Classifying Words')
    classifications, classificationProbabilities = classifyWordLists(dev_set, priorProbabilities, wordProbabilities, labels)
    print('Finished Classifying Words')
    if (returnProbs):
        return classifications, classificationProbabilities
    return classifications

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=0.0003, bigram_lambda=0.05,pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model

    # Step 0: Initialize all variables
    # global UNKNOWN
    # UNKNOWN = get_random_string(25)
    priorProbabilities = [1.0 - pos_prior, pos_prior]

    # Step 1: Run the unigram model, and get the probabilities
    print('Gathering Unigram Probabilities:\n')
    ts = copy.deepcopy(train_set)
    ds = copy.deepcopy(dev_set)
    ts = removeStopWords(ts)
    ds = removeStopWords(ds)
    unigramClassifications, unigramClassificationProbabilities = naiveBayes(ts, train_labels, ds, smoothing_parameter=unigram_smoothing_parameter, pos_prior=pos_prior, returnProbs=True)
    print()

    print('Beginning Bigram Processing:\n')

    # Step 2: Remove stopwords from train and test set
    print('Removing Stop Words (pre-bigram)')
    # train_set = removeStopWords(train_set)
    # dev_set = removeStopWords(dev_set)

    # Step 3: Generate a new train and dev set
    # Where each entry is a list of bigrams rather than a list of words

    print('Generating bigrams')
    # print('Non-Bigram Sample:\n', train_set[1], '\n')
    train_set_bigrams = generateBigrams(train_set)
    dev_set_bigrams = generateBigrams(dev_set)
    # print('Bigram Sample:\n', train_set_bigrams[1], '\n')

    # def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8, returnProbs=False):

    # Step 4: Map bigrams to their frequencies
    labels = (0, 1)
    print('Generating Frequency Dictionaries for Bigrams:')
    masterBigramDict, bigramDicts = getFrequencyDicts(train_set_bigrams, train_labels, labels)
    # for e in bigramDicts[0]:
    #     print(e, bigramDicts[0][e])

    # Step 5 (optional): Shorten Bigrams (5000 bigrams)
    # Find top 5000 bigrams, then across all the dicts, keep only those words (all other words should be added to UNKNOWN key)
    # the unknown key is randomly generated
    # in essence, bigram count is shortened to 5000 words

    # # Use the following code if you WANT to shorten the dicts
    # maxLength = 5000
    # print('Shortening Dictionaries to', maxLength, 'words.')
    # shortenedDicts = shortenDicts(bigramDicts, maxLength)

    # USE THE following code if you DON'T want to shorten the dicts:
    msg = 'I_AM_NOT_SHORTENING_DICTS_I_AM_SIMPLY_ADDING_UNKNOWN_KEY'
    shortenedBigramDicts = bigramDicts
    for d in shortenedBigramDicts:
        if (UNKNOWN not in d):
            print('put unknown')
            d[UNKNOWN] = 0

    # Step 6: For each label, generate the probability that a specific bigram will occur
    print('Generating bigram probabilities.')
    bigramProbabilities = getWordProbabilities(shortenedBigramDicts, bigram_smoothing_parameter)

    # for i in range(10):
    #     print(bigramProbabilities[i])
    # for x in bigramProbabilities[0]:
    #     print(x, bigramProbabilities[0][x])

    # Step 7: GO WILD AND CLASSIFY THE BIGRAMS!!!
    print('Classifying bigrams.')
    bigramClassifications, bigramClassificationProbabilities = classifyWordLists(dev_set_bigrams, priorProbabilities, bigramProbabilities, labels)

    # bigramClassifications, bigramClassificationProbabilities = naiveBayes(train_set_bigrams, train_labels, dev_set_bigrams, smoothing_parameter=bigram_smoothing_parameter, pos_prior=pos_prior, returnProbs=True)

    # Now that we have the "probabilities" generated by the bigram and unigram
    # use lambda to calculate a composite probability
    # and use that for the final classification

    labels = (0, 1)
    classifications = []
    for i in range(len(bigramClassificationProbabilities)):
        currProbs = []
        # print(bigramClassificationProbabilities[i])
        for p in range(len(bigramClassificationProbabilities[i])):
            currProbs.append((1 - bigram_lambda) * unigramClassificationProbabilities[i][p] + bigram_lambda * bigramClassificationProbabilities[i][p])
            # currProbs.append(bigramClassificationProbabilities[i][p])
        classifications.append(labels[currProbs.index(max(currProbs))])     # index 0 is neg, index 1 is pos

    return classifications

# input a list of lists of strings
# output: a list of tuples of bigrams (each bigram is alphabetically sorted tuple)
# ex.
# input: [['a', 'b', 'c'], ['d', 'e', 'f']]
# output: [[['a', 'b'], ['a', 'c'], ['b', 'c']], [['d', 'e'], ['d', 'f'], ['e', 'f']]]
def generateBigrams(theData):
    bigramData = []
    for entry in theData:
        currBigrams = []
        for i in range(len(entry) - 1):
            currBigrams.append((entry[i], entry[i+1]))
        bigramData.append(currBigrams)

    return bigramData
