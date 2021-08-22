"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
import numpy as np
import copy

UNKNOWN = 'UNKNOWN_ELEMENT'
UNSEEN_TAG = 'UNSEEN_TAG'
BACKTRACE_END = -1

def viterbi_1(train, test):
    print(len(test))
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # Step 1: Generate Frequencies of tags, tagPairs (tagB | tagA), and words given tags (word | tagB)
    tagFrequencies, tagPairFrequencies, wordTagFrequencies, uniqueWords, uniqueTags = generateFrequencies(train, test)

    # for tag in tagPairDicts:
    #     print(tag)
    # print(len(tagPairDicts))
    # print()
    # for tag in wordTagDicts:
    #     print(tag)
    # print(len(uniqueWords))
    #
    # print('DONE')

    # Step 2: Calculate Probabilities (with smoothing)
    tagProbabilitySmoothingParameter = 0.001
    tagPairProbabilitySmoothingParameter = 0.001
    wordTagProbabilitySmoothingParameter = 0.001

    tagProbabilities = generateProbabilities(tagFrequencies, tagProbabilitySmoothingParameter)

    tagPairProbabilities = {}
    for tagA in tagPairFrequencies:
        tagPairProbabilities[tagA] = generateProbabilities(tagPairFrequencies[tagA], tagPairProbabilitySmoothingParameter)

    wordTagProbabilities = {}
    for tag in wordTagFrequencies:
        wordTagProbabilities[tag] = generateProbabilities(wordTagFrequencies[tag], wordTagProbabilitySmoothingParameter)

    # Step 3: Compute log of all probabilities
    tagProbabilitiesLog = getLogProbabilities(tagProbabilities)

    tagPairProbabilitiesLog = {}
    for tagA in tagPairProbabilities:
        tagPairProbabilitiesLog[tagA] = getLogProbabilities(tagPairProbabilities[tagA])

    wordTagProbabilitiesLog = {}
    for tag in wordTagProbabilities:
        wordTagProbabilitiesLog[tag] = getLogProbabilities(wordTagProbabilities[tag])

    # Step 4: Construct the trellis (viterbi matrix and backtrace matrix)
    # Step 5: Backtrace the backtrace matrix to find the best labels
    # Both steps are intertwined
    labeledSentences = []
    for sentence in test:
        viterbiMatrix, backtraceMatrix, tagIndicesDict = constructTrellis(sentence, tagProbabilitiesLog, tagPairProbabilitiesLog, wordTagProbabilitiesLog, uniqueTags)
        labeledSentences.append(backtraceTrellis(viterbiMatrix, backtraceMatrix, sentence, tagIndicesDict))


    return labeledSentences

def backtraceTrellis(viterbiMatrix, backtraceMatrix, sentence, tagIndicesDict):
    labeledSentence = []
    n = len(sentence)
    m = len(tagIndicesDict)
    k = n-1

    # Find the index of the max value in last row of viterbi matrix
    # That's the index where backtrace will begin
    maxVal = viterbiMatrix[k, 0]
    maxValTagIndex = 0
    for tagB_index in range(m):
        if (viterbiMatrix[k, tagB_index] > maxVal):
            maxVal = viterbiMatrix[k, tagB_index]
            maxValTagIndex = tagB_index

    # the last word labeled
    labeledSentence = [(sentence[k], tagIndicesDict[maxValTagIndex])]
    currentTagIndex = maxValTagIndex     # currentTagIndex contains index of current tag which must be appended

    k = k - 1       # k = n - 2
    while (k >= 0):
        # find the currentTagIndex (using the backtrace matrix)
        currentTagIndex = backtraceMatrix[k + 1, currentTagIndex]
        labeledSentence = [(sentence[k], tagIndicesDict[currentTagIndex])] + labeledSentence

        k = k - 1

    # print(labeledSentence)
    # print()
    return labeledSentence


def constructTrellis(sentence, tagProbabilitiesLog, tagPairProbabilitiesLog, wordTagProbabilitiesLog, uniqueTags):
    # Step 1: Assign indices to each tag (column of viterbi matrix corresponds to that tag)
    tagIndex = 0
    tagIndicesDict = {}
    for tag in uniqueTags:
        tagIndicesDict[tagIndex] = tag
        tagIndex += 1

    # Step 2: Initialize the trellis
    n = len(sentence)
    m = len(uniqueTags)
    viterbiMatrix = np.int32(np.zeros((n, m)))
    backtraceMatrix = np.int32(np.zeros((n, m)))

    # Step 3: Fill out the first row of the trellis
    # Viterbi matrix first row: P(tag) * P(word0 | tag)
    # Note: log probabilities are used, so we sum rather than multiply
    for i in range(m):
        currentTag = tagIndicesDict[i]
        currentWord = sentence[0]
        currentWordTagProbabilities = wordTagProbabilitiesLog[currentTag]   # P(words | currentTag)
        if (currentWord in currentWordTagProbabilities):
            viterbiMatrix[0, i] = tagProbabilitiesLog[currentTag] + currentWordTagProbabilities[currentWord]
        else:
            viterbiMatrix[0, i] = tagProbabilitiesLog[currentTag] + currentWordTagProbabilities[UNKNOWN]
        backtraceMatrix[0, i] = BACKTRACE_END

    # Step 4: Iteratively fill out rows of the viterbi matrix
    # The calculation for each cell in the current row (row k+1) should use the previous row's tag which maximized the value
    # We calculate as follows: max_tagA(viterbiMatrix[k, tagA] * P(tagB | tagA) * P(currentWord | tagB))  - we use the tagA which maximizes this calculation
    # backtraceMatrix[k, tagB] contains the index of the tagA which maximizes path value to that cell (we prefer highest value path)
    # Note: log probabilities are used, so we sum rather than multiply
    for k in range(0, n - 1):
        for tagB_index in range(m):
            # current tag
            tagB = tagIndicesDict[tagB_index]

            # current emission probability (constant regardless of tagA)
            # must find only ONCE every cell
            currentWord = sentence[k+1]
            currentWordTagProbabilities = wordTagProbabilitiesLog[tagB]     # P(words | tagB)
            if (currentWord in currentWordTagProbabilities):
                emissionProbability = currentWordTagProbabilities[currentWord]
            else:
                emissionProbability = currentWordTagProbabilities[UNKNOWN]

            # current transition probability
            # tagA must maximize: viterbiMatrix[k, tagA] * P(tagB | tagA) * P(currentWord | tagB)
            # we will do calculation using ALL potential tagA values
            # and pick the tagA which maximizes the value, and include that tagA in backtrace matrix
            maxTagA_Index = 0
            maxValue = -100000000000000000
            for tagA_index in range(m):
                tagA = tagIndicesDict[tagA_index]

                prev = viterbiMatrix[k, tagA_index]

                if (tagA in tagPairProbabilitiesLog):
                    currentTagPairProbabilities = tagPairProbabilitiesLog[tagA]
                    if (tagB in currentTagPairProbabilities):
                        transitionProbability = currentTagPairProbabilities[tagB]
                    else:
                        transitionProbability = currentTagPairProbabilities[UNKNOWN]
                else:
                    # print(tagA)
                    transitionProbability = -100000000000000000       # this is when a potential tagA is NEVER followed by another tag, so it obviously can't be the previous tag

                currentValue = prev + transitionProbability + emissionProbability

                if (currentValue > maxValue):
                    maxTagA_index = tagA_index
                    maxValue = currentValue

            # fill out the viterbi and backtrace matrices
            viterbiMatrix[k+1, tagB_index] = maxValue
            backtraceMatrix[k+1, tagB_index] = maxTagA_index

    # Step 5: Return!!!!!!
    return viterbiMatrix, backtraceMatrix, tagIndicesDict

def getLogProbabilities(dict):
    logDict = {}
    for key in dict:
        logDict[key] = math.log(dict[key])

    return logDict

# converts a frequencyDict into a probabilityDict
# frequencies replaced with probabilities
def generateProbabilities(dict, smoothingParameter=1.0):
    # Step 1: Calculate total number of elements
    elemCount = 0
    for elem in dict:
        elemCount += dict[elem]

    # Step 2: Calculate probability using laplace smoothing
    a = smoothingParameter
    d = len(dict)
    N = elemCount

    probDict = {}
    for elem in dict:
        probDict[elem] = (dict[elem] + a) / (N + a * d)

    return probDict

def generateFrequencies(train, test):
    tagFrequencies = {}     # maps each tag to the number of times it occurs over the train data
    tagPairDicts = {}   # each tag corresponds to dict containing frequency of subsequent tags (transition probabilities)
    wordTagDicts = {}   # each tag corresponds to dict containing frequency of words (emmision probabilities)
    uniqueWords = set()     # set of unique words in training data
    uniqueTags = set()      # set of unique tags in training data

    for sentence in train:
        for i in range(len(sentence)):
            # current word and tag
            currentWord = sentence[i][0]
            currentTag = sentence[i][1]

            # modify tagFrequencies
            if (currentTag in tagFrequencies):
                tagFrequencies[currentTag] += 1
            else:
                tagFrequencies[currentTag] = 1

            # modify wordTagDicts
            # make sure that every wordTagDict has an UNKNOWN key (with value 0)
            if (currentTag in wordTagDicts):
                currentWordTagDict = wordTagDicts[currentTag]
            else:
                currentWordTagDict = {}
                currentWordTagDict[UNKNOWN] = 0

            if (currentWord in currentWordTagDict):
                currentWordTagDict[currentWord] += 1
            else:
                currentWordTagDict[currentWord] = 1

            wordTagDicts[currentTag] = currentWordTagDict

            # modity tagPairDicts (only when i > 0, when there IS a previous tag in sentence)
            # make sure every tagPairDict has an UNKNOWN key (with value 0)
            if (i > 0):
                previousWord = sentence[i-1][0]
                previousTag = sentence[i-1][1]

                if (previousTag in tagPairDicts):
                    currentTagPairDict = tagPairDicts[previousTag]
                else:
                    currentTagPairDict = {}
                    currentTagPairDict[UNKNOWN] = 0

                if (currentTag in currentTagPairDict):
                    currentTagPairDict[currentTag] += 1
                else:
                    currentTagPairDict[currentTag] = 1

                tagPairDicts[previousTag] = currentTagPairDict

            ukDict = {}
            ukDict[UNKNOWN] = 0
            tagPairDicts[UNKNOWN] = ukDict

            # modify uniqueWords and uniqueTags
            uniqueWords.add(currentWord)
            uniqueTags.add(currentTag)

    return tagFrequencies, tagPairDicts, wordTagDicts, uniqueWords, uniqueTags
