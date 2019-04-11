'''COT6930 Natural Language Processing Spring 2019
Christian Garbin

Assignment 3
Sentiment analysis with TextBlob
'''

from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.sentiments import PatternAnalyzer

import textblob as tb
print(tb.__version__)

# Read all sentences in the file into one string.
# The `replace(..)` is used to combine sentences that are split into two
# lines in the file back into one line.
# TextBlob will parse into sentences, using . ! ? etc. to separate them.
with open('./assignment3/homework 3 dataset.txt', 'r') as file:
    text = file.read().replace('\n', ' ')

blob = TextBlob(text)

print('Naive Bayes Analyzer')
tb = Blobber(analyzer=NaiveBayesAnalyzer())
for sentence in blob.sentences:
    sentiment = tb(str(sentence)).sentiment
    print('{} +{:.2f} -{:.2f} {}'.format(sentiment.classification,
                                         sentiment.p_pos, sentiment.p_neg,
                                         sentence))

print('\n\nPattern Analyzer')
tb = Blobber(analyzer=PatternAnalyzer())
for sentence in blob.sentences:
    sentiment = tb(str(sentence)).sentiment
    classification = 'pos' if sentiment.polarity >= 0 else 'neg'
    print('{} {:+.2f} {:.2f} {}'.format(classification,
                                        sentiment.polarity,
                                        sentiment.subjectivity,
                                        sentence))
