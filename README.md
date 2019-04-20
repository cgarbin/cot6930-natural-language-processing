# COT6930 Natural Language Processing, Spring 2019

## Assignment 1

Text processing with [OpenNLP](https://opennlp.apache.org/).

Assignment report [here](https://github.com/cgarbin/cot6930-natural-language-processing/blob/master/assignment1/COT-6930%20assignment%201%20report%20cgarbin.pdf).

- Detect sentences with `SentenceDetectorME`
- Extract tokens with `SimpleTokenizer`, `WhitespaceTokenizer`, `TokenizerME` and compare their performance
- Detect parts-of-speech with `POSTaggerME`
- Find named entities with the generic entity finder `NameFinderME`, inititalized with persons (`en-ner-person.bin`),
  locations (`en-ner-location.bin`), money/currencies (`en-ner-money.bin`), and percentages (`en-ner-percentage.bin`).

## Assignment 2

Document classification using [Weka](https://www.cs.waikato.ac.nz/~ml/weka/).

Assignment report [here](https://github.com/cgarbin/cot6930-natural-language-/trprocessingee/master/assignment2) (the README file).

- Create ARFF train and test file from plain text file (already tokenized and stemmed)
- Use Weka's `StringToWordVector` to create word vectors and `FilteredClassifier` to split into train and test datasets
- Use Weka's `AttributeSelection` to select attributes (words) from the text, to fine-tune the classifiers
- Compare the `NaiveBayesMultinomial` with the `LibSVM` classifiers

## Assignment 3

Sentiment analysis with [TextBlob](https://textblob.readthedocs.io/en/dev/).

Assignment report [here](https://github.com/cgarbin/cot6930-natural-language-processing/blob/master/assignment3/COT-6930%20assignment%203%20cgarbin.pdf).

Compare the performance of `PatternAnalyzer` and `NaiveBayesAnalyzer` in sentiment analysis of restaurant reviews.

## Class project

TensforFlow introduction and applications for natural language processing (NLP).

Introduction [here](https://github.com/cgarbin/cot6930-natural-language-processing/tree/master/tensorflow-presentation)
and slide deck used for presentation [here](https://github.com/cgarbin/cot6930-natural-language-processing/blob/master/tensorflow-presentation/COT-6930%20presentation%20-%20TensorFlow.pdf).
