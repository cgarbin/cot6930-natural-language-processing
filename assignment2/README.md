# Assignment 2

Document classification using [Weka](https://www.cs.waikato.ac.nz/ml/weka/).

## Assignment details

> Use Naïve Bayes and SVM in Weka to conduct text classification and return
> the classification accuracy.

Input data:

> WebKB containing 2803 training text data and 1396 test data. This data set
> contains WWW-pages collected from computer science departments of various
> universities. These web pages are classified into 4 categories: student,
> faculty, project, and course. The data set has been preprocessed with
> removing stop words and stemming. So you only need to count the word
> frequency to generate a document-word matrix before you start classification.

## Step 1 - Creating the Document-Word matrix

### Preprocessing the data

The goal of the preprocessing step is to transform the data from its current
format to a format that the tool expects.

In this case we need to transform the space-separated text file into an
[ARFF](https://www.cs.waikato.ac.nz/ml/weka/arff.html) file.

This is an example of the input file. Each line represent a document. The first
word in each line is the document class, followed by the document, already
tokeninzed and stemmed.

```
student	brian comput scienc depart ... advisor david wood tabl content  ...
faculty	russel ... california san diego jolla offic appli physic mathemat ...
```

The transformed file looks like this:

```
@relation type

@attribute page_type {type_student,type_course,type_faculty,type_project}
@attribute text String

@data
type_student, 'brian comput scienc depart ... advisor david wood tabl ...'
type_faculty, 'russel ... california san diego jolla offic appli physic ...'
```

The notable features of the new format are:

1. A header that specifies the format of the lines. In this case the format of
   each line is the class, followed by the document.
2. Each document is still a line, but the class and the content of the document
   are separated from each other, as different attributes.

Note that the attribute starts with the rpefix `type_`. This was done because
Weka's classifiers (at least some of them) expect the attribute name to be
unique, i.e. to not appear as part of the document itself.

TODO: add example of this error.

Although Weka has is capable of transforming data, we decided to transform the
data using a Python script. The python script is shown in [this appendix
section](#python-script-for-text-to-arff-transformation).

## Apendix

### Python script for text to ARFF transformation

TODO: add Python script here.
