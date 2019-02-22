""" Converts a document file to ARFF format.

Input file: first word is the class; remainder of the line is the document.

Output: an ARFF file, with the header followed lines where the class is the
   first word and the remainder words are in quotes (need to be vectorized
   by another tool."

Note that the class name aer prepended with type_ so they don't match the same
word inside the document. Weka throws an error when that happens.
"""

import sys
import os


def main():

    # Header
    print("@relation type\n")
    print(
        "@attribute page_type "
        "{type_student,type_course,type_faculty,type_project}")
    print("@attribute text String")
    print("\n@data")

    file = sys.argv[1]

    with open(file) as f:
        for line in f:
            s = line.split()
            # Use prefix for type to avoid "Attribute names are not unique"
            # error caused by having the attribute name also as part of the
            # document contents (the text).
            print("type_{}, '{}'".format(s[0], " ".join(s[1:])))


if __name__ == '__main__':
    main()
