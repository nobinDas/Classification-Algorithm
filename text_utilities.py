# -*- coding: utf-8 -*-
"""
Some simple routines for breaking a list of files into chunks of text with 
associated authors. Used to generate input for text classificaiton problems.
"""
import string
import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd

# substitute a directory name here.
working_dir = Path('./HW1 texts')

files = ['Alcott-Lousia May-Eight Cousins.txt',
    "Alcott-Lousia May-Jo's Boys.txt",
    'Alcott-Lousia May-Little Men.txt',
    'Austen-Jane-Mansfield Park.txt',
    'Austen-Jane-Northanger Abbey.txt',
    'Austen-Jane-Persuasion.txt',
    'Austen-Jane-Pride and Prejudice.txt',
    'Doyle-Arthur Conan-The Hound of the Baskervilles.txt',
    'Doyle-Arthur Conan-The Return of Sherlock Holmes.txt',
    'Hawthorne-Nathaniel-Mosses from an Old Manse and Other Stories.txt',
    'Hawthorne-Nathaniel-The Blithedale Romance.txt',
    'Hawthorne-Nathaniel-The House of the Seven Gables.txt',
    'Kafka-Franz-Metamorphosis.txt',
    'Kafka-Franz-The Castle.txt',
    'Melville-Herman-Moby Dick.txt',
    'Melville-Herman-The Piazza Tales.txt',
    'Melville-Herman-Typee.txt',
    'Wilde-Oscar-A Woman of No Importance.txt',
     "Wilde-Oscar-Lady Windermere's Fan.txt",
     'Wilde-Oscar-The Canterville Ghost.txt',
     'Wilde-Oscar-The Importance of Being Earnest.txt',
     'Melville-Herman-White Jacket.txt',
     'Kafka-Franz-The Trial.txt',
    'Hawthorne-Nathaniel-The Scarlet Letter.txt',
     'Doyle-Arthur Conan-The White Company.txt',
     'Austen-Jane-Sense and Sensibility.txt',
     'Alcott-Lousia May-Little Women.txt',
      'Wilde-Oscar-The Picture of Dorian Gray.txt']

train_files = [
    'Alcott-Lousia May-Eight Cousins.txt',
    "Alcott-Lousia May-Jo's Boys.txt",
    'Alcott-Lousia May-Little Men.txt',
    'Austen-Jane-Mansfield Park.txt',
    'Austen-Jane-Northanger Abbey.txt',
    'Austen-Jane-Persuasion.txt',
    'Austen-Jane-Pride and Prejudice.txt',
    'Doyle-Arthur Conan-The Hound of the Baskervilles.txt',
    'Doyle-Arthur Conan-The Return of Sherlock Holmes.txt',
    'Hawthorne-Nathaniel-Mosses from an Old Manse and Other Stories.txt',
    'Hawthorne-Nathaniel-The Blithedale Romance.txt',
    'Hawthorne-Nathaniel-The House of the Seven Gables.txt',
    'Kafka-Franz-Metamorphosis.txt',
    'Kafka-Franz-The Castle.txt',
   'Melville-Herman-Moby Dick.txt',
    'Melville-Herman-The Piazza Tales.txt',
    'Melville-Herman-Typee.txt',
    'Wilde-Oscar-A Woman of No Importance.txt',
     "Wilde-Oscar-Lady Windermere's Fan.txt",
     'Wilde-Oscar-The Canterville Ghost.txt',
     'Wilde-Oscar-The Importance of Being Earnest.txt',
    
    ]
test_files = [
     'Alcott-Lousia May-Little Women.txt',
     'Austen-Jane-Sense and Sensibility.txt',
     'Doyle-Arthur Conan-The White Company.txt',
     'Hawthorne-Nathaniel-The Scarlet Letter.txt',
     'Kafka-Franz-The Trial.txt',     
     'Melville-Herman-White Jacket.txt',
      'Wilde-Oscar-The Picture of Dorian Gray.txt'     ]


def get_author_from_filename(file):
    """Extract the author name and title from a filename 
       of the form lname-fname-title.txt
       """
    lname,fname,filename = file.split('-')
    return lname + '-'+ fname

def read_file(f, lower=True):
    """Simple wrapper to read a file. Converts to lowercase if specified"""
    with open(f,encoding='utf-8') as reader:
        text = reader.read()
        if lower:
            return text.lower()
        return text

def tokenize(text):
    """Simple tokenizer, invoking string split()
    """
    return text.split()

def process_file(file, n = 500, lower=True, post_process=False):
    """Tokenize a file and apply the functions specified in the post_tokenization pipeline. 
    Usually returns a sequence of tokens, but the pipeline could change this. 
    """
    
    # combine workding directory and filename to get path to file
    file_path = Path(working_dir)/file
    
    text = read_file(file_path, lower = lower)
    tokens= tokenize(text)
    
    # determine number of rows given bucket size
    rows = len(tokens) // n

    # truncate input to ensure even splits
    tokens = tokens[:(rows*n)]
    
    # use numpy to reshape input into rows of n tokens
    ar = np.array(tokens)
    ar = ar.reshape((rows,n))
    rows, columns = ar.shape
    
    # recombine each row into a string. 
    strings = []
    for i in range(rows):
        temptoks = ar[i]
        if post_process:
            temptoks = cleanup(temptoks)
        if len(temptoks) != 0:
            strings.append(' '.join(temptoks))
            
    target = get_author_from_filename(file)
    df = pd.DataFrame(strings, columns=['text'])
    df['author'] = target
    return df
    
def generate_training_data():
    """Return a data frame containing two columns, 'text', and 'author'. 
    Each row contains a fragment of text from a larger work by the author.""" 
    training_dfs = []
    for f in train_files:
        training_dfs.append(process_file(f))
    return pd.concat(training_dfs)

def generate_testing_data():
    """Return a data frame containing two columns, 'text', and 'author'. 
    Each row contains a fragment of text from a larger work by the author.""" 
    testing_dfs = []
    for f in test_files:
        testing_dfs.append(process_file(f))
    return pd.concat(testing_dfs)


def cleanup(tokens, nopunct=True, nowhite=True, nonum=True, stop=[]):
    """Use some basic python to clean up tokens.
    stopwords, punctuation, digits are removed."""

    toremove=""
    if nopunct:
        toremove = toremove + string.punctuation
    if nonum:
        toremove = toremove + "0123456789"
    if nowhite:
        toremove = toremove + string.whitespace
    if toremove:
        tab = "".maketrans("", "", toremove)
        tokens = [t.translate(tab).strip() for t in tokens]
        tokens = [t for t in tokens if t]
    return remove_stopwords(tokens, stop)

def remove_stopwords(tokens, stopwords=[]):
    """Takes a list of tokens and returns a list with provided stopwords removed."""
    if stopwords:
        return [t for t in tokens if t not in stopwords]
    else:
        return tokens


def go():
    training_df = generate_training_data()
    testing_df = generate_testing_data()
    return training_df, testing_df
    