import pandas as pd
import numpy as np
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import math
import sys

class Classifier:
  """This is a class that can train a naive Bayesian text classifier.
     Inputs should be a dataframe of training data with columns 'label' and 'content',
     as well as a train split variable (0,1] to designate a train/test data split of that data.
     When the class is instantiated, it is trained on the data. To test it, run test_model().
     To classify unclassified texts and return a dataframe, run classify_texts(dataframe).
     Debug is an optional argument for more output to command line."""


  def __init__(self, input_data, train_split=0.7, debug=None, label_column='label', text_column='content'):
    """Constructor class."""

    self.debug = debug if debug is not None else False
    self.train_split = train_split
    self.label_column = label_column
    self.text_column = text_column
    self.__validate_inputs(input_data, train_split)
    self.__stemmer = SnowballStemmer("english")
    self.__stop_words = set(stopwords.words('english'))

    self.input_data = input_data.rename(columns={self.label_column: 'label', self.text_column: 'content'})
    self.vocab_size = 0
    self.num_test_texts = 0
    self.num_correct_test_texts = 0
    self.__words = {}
    self.__labels = {}
    self.__words_per_label = {}

    self.__test_train_split()
    self.__train_model()


  def __validate_inputs(self, input_data, train_split):
    """Check class inputs for validity."""
    if train_split <= 0 or train_split > 1:
      raise Exception('ERROR: Train data split must be a real number on the interval (0,1]')
    if self.label_column not in input_data or self.text_column not in input_data:
      raise Exception('ERROR: Input data must contain columns \'', self.label_column, '\' and \'', self.text_column, '\'')
    # if any(input_data.isna()):
    #   raise Exception('ERROR: Input data cannot contain NULLs')


  def __test_train_split(self):
    """Randomize order of the rows and split the data into a testing and training dataset."""
    if self.train_split == 1:
      self.train_data = self.input_data
      self.test_data = None
      self.num_train_texts = len(self.train_data)
      return

    self.input_data['split'] = np.random.randn(self.input_data.shape[0], 1)
    msk = np.random.rand(len(self.input_data)) <= self.train_split
    self.train_data = self.input_data[~msk]
    self.test_data = self.input_data[msk]
    self.num_train_texts = len(self.train_data)


  def __train_model(self):
    """Train the model on the training data."""

    for row in self.train_data.itertuples():
      content = row.content.split(' ')
      label = row.label
      content = self.process_content(content)

      if label in self.__labels.keys(): self.__labels[label] += 1
      else: self.__labels[label] = 1

      for word in content:
        # add to all words dictionary
        if word in self.__words.keys(): self.__words[word] += 1
        else:
          self.__words[word] = 1
          self.vocab_size += 1

        # add to labels<word> dictionary
        if label in self.__words_per_label.keys():
          if word in self.__words_per_label[label].keys(): self.__words_per_label[label][word] += 1
          else: self.__words_per_label[label][word] = 1
        else: self.__words_per_label[label] = {word: 1}

    if self.debug:
      print('Training the Model on', self.num_train_texts, 'texts')
      print('vocabulary size =', self.vocab_size)
      print('Detected', len(self.__labels), 'different labels\n')
      print('Log priors of each label:')
      for label in self.__labels.keys():
        print('\t', label, '=', self.__log_prior(label))

      print('\nModel parameters:')
      for label, val in self.__words_per_label.items():
        for word, wcount in val.items():
          print('\t', label, ':', word, ', count =', wcount, ', log-likelihood =', '{0:.4f}'.format(self.__log_likelihood(word, label)))
      print('\n')


  def __log_prior(self, label):
    """Return the log prior of a label."""

    return math.log(self.__labels[label] / self.num_train_texts)


  def __log_likelihood(self, word, label):
    """Return the log likelihood of a word with a label."""

    if word not in self.__words_per_label[label].keys():
      # if word is not in any of trainig texts it computes natural log of 1 / num texts
      if word not in self.__words.keys():
        return math.log(1.0/self.num_train_texts)
      else:
        # if word is in a training text but not one with that particular
        # label computes natural log of texts with that label / total texts
        return math.log(self.__words[word] / self.num_train_texts)
    else:
      # if word is in a training text with that label computes the natural log of
      # number of texts with that label that contain that word / the num texts
      # with the label
      return math.log(self.__words_per_label[label][word] / self.__labels[label])


  def __calc_probability(self, text):
    """Calculate the probability of a text and return the label with the log probability."""

    highest_log_prob = -sys.maxsize
    predicted_label = ''
    for label in self.__labels.keys():
      temp_log_prob = self.__find_log_probability(text, label)
      if temp_log_prob > highest_log_prob:
        highest_log_prob = temp_log_prob
        predicted_label = label
    return predicted_label, highest_log_prob


  def __find_log_probability(self, text, label):
    """Find the log probablity of a text with a label."""

    log_probability = self.__log_prior(label)
    for word in text:
      log_probability += self.__log_likelihood(word, label)
    return log_probability


  def find_label(self, content):
    """A wrapper around __calc_probability for external use on a single input line."""
    content = self.process_content(content)
    return self.__calc_probability(content)


  def test_model(self):
    """Test the model on the testing dataset."""

    if self.train_split == 1:
      raise Exception('ERROR: No test data provided')
    for row in self.train_data.itertuples():
      content = row.content.split(' ')
      label = row.label
      content = self.process_content(content)

      predicted_label, highest_log_prob = self.__calc_probability(content)
      if predicted_label == label: self.num_correct_test_texts += 1
      self.num_test_texts += 1

      if self.debug:
        print('content =', row.content)
        print('\tpredicted_label =', predicted_label, ', actual label =', label, ', with log-probability =', '{0:.4f}'.format(highest_log_prob))


    print('\ngot', self.num_correct_test_texts, 'out of', self.num_test_texts, 'test texts correct')
    print('test accuracy:', self.num_correct_test_texts / self.num_test_texts * 100, '%')


  def classify_texts(self, data, text_column='content'):
    """Take in a dataframe and return a new dataframe with a label column using the model."""

    data = data.rename(columns={text_column: 'content'})
    classified_texts = 0
    return_data = data
    column = []
    for row in data.itertuples():
      content = row.content.split(' ')
      content = self.process_content(content)
      predicted_label, highest_log_prob = self.__calc_probability(content)
      classified_texts += 1
      column.append(predicted_label)
      if self.debug:
        print('content =', row.content)
        print('\tpredicted_label =', predicted_label, ', with log-probability =', '{0:.4f}'.format(highest_log_prob))
    print('\nclassified', classified_texts, 'different texts')
    return_data['label'] = column
    return return_data


  def process_content(self, content):
    """Turn the content into processable content.
       Remove stop words, remove puctuation, lowercase, and get root words."""

    content = [w.lower() for w in content]
    content = [w.translate(str.maketrans('', '', string.punctuation)) for w in content]
    content = [self.__stemmer.stem(w) for w in content]
    content = [w for w in content if not w in self.__stop_words]
    return content
