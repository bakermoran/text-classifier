import pandas as pd
from classifier import classifier as c

def tests():
  """Run some tests on the classifer class."""

  # read in the data
  text = pd.read_csv('input_texts.csv')
  tweet = pd.read_csv('input_tweet.csv')

  # a test train/test split value
  train_split = .7

  # initialize the classifiers
  text_classifier = c.Classifier(text, train_split=train_split)
  tweet_classifier = c.Classifier(tweet, train_split=train_split)

  # run both models with internal test data
  print('\ntext data')
  text_classifier.test_model()
  print('\ntweet data')
  tweet_classifier.test_model()

  # test the single text value classifier function
  content = 'input some tweet data here'
  print('testing a single line input on tweet data:', content)
  label = tweet_classifier.find_label(content)
  print(label)
  print('SUCCESS')

  # test inputing unclassified data
  print('testing an input of unclassified texts')
  unclassified_data = pd.read_csv('input_texts.csv')
  unclassified_data = unclassified_data.drop(['label'], axis=1)
  output_classified = text_classifier.classify_texts(unclassified_data)
  print(output_classified.head())
  print('SUCCESS')

  # test different column names
  print('testing with different column name inputs')
  tweet_new_names = pd.read_csv('input_tweet.csv')
  tweet_new_names = tweet_new_names.rename(columns={'label': 'feeling', 'content': 'tweet_text'})
  tweet_classifier_new_names = c.Classifier(tweet_new_names, train_split=train_split, label_column='feeling', text_column='tweet_text')
  tweet_classifier_new_names.test_model()
  print('SUCCESS')

if __name__ == '__main__':
  tests()
