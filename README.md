# This is a naive Bayesian text classifier

It's purpose is to take in a set of data with two columns, a label and some content. It uses the content to learn what labels go with it, and can then be used to label some other un-labeled content.

## Instantiation variables

* `input_data` - a pandas dataframe with two columns, one for the label and one for the content
* `train_split` - the proportion of the data to be used to train the model (defaults to 70% train), the rest can be used to test its accuracy. Data is randomly assigned a which label it gets.
* `debug` - print out extra output for debugging purposes (defaults to `False`)
* `label_column` - the name of the label column (defaults to `label`)
* `text_column` - the name of the label column (defaults to `content`)

## Public class methods and variables

### Methods

* `test_model()`
  * requires - `self.train_split` < 1
  * modifies - `self.num_test_texts`, `self.num_correct_test_texts`, output stream
  * effects - tests the model on the preset test data and outputs a percent correct to the command line. If debug is set to true, outputs each test text, the predicted label, correct label, and log-probability value
* `classify_texts(data, text_column='content')`
  * requires - `data` is a valid dataframe with a column for the content that is to be labeled that matches `text_column` (which defaults to 'content')
  * modifies - none
  * effects - returns a dataframe with the original column, as well as a column titiles 'label' with the predicted label of the content
* `process_content(content)`
  * requires - `content` is a vector of strings
  * modifies - `content`
  * effects - returns `content` after removing stop words, puctuation, all lowercase, and changing to root words
* `find_label(content)`
  * requires - `content` is a vector of strings (should probably call `process_content(content)` on it first)
  * modifies - output stream
  * effects - returns the predicted label for the strings based on the model

### Variables

* all variables in `Instantiation variables`
* `vocab_size` - the number of distinct words in the training data
* `num_test_texts` - the number of data in the test data (only set after `test_model()` is called)
* `num_correct_test_texts` - the number of data in the test data that were correctly labeled (only set after `test_model()` is called)

## Usage guide

First install package. Navigate to the directory in terminal and pip install it.

```bash
cd text-classifier
pip3 install -e .
```

Second, import in your analysis file.

Then use:

```python
# read in the data
raw_data = pd.read_csv('input_data.csv')

# initialize the classifier
text_classifier = Classifier(text, train_split=.5, debug=False, label_column='label', text_column='content')

# run models with internal test data
text_classifier.test_model()

# run on a single input
content = 'input some tweet data here'
content = tweet_classifier.process_content(content)
label = tweet_classifier.find_label(content)
print(label)

 # with unclassified data
unclassified_data = pd.read_csv('other_input.csv')
output_classified = text_classifier.classify_texts(unclassified_data)
print(output_classified.head())
```
