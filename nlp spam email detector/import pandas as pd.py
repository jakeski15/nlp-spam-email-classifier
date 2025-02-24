import pandas as pd
import nltk
import numpy

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('wordnet')
nltk.download('stopwords')

stopwords = stopwords.words('english')
dataframe = pd.read_csv(r'D:\pythonscripts\nlp spam email detector\Spam Email raw text for NLP.csv')
lemmatizer = WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r"\w+")

#turns a string (email) to a list of useful tokens for the model
def mail_to_token_list(s):
  s_tokenized = tokenizer.tokenize(s)
  lc_tokens = [word.lower() for word in s_tokenized]
  lemmatized_tokens = [lemmatizer.lemmatize(word) for word in lc_tokens]
  useful_tokens = [word for word in lemmatized_tokens if word not in stopwords]
  return useful_tokens

#should_keep_token returns True if we should keep the token, False if we should not keep the token
def should_keep_token(token, n):
  if token_counter[token] >= n:
    return True
  else:
    return False

dataframe = dataframe.sample(frac=1, random_state=1)
dataframe = dataframe.reset_index(drop=True)
split_index = int(len(dataframe) * 0.8)
train_data = dataframe[:split_index]
test_data = dataframe[split_index:]
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

#gets a counter of the number of times each useful token appears
token_counter = {}
for message in train_data['MESSAGE']:
  message_tokens = mail_to_token_list(message)
  for tok in message_tokens:
    if tok in token_counter:
      token_counter[tok] += 1
    else:
      token_counter[tok] = 1

bad_words = set()
for i in token_counter:
  if should_keep_token(i, 2000):
    bad_words.add(i)

list_bad_words = list(bad_words)
bad_words_indecies = {word:i for word, i in zip(bad_words, range(len(bad_words)))} #maps each word to an index

#count the "bad" words in the given input string s
def count_bad_words_in_message(s): 
  s_toks = mail_to_token_list(s)
  count_vector = numpy.zeros(len(bad_words))
  for tok in s_toks:
    if tok in bad_words_indecies:
      count_vector[bad_words_indecies[tok]] += 1
  return count_vector

#converts a dataframe to X-y model
def dataframe_to_X_y(df):
  y = df['CATEGORY'].to_numpy().astype(int) #the y column is a list of values 0 and 1 which represents not spam versus spam. We get this from df[category], and then convert to a numpy array
  messages = df['MESSAGE'] #gets the list of messages
  count_vectors = []
  for message in messages:
    count_vector = count_bad_words_in_message(message)
    count_vectors.append(count_vector)
  X = numpy.array(count_vectors).astype(int)
  return X, y

X_train, y_train = dataframe_to_X_y(train_data)
X_test, y_test = dataframe_to_X_y(test_data)

scaler = MinMaxScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

#use the sklearn model to see how accurate our predictions are based on the tests
lr = LogisticRegression().fit(X_train, y_train)
print(classification_report(y_test, lr.predict(X_test)))
rfc = RandomForestClassifier().fit(X_train, y_train)
print(classification_report(y_test, rfc.predict(X_test)))
