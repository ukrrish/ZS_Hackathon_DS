
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
import time
import re
import os
from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]


class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self
    def transform(self, posts):
        # construct object dtype array with two columns
        # first column = 'subject' and second column = 'body'
        features = np.empty(shape=(len(posts), 2), dtype=object)
        for index,row text in iterrows(posts):
            features[i, 1] = row["Headline"]
            features[i, 0] = row["Title"]
        return features

class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self
    def transform(self, posts):
        return (list(map(lambda x:{"Topic":x},posts)))

union = FeatureUnion(
        transformer_list=[
            # Pipeline for pulling features from the post's subject line
            ('subject', Pipeline([
                ('selector', ItemSelector(key='Title')),
                ('tfidf', TfidfVectorizer(tokenizer=tokenizer, stop_words='english')),
                #('best', TruncatedSVD(n_components=50)),
            ])),
            # Pipeline for standard bag-of-words model for body
            ('body_bow', Pipeline([
                ('selector', ItemSelector(key='Headline')),
                ('tfidf', TfidfVectorizer(tokenizer=tokenizer, stop_words='english')),
                #('best', TruncatedSVD(n_components=100)),
            ])),
            # Pipeline for pulling ad hoc features from post's body
            ('body_stats', Pipeline([
                ('selector', ItemSelector(key='Topic')),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),
        ],
        # weight components in FeatureUnion
        transformer_weights={
            'subject': 1.0,
            'body_bow': 0.5,
            'body_stats': 1.0,
        },
    )

seed = 101

PATH  = "/home/ubuntu/ankit93726/Hackerearth/UK-ZS/dataset/"
TRAIN_FILE = "train_file.csv"
TEST_FILE = "test_file.csv"


def tokenizer(text):
    if text:
        result = re.findall('[a-z]{2,}', text.lower())
    else:
        result = []
    return result


df = pd.read_csv(os.path.join(PATH,TRAIN_FILE))
df.head()
#map(lambda x:df.x.isnull().sum(),df.columns)
# Index(['IDLink', 'Title', 'Headline', 'Source', 'Topic', 'PublishDate',
#        'Facebook', 'GooglePlus', 'LinkedIn', 'SentimentTitle',
#        'SentimentHeadline'],
#       dtype='object')


#X = (df['Title'] + ' ' + df['Headline']).values
#y = np.log1p(df['SentimentTitle'].values)

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3)
#vect = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
y_train = np.log1p(train['SentimentTitle'].values)
y_test = np.log1p(test['SentimentTitle'].values)


start = time.time()
X_train_vect= union.fit_transform(train)
end = time.time()
print('Time to train vectorizer and transform training text: %0.2fs' % (end - start))




model = SGDRegressor(loss='squared_loss', penalty='l2', random_state=seed, max_iter=10)
params = {'penalty':['none','l2','l1'],
          'alpha':[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}
gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  scoring='neg_mean_squared_error',
                  n_jobs=5,
                  cv=5,
                  verbose=3)
start = time.time()
gs.fit(X_train_vect, y_train)
end = time.time()
print('Time to train model: %0.2fs' % (end -start))


model = gs.best_estimator_
print(gs.best_params_)
print(gs.best_score_)


pipe = Pipeline([('vect',union),('model',model)])
start = time.time()
y_pred = pipe.predict(test)
end = time.time()
print('Time to generate predictions on test set: %0.2fs' % (end - start))




print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))




df_test = pd.read_csv(os.path.join(PATH,TEST_FILE))
df_test.head()

df_test['SentimentTitle'] = np.exp(pipe.predict((df_test))-1)



submission_file = df_test[["IDLink"  ,"SentimentTitle"  ,"SentimentHeadline"]]
submission_file.to_csv(os.path.join(PATH,"submission_tfidf_tsvd_dictvec_sgd.csv"),index=False)
