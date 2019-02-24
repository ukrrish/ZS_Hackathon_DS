import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import sys
sys.path.insert(0, '../input/wordbatch/wordbatch/')
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re


NUM_SOURCES = 5756 #BRAND
NUM_TOPIC = 4
develop = False
# develop= True

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))

def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")




def handle_missing_inplace(dataset):
    dataset['Source'].fillna(value='missing', inplace=True)


def to_categorical(dataset):
    dataset['Topic'] = dataset['Topic'].astype('category')
    dataset['Source'] = dataset['Source'].astype('category')

def cutting(dataset):
    pop_brand = dataset['Source'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_SOURCES]
    dataset.loc[~dataset['Source'].isin(pop_brand), 'Source'] = 'missing'

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])


def main():
    start_time = time.time()
    from time import gmtime, strftime
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    seed = 101
	PATH  = "../input/ZS_HACKATHON/"
	TRAIN_FILE = "train_file.csv"
	TEST_FILE = "test_file.csv"
#
	train = pd.read_csv(os.path.join(PATH,TRAIN_FILE))
    test = pd.read_csv(os.path.join(PATH,TEST_FILE))
#
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    nrow_test = train.shape[0]  # -dftt.shape[0] 
    #dftt = train[(train.price < 1.0)]
    #train = train.drop(train[(train.price < 1.0)].index)
    #del dftt['price']
    nrow_train = train.shape[0]
    # print(nrow_train, nrow_test)
    y = np.log1p(train["SentimentHeadline"]) #Change
    merge: pd.DataFrame = pd.concat([train,test])
    submission: pd.DataFrame = test[['IDLink']]
    del train
    del test
    gc.collect()
    #merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
#
    handle_missing_inplace(merge)
    to_categorical(merge)
#
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=8)
    wb.dictionary_freeze= True
    X_name = wb.fit_transform(merge['Title'])
    del(wb)
    X_name = X_name[:, np.where(X_name.getnnz(axis=0) > 1)[0]]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = CountVectorizer()
    X_category1 = wb.fit_transform(merge['Topic'])
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None})
                             , procs=8)
    X_description = wb.fit_transform(merge['Headline']) #Change
    del(wb)
    X_description = X_description[:, np.where(X_description.getnnz(axis=0) > 1)[0]]
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(merge['Source'])
    X_dummies = csr_matrix(pd.get_dummies(merge[['IDLink', 'Facebook']],
                                          sparse=True).values)
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_name)).tocsr()
    sparse_merge = sparse_merge[:, np.where(sparse_merge.getnnz(axis=0) > 100)[0]]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)
    gc.collect()
    train_X, train_y = X, y
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)
    model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)
    model.fit(train_X, train_y)
    print('[{}] Train FTRL completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
    predsF = model.predict(X_test)
    print('[{}] Predict FTRL completed'.format(time.time() - start_time))
    model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=17, inv_link="identity", threads=4)
    model.fit(train_X, train_y)
    print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    if develop:
        preds = model.predict(X=valid_X)
        print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
    predsFM = model.predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    params = {
        'learning_rate': 0.6,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.65,
        'nthread': 4,
        'min_data_in_leaf': 100,
        'max_bin': 31
    }
    # Remove features with document frequency <=100
    print(sparse_merge.shape)
    sparse_merge = sparse_merge[:, np.where(sparse_merge.getnnz(axis=0) > 100)[0]]
    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_test:]
    print(sparse_merge.shape)
    train_X, train_y = X, y
    if develop:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.05, random_state=100)
    d_train = lgb.Dataset(train_X, label=train_y)
    watchlist = [d_train]
    if develop:
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        watchlist = [d_train, d_valid]
    model = lgb.train(params, train_set=d_train, num_boost_round=7100, valid_sets=watchlist, \
                      early_stopping_rounds=1000, verbose_eval=1000)
    if develop:
        preds = model.predict(valid_X)
        print("LGB dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
    predsL = model.predict(X_test)
    print('[{}] Predict LGB completed.'.format(time.time() - start_time))
    preds = (predsF * 0.18 + predsL * 0.27 + predsFM * 0.55)
    submission['SentimentHeadline'] = np.expm1(preds)
    submission.to_csv(os.path.join(PATH,"submission_wordbatch_ftrl_fm_lgb.csv"), index=False)
if __name__ == '__main__':
    main()














