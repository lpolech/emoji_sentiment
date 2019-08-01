from __future__ import print_function, division
import numpy as np
import pandas as pd
import re, os, io, argparse, nltk, random, sys
import dill as cp
import scipy
import gensim.models as gs
import matplotlib.pyplot as plt
from functools import reduce
from filereader import preprocess, my_tokenizer, read_word_vectors
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay

from scipy.sparse import vstack, hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB#, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import dill as cp
import os

# Business days for NYSE/NASDAQ
class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]
bday_us = CustomBusinessDay(calendar=USTradingCalendar())

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_index', type=int, default=1,
                        help='give an integer between 1 and 9')
    parser.add_argument('--mixture', nargs='+', type=str, default=None,
                        help='Provide a tuple of word embeddings, emojis, and classifier.')
    parser.add_argument('--data_type', type=str, default=None,
                        help='Provide data type.')
    return parser.parse_args()

def list_emojis():
    emoji_list = []
    unicode_list = []
    emoji_dict = {}
    with open("emoji_full_data.js") as input_file:
        for line in input_file.readlines()[1:]:
            splitted_line = line.split("\t")
            emoji_dict[splitted_line[2]]=(splitted_line[1].upper(),splitted_line[3].strip('\n').upper())
            emoji_list.append(splitted_line[2])
            unicode_list.append(splitted_line[1])

    unicode_list_index=sorted(range(len(unicode_list)), key=lambda k: len(unicode_list[k]))
    unicode_list_index.reverse()
    return [emoji_list[item] for item in unicode_list_index], emoji_dict

def data_with_emoji():
    from sklearn.feature_extraction.text import CountVectorizer
    from filereader import preprocess, my_tokenizer
    from scipy import sparse

    df = pd.DataFrame()
    for file_index in range(9):
        df = pd.concat([df,pd.read_pickle(files_dir+'emoji_messages_'+str(file_index+1)+'.pkl')], ignore_index=True)
    print('Files are read!')
    df_sent = df[df.sentiment!='NA']
    df_sent.sentiment = df_sent.sentiment.map({'Bullish': 0, 'Bearish': 1})
    msk = np.random.rand(len(df_sent)) <= 0.9
    train_msgs = df_sent.sample(frac=0.9,random_state=1)
    test_msgs = df_sent.drop(train_msgs.index)
    print("Train and test sets are ready!")
    data = scipy.sparse.load_npz('./Data/data_unibi.csv.npz')

    emoji_vectoriser =  CountVectorizer(tokenizer = lambda text: emoji_pattern.findall(text),
                    preprocessor = lambda message: preprocess(message),
                    analyzer = 'word', max_df = 0.75, min_df = 5, max_features = None,
                    binary = False, ngram_range = (1,1))
    emoji_train = emoji_vectoriser.fit_transform(train_msgs.msgBody)
    emoji_test =  emoji_vectoriser.transform(test_msgs.msgBody)
    with open('vectoriser_emoji.pkl','wb') as fid:
        cp.dump(emoji_vectoriser, fid)
    print("Vectoriser is saved!")
    sentiment = np.asarray([train_msgs.sentiment.tolist() + test_msgs.sentiment.tolist()])
    sentiment = sparse.csr_matrix(sentiment.reshape(2003064, 1))
    emoji_count_data = vstack([emoji_train, emoji_test])
    emoji_count_data = emoji_count_data.astype('float64')
    emoji_count_data_label = hstack([emoji_count_data, sentiment])
    emoji_count_data_label = emoji_count_data_label.tocsr()
    scipy.sparse.save_npz('data_emoji_count.csv.npz', emoji_count_data_label)
    emoji_array = emoji_vectoriser.get_feature_names()
    for polarity_type in ['Polarity1', 'Polarity2', 'Polarity3', 'Polarity4']:
        emoji_file = pd.read_csv('scatter_results_sent.js', sep='\t')
        emoji_pols = pd.Series(emoji_file[polarity_type].values,index=emoji_file.Emoji).to_dict()
        idx = 0
        emoji_pol_data = emoji_count_data
        for item in emoji_array:
            if float(emoji_pols[emoji_array[idx]])!=np.nan:
                emoji_pol_data[:, idx] = emoji_count_data[:, idx]*float(emoji_pols[emoji_array[idx]])
            idx+=1
        emoji_pol_data.data = np.nan_to_num(emoji_pol_data.data)
        out_data = hstack([emoji_pol_data, data])
        scipy.sparse.save_npz('data_emoji_'+polarity_type+'_unibi.csv.npz', out_data)
        out_data = hstack([emoji_pol_data, sentiment])
        scipy.sparse.save_npz('data_emoji_'+polarity_type+'.csv.npz', out_data)
        print(polarity_type + ' has been processed!!')

def emoji_pol(polarity_type='Polarity1'):
    from scipy import sparse

    emoji_file = pd.read_csv('scatter_results_sent.js', sep='\t')
    emoji_pols = pd.Series(emoji_file[polarity_type].values,index=emoji_file.Emoji).to_dict()
    with open('vectoriser_emoji.pkl','rb') as fid:
        vectoriser = cp.load(fid)
    emoji_array = vectoriser.get_feature_names()
    spars_inmat = scipy.sparse.load_npz('./data/data_emoji.csv.npz')
    spars_inmat = spars_inmat.astype('float64')
    idx = 0
    for item in emoji_array:
        if float(emoji_pols[emoji_array[idx]])!=np.nan:
            spars_inmat[:, idx] = spars_inmat[:, idx]*float(emoji_pols[emoji_array[idx]])
        idx+=1
    scipy.sparse.save_npz('data_emoji_'+polarity_type+'.csv.npz', spars_inmat)

def classifier_fun(clf, param):
    class_weight = {1:8, 0:1}
    if clf=='NaiveBayes':
        # return GaussianNB()
        return MultinomialNB(alpha=param['alpha'])
    elif clf=='RFC':
        return RandomForestClassifier(n_estimators = param['n_estimators'], max_features=param['max_features'],
                        max_depth = param['max_depth'], min_samples_split= param['min_samples_split'],
                        min_samples_leaf = param['min_samples_leaf'], bootstrap=param['bootstrap'],
                        criterion = param['criterion'], class_weight = class_weight)
    elif clf=='LogSig':
        return LogisticRegression(penalty=param['penalty'],C=param['C'], class_weight = class_weight)
    elif clf=='SVC':
        return SVC(C=param['C'], kernel=param['kernel'], probability=True,
                        gamma = param['gamma'], degree=param['degree'], class_weight = class_weight)
    elif clf=='MLP':
        return MLPClassifier(hidden_layer_sizes=(100,), class_weight = class_weight)
    elif clf=='LinearSVC':
        classifier = LinearSVC(penalty=param['penalty'],C=param['C'], loss=param['loss'], class_weight = class_weight)
        return CalibratedClassifierCV(classifier,method='sigmoid',cv=3)

def featurise_input(input_train, input_test, feature_type='unibi'):
    if feature_type=='Polarity1':
        vectoriser =  CountVectorizer(tokenizer = lambda text: emoji_pattern.findall(text),
                        preprocessor = lambda message: preprocess(message),
                        analyzer = 'word', max_df = 0.75, min_df = 5, max_features = None,
                        binary = False, ngram_range = (1,1))
        train = vectoriser.fit_transform(input_train)
        test = vectoriser.transform(input_test)
        emoji_file = pd.read_csv('scatter_results_sent.js', sep='\t')
        emoji_pols = pd.Series(emoji_file[feature_type].values,index=emoji_file.Emoji).to_dict()
        emoji_array = vectoriser.get_feature_names()
        idx = 0
        for item in emoji_array:
            if float(emoji_pols[emoji_array[idx]])!=np.nan:
                train[:, idx] = train[:, idx]*float(emoji_pols[emoji_array[idx]])
                test[:, idx] = test[:, idx]*float(emoji_pols[emoji_array[idx]])
            idx+=1
        train.data = np.nan_to_num(train.data)
        test.data = np.nan_to_num(test.data)
    elif feature_type=='Polarity3':
        vectoriser =  CountVectorizer(tokenizer = lambda text: emoji_pattern.findall(text),
                        preprocessor = lambda message: preprocess(message),
                        analyzer = 'word', max_df = 0.75, min_df = 5, max_features = None,
                        binary = False, ngram_range = (1,1))
        train = vectoriser.fit_transform(input_train)
        test = vectoriser.transform(input_test)
        emoji_file = pd.read_csv('scatter_results_sent.js', sep='\t')
        emoji_pols = pd.Series(emoji_file[feature_type].values,index=emoji_file.Emoji).to_dict()
        emoji_array = vectoriser.get_feature_names()
        idx = 0
        for item in emoji_array:
            if float(emoji_pols[emoji_array[idx]])!=np.nan:
                train[:, idx] = train[:, idx]*float(emoji_pols[emoji_array[idx]])
                test[:, idx] = test[:, idx]*float(emoji_pols[emoji_array[idx]])
            idx+=1
        train.data = np.nan_to_num(train.data)
        test.data = np.nan_to_num(test.data)
    elif feature_type=='Polarity1_unibi':
        feature_type = 'Polarity1'
        with open('./emoji'+'_'+'vectoriser.pkl','rb') as fid:
            emoji_vectoriser = cp.load(fid)
        etrain = emoji_vectoriser.transform(input_train)
        etest = emoji_vectoriser.transform(input_test)
        with open('./unibi'+'_'+'vectoriser.pkl','rb') as fid:
            vectoriser = cp.load(fid)
        utrain = vectoriser.transform(input_train)
        utest = vectoriser.transform(input_test)
        emoji_file = pd.read_csv('scatter_results_sent.js', sep='\t')
        emoji_pols = pd.Series(emoji_file[feature_type].values,index=emoji_file.Emoji).to_dict()
        emoji_array = emoji_vectoriser.get_feature_names()
        idx = 0
        for item in emoji_array:
            if float(emoji_pols[emoji_array[idx]])!=np.nan:
                etrain[:, idx] = etrain[:, idx]*float(emoji_pols[emoji_array[idx]])
                etest[:, idx] = etest[:, idx]*float(emoji_pols[emoji_array[idx]])
            idx+=1
        train = hstack([etrain,utrain])
        test = hstack([etest,utest])
        train.data = np.nan_to_num(train.data)
        test.data = np.nan_to_num(test.data)
    elif feature_type=='Polarity3_unibi':
        feature_type = 'Polarity3'
        with open('./emoji'+'_'+'vectoriser.pkl','rb') as fid:
            emoji_vectoriser = cp.load(fid)
        etrain = emoji_vectoriser.transform(input_train)
        etest = emoji_vectoriser.transform(input_test)
        with open('./unibi'+'_'+'vectoriser.pkl','rb') as fid:
            vectoriser = cp.load(fid)
        utrain = vectoriser.transform(input_train)
        utest = vectoriser.transform(input_test)
        emoji_file = pd.read_csv('scatter_results_sent.js', sep='\t')
        emoji_pols = pd.Series(emoji_file[feature_type].values,index=emoji_file.Emoji).to_dict()
        emoji_array = emoji_vectoriser.get_feature_names()
        idx = 0
        for item in emoji_array:
            if float(emoji_pols[emoji_array[idx]])!=np.nan:
                etrain[:, idx] = etrain[:, idx]*float(emoji_pols[emoji_array[idx]])
                etest[:, idx] = etest[:, idx]*float(emoji_pols[emoji_array[idx]])
            idx+=1
        train = hstack([etrain,utrain])
        test = hstack([etest,utest])
        train.data = np.nan_to_num(train.data)
        test.data = np.nan_to_num(test.data)
    return train, test

def classify_texts(data_type = 'emoji_unibi'):
    clf = data_type.split("-")[1]
    data_type = data_type.split("-")[0]
    df = pd.DataFrame()
    for file_index in range(9):
        df = pd.concat([df,pd.read_pickle(files_dir+'emoji_messages_'+str(file_index+1)+'.pkl')], ignore_index=True)
    print('Files are read!')
    df_sent = df[df.sentiment!='NA']
    df_sent.sentiment = df_sent.sentiment.map({'Bullish': 0, 'Bearish': 1})
    df_sent.reset_index(inplace=True)

    if clf=='LogSig':
        param_set = {"C": 1,
                 "penalty": 'l2',
                 "random_float":random.uniform(0,100)}
        classifier = classifier_fun(clf, param_set)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        splitted_indices = skf.split(np.zeros(df_sent.shape[0]), df_sent.sentiment)
        with open('./results/'+clf+'_'+data_type+'.csv', 'w') as results_file:
            for train_idx, test_idx in splitted_indices:
                train, test = featurise_input(df_sent.loc[train_idx, 'msgBody'], df_sent.loc[test_idx, 'msgBody'], feature_type=data_type)
                classifier.fit(train, df_sent.sentiment.loc[train_idx])
                print("Classifier is fit now!!!")
                with open('./'+clf+'_'+data_type+'_'+'classifier.pkl','wb') as fid:
                    cp.dump(classifier, fid)
                y_pred = classifier.predict_proba(test)
                y_score = y_pred[:,1]
                import ipdb; ipdb.set_trace()
                y_pred = np.argmax(y_pred,axis = 1)
                accuracy = accuracy_score(df_sent.sentiment.loc[test_idx],y_pred)
                f1 = f1_score(df_sent.sentiment.loc[test_idx], y_pred)
                precision = precision_score(df_sent.sentiment.loc[test_idx], y_pred)
                recall = recall_score(df_sent.sentiment.loc[test_idx], y_pred)
                roc = roc_auc_score(df_sent.sentiment.loc[test_idx], y_score)
                mcc = matthews_corrcoef(df_sent.sentiment.loc[test_idx], y_pred)
                print(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc))
                results_file.write(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc)+'\n')
    elif clf=='Random':
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        splitted_indices = skf.split(np.zeros(df_sent.shape[0]), df_sent.sentiment)
        with open('./results/'+clf+'_'+data_type+'.csv', 'w') as results_file:
            for train_idx, test_idx in splitted_indices:
                print("Classifier is fit now!!!")
                y_score = np.random.rand(test_idx.shape[0], 1)
                y_pred = np.concatenate((1-y_score, y_score),axis = 1)
                y_pred = np.argmax(y_pred,axis = 1)
                accuracy = accuracy_score(df_sent.sentiment.loc[test_idx],y_pred)
                f1 = f1_score(df_sent.sentiment.loc[test_idx], y_pred)
                precision = precision_score(df_sent.sentiment.loc[test_idx], y_pred)
                recall = recall_score(df_sent.sentiment.loc[test_idx], y_pred)
                roc = roc_auc_score(df_sent.sentiment.loc[test_idx], y_score)
                mcc = matthews_corrcoef(df_sent.sentiment.loc[test_idx], y_pred)
                print(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc))
                results_file.write(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc)+'\n')

    elif clf=='RFC':
        param_set = {'n_estimators': 10, 'max_features': 'auto',
                        'max_depth': None, 'min_samples_split': 2,
                        'min_samples_leaf': 1, 'bootstrap': True,
                        'criterion': 'gini'}
        classifier = classifier_fun(clf, param_set)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        splitted_indices = skf.split(np.zeros(df_sent.shape[0]), df_sent.sentiment)
        with open('./results/'+clf+'_'+data_type+'.csv', 'w') as results_file:
            for train_idx, test_idx in splitted_indices:
                train, test = featurise_input(df_sent.loc[train_idx, 'msgBody'], df_sent.loc[test_idx, 'msgBody'], feature_type=data_type)
                classifier.fit(train, df_sent.sentiment.loc[train_idx])
                print("Classifier is fit now!!!")
                with open('./'+clf+'_'+data_type+'_'+'classifier.pkl','wb') as fid:
                    cp.dump(classifier, fid)
                y_pred = classifier.predict_proba(test)
                y_score = y_pred[:,1]
                y_pred = np.argmax(y_pred,axis = 1)
                accuracy = accuracy_score(df_sent.sentiment.loc[test_idx],y_pred)
                f1 = f1_score(df_sent.sentiment.loc[test_idx], y_pred)
                precision = precision_score(df_sent.sentiment.loc[test_idx], y_pred)
                recall = recall_score(df_sent.sentiment.loc[test_idx], y_pred)
                roc = roc_auc_score(df_sent.sentiment.loc[test_idx], y_score)
                mcc = matthews_corrcoef(df_sent.sentiment.loc[test_idx], y_pred)
                print(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc))
                import ipdb; ipdb.set_trace()
                results_file.write(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc)+'\n')

    elif clf=='LinearSVC':
        param_set = {"C": 1.0,#random.choice([0.001, 0.01, 0.1, 1, 10, 100, 1000]),
                 "penalty": 'l2',#random.choice(['l1','l2']),
                 "loss": 'squared_hinge',#random.choice(['hinge','squared_hinge']),
                  "random_float":random.uniform(0,100)}
        if param_set['loss']=='squared_hinge': param_set['penalty']='l2'
        classifier = classifier_fun(clf, param_set)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        splitted_indices = skf.split(np.zeros(df_sent.shape[0]), df_sent.sentiment)
        with open('./results/'+clf+'_'+data_type+'.csv', 'w') as results_file:
            for train_idx, test_idx in splitted_indices:
                train, test = featurise_input(df_sent.loc[train_idx, 'msgBody'], df_sent.loc[test_idx, 'msgBody'], feature_type=data_type)
                classifier.fit(train, df_sent.sentiment.loc[train_idx])
                with open('./'+clf+'_'+data_type+'_'+'classifier.pkl','wb') as fid:
                    cp.dump(classifier, fid)
                print("Classifier is fit now!!!")
                y_pred = classifier.predict_proba(test)
                y_score = y_pred[:,1]
                y_pred = np.argmax(y_pred,axis = 1)
                accuracy = accuracy_score(df_sent.sentiment.loc[test_idx],y_pred)
                f1 = f1_score(df_sent.sentiment.loc[test_idx], y_pred)
                precision = precision_score(df_sent.sentiment.loc[test_idx], y_pred)
                recall = recall_score(df_sent.sentiment.loc[test_idx], y_pred)
                roc = roc_auc_score(df_sent.sentiment.loc[test_idx], y_score)
                mcc = matthews_corrcoef(df_sent.sentiment.loc[test_idx], y_pred)
                print(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc))
                results_file.write(clf+';'+data_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc)+'\n')

def calc_pol(x, pattern, pol_dict):
    return sum([pol_dict[item] for item in pattern.findall(x)])

def emolexiclassifier(polarity_type = 'Polarity1'):
    emoji_file = pd.read_csv('scatter_results_sent.js', sep='\t').fillna(0)
    emoji_pols = pd.Series(emoji_file[polarity_type].values,index=emoji_file.Emoji).to_dict()
    df = pd.DataFrame()
    for file_index in range(9):
        df = pd.concat([df,pd.read_pickle(files_dir+'emoji_messages_'+str(file_index+1)+'.pkl')], ignore_index=True)
    print('Files are read!')
    df_sent = df[df.sentiment!='NA'].reset_index()
    df_sent.sentiment = df_sent.sentiment.map({'Bullish': 0, 'Bearish': 1})
    df_sent[polarity_type] = df_sent.msgBody.apply(calc_pol, pattern = emoji_pattern, pol_dict = emoji_pols)
    df_sent['PredClass'] = df_sent[polarity_type].apply(lambda x: 0 if x>=0 else 1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    splitted_indices = skf.split(np.zeros(df_sent.shape[0]), df_sent.sentiment)
    with open('emolexicon'+'_'+polarity_type+'.csv', 'w') as results_file:
        for train_idx, test_idx in splitted_indices:
            accuracy = accuracy_score(df_sent.PredClass[test_idx],df_sent.sentiment[test_idx])
            f1 = f1_score(df_sent.PredClass[test_idx],df_sent.sentiment[test_idx])
            precision = precision_score(df_sent.PredClass[test_idx],df_sent.sentiment[test_idx])
            recall = recall_score(df_sent.PredClass[test_idx],df_sent.sentiment[test_idx])
            # roc = roc_auc_score(df_sent.PredClass[test_idx],df_sent.sentiment[test_idx])
            mcc = matthews_corrcoef(df_sent.PredClass[test_idx],df_sent.sentiment[test_idx])
            print('emolexicon'+';'+polarity_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(mcc))
            results_file.write('emolexicon'+';'+polarity_type+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(mcc)+'\n')

def prepare_tweet_vector_averages(tweets, w2v_type = 'domain-general', emoji = True):
    """Take the vector sum of all tokens in each tweet

    Args:
        tweets: All tweets
        p2v: Phrase2Vec model

    Returns:
        Average vectors for each tweet
        Truth
    """

    avg_vecs = np.zeros((len(tweets), 300), dtype=np.float64)
    y = list()
    if w2v_type=='domain-general':
        w2v = read_word_vectors(word_vector_path='./word2vec/GoogleNews-vectors-negative300.txt')
        e2v = read_word_vectors(word_vector_path='./word2vec/emoji2vec.txt')
        if emoji=='True':
            for i in range(len(tweets)):
                tokens = [token for token in my_tokenizer(preprocess(tweets[i]), emoji=False) if token in w2v.keys()]
                emojis = [emoji for emoji in emoji_pattern.findall(tweets[i]) if emoji in e2v.keys()]
                if len(tokens)<1 and len(emojis)<1:
                    continue
                avg_vecs[i, :] = np.mean([w2v[x] for x in tokens] + [e2v[x] for x in emojis], axis=0)
        else:
            for i in range(len(tweets)):
                tokens = [token for token in my_tokenizer(preprocess(tweets[i]), emoji=False) if token in w2v.keys()]
                if len(tokens)<1:
                    continue
                avg_vecs[i, :] = np.mean([w2v[x] for x in tokens], axis=0)
    elif w2v_type=='domain-specific':
        w2v = read_word_vectors(word_vector_path='./word2vec/word2vec.sgns.300d.nonpadded2018.txt')
        if emoji=='True':
            for i in range(len(tweets)):
                tokens = [token for token in my_tokenizer(preprocess(tweets[i]), emoji=False) if token in w2v.keys()]
                emojis = [emoji for emoji in emoji_pattern.findall(tweets[i]) if emoji in w2v.keys()]
                if len(tokens)<1 and len(emojis)<1:
                    continue
                avg_vecs[i, :] = np.mean([w2v[x] for x in tokens] + [w2v[x] for x in emojis], axis=0)
        else:
            for i in range(len(tweets)):
                tokens = [token for token in my_tokenizer(preprocess(tweets[i]), emoji=False) if token in w2v.keys()]
                if len(tokens)<1:
                    continue
                avg_vecs[i, :] = np.mean([w2v[x] for x in tokens], axis=0)

    return np.nan_to_num(avg_vecs).astype(np.float64)


def classify_stocktwits_w2v(mixture = ('domain-general', True, 'LogSig')):
    w2v_type = mixture[0]; emoji = mixture[1]; clf = mixture[2]
    msgs = pd.read_csv('./data/m2v_'+w2v_type+'_'+str(emoji)+'.csv', sep = '\t')
    msgs = msgs.as_matrix()
    sentiment = msgs[:,-1].reshape((msgs.shape[0], ))
    msgs = msgs[:, :-1]
    print("Messages are tokenized and vectorised!")
    if clf=='LogSig':
        param_set = {"C": 1,
                 "penalty": 'l2',
                 "random_float":random.uniform(0,100)}
        classifier = classifier_fun(clf, param_set)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        splitted_indices = skf.split(np.zeros(msgs.shape[0]), sentiment)
        with open('./results/'+clf+'_'+w2v_type+'_'+str(emoji)+'.csv', 'w') as results_file:
            for train_idx, test_idx in splitted_indices:
                classifier.fit(msgs[train_idx,:], sentiment[train_idx])
                print("Classifier is fit now!!!")
                y_pred = classifier.predict_proba(msgs[test_idx,:])
                y_score = y_pred[:,1]
                y_pred = np.argmax(y_pred,axis = 1)
                accuracy = accuracy_score(sentiment[test_idx],y_pred)
                f1 = f1_score(sentiment[test_idx], y_pred)
                precision = precision_score(sentiment[test_idx], y_pred)
                recall = recall_score(sentiment[test_idx], y_pred)
                roc = roc_auc_score(sentiment[test_idx], y_score)
                mcc = matthews_corrcoef(sentiment[test_idx], y_pred)
                print(clf+';'+w2v_type+';'+str(emoji)+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc))

                results_file.write(clf+';'+w2v_type+';'+str(emoji)+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc)+'\n')
    elif clf=='RFC':
        param_set = {'n_estimators': 10, 'max_features': 'auto',
                        'max_depth': None, 'min_samples_split': 2,
                        'min_samples_leaf': 1, 'bootstrap': True,
                        'criterion': 'gini'}
        classifier = classifier_fun(clf, param_set)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        splitted_indices = skf.split(np.zeros(msgs.shape[0]), sentiment)
        with open('./results/'+clf+'_'+w2v_type+'_'+str(emoji)+'.csv', 'w') as results_file:
            for train_idx, test_idx in splitted_indices:
                classifier.fit(msgs[train_idx,:], sentiment[train_idx])
                print("Classifier is fit now!!!")
                y_pred = classifier.predict_proba(msgs[test_idx,:])
                y_score = y_pred[:,1]
                y_pred = np.argmax(y_pred,axis = 1)
                accuracy = accuracy_score(sentiment[test_idx],y_pred)
                f1 = f1_score(sentiment[test_idx], y_pred)
                precision = precision_score(sentiment[test_idx], y_pred)
                recall = recall_score(sentiment[test_idx], y_pred)
                roc = roc_auc_score(sentiment[test_idx], y_score)
                mcc = matthews_corrcoef(sentiment[test_idx], y_pred)
                print(clf+';'+w2v_type+';'+str(emoji)+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc))
                with open('./'+clf+'_'+w2v_type+'_'+str(emoji)+'_'+'classifier.pkl','wb') as fid:
                    cp.dump(classifier, fid)
                import ipdb; ipdb.set_trace()
                results_file.write(clf+';'+w2v_type+';'+str(emoji)+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc)+'\n')

    elif clf=='LinearSVC':
        param_set = {"C": 1.0,
                 "penalty": 'l2',
                 "loss": 'squared_hinge',
                  "random_float":random.uniform(0,100)}
        if param_set['loss']=='squared_hinge': param_set['penalty']='l2'
        classifier = classifier_fun(clf, param_set)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        splitted_indices = skf.split(np.zeros(msgs.shape[0]), sentiment)
        with open('./results/'+clf+'_'+w2v_type+'_'+str(emoji)+'.csv', 'w') as results_file:
            for train_idx, test_idx in splitted_indices:
                classifier.fit(msgs[train_idx,:], sentiment[train_idx])
                print("Classifier is fit now!!!")
                y_pred = classifier.predict_proba(msgs[test_idx,:])
                y_score = y_pred[:,1]
                y_pred = np.argmax(y_pred,axis = 1)
                accuracy = accuracy_score(sentiment[test_idx],y_pred)
                f1 = f1_score(sentiment[test_idx], y_pred)
                precision = precision_score(sentiment[test_idx], y_pred)
                recall = recall_score(sentiment[test_idx], y_pred)
                roc = roc_auc_score(sentiment[test_idx], y_score)
                mcc = matthews_corrcoef(sentiment[test_idx], y_pred)
                print(clf+';'+w2v_type+';'+str(emoji)+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc))
                with open('./'+clf+'_'+w2v_type+'_'+str(emoji)+'_'+'classifier.pkl','wb') as fid:
                    cp.dump(classifier, fid)
                results_file.write(clf+';'+w2v_type+';'+str(emoji)+';'+str(accuracy)+';'+str(f1)+';'+str(precision)+';'+str(recall)+';'+str(roc)+';'+str(mcc)+'\n')

emoji_pattern = re.compile(u"(%s)" % "|".join(map(re.escape,list_emojis()[0])), flags=re.UNICODE)

args = ArgumentParser()
