import numpy as np
from collections import Counter
from multiprocessing import cpu_count
from time import time
from tabulate import tabulate
from pprint import pprint

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit as sss, ShuffleSplit as ss, TimeSeriesSplit as ts, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


TREE_N_ENSEMBLE_MODELS = [RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, DecisionTreeClassifier, DecisionTreeRegressor,
                        ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor]

def oversample_indices_clf(inds, y):
    """
    Balance number of samples for all classes (in case of a classification problem)
    ->type inds: numpy array
    ->type y: numpy array

    ->return: a numpy array of indices
    """

    assert len(inds) == len(y)

    countByClass = dict(Counter(y))
    maxCount = max(countByClass.values())

    extras = []

    for class_, count in countByClass.items():
        if maxCount == count: continue

        ratio = int(maxCount / count)
        current_inds = inds[y == class_]

        extras.append(np.concatenate(
            (np.repeat(current_inds, ratio - 1),
             np.random.choice(current_inds, maxCount - ratio * count, replace=False)
            ))
        )

        print(f'Oversampling class {class_}, {ratio-1} times')

    return np.concatenate((inds, *extras))


def cv_clf(x, y, test_size = 0.2, n_splits = 5, random_state=None, Oversample = True):
    """
    an iterator of cross-validation groups with up sampling
    ->param x
    ->param y
    ->param test_size
    ->param n_splits
    """

    # StratifiedShuffleSplit
    sss_obj = sss(n_splits, test_size, random_state=random_state).split(x, y)

    # no oversampling needed
    if not Oversample:
        return sss_obj

    # with oversampling
    for train_inds, valid_inds in sss_obj:
        yield (oversample_indices_clf(train_inds, y[train_inds]), valid_inds)


def cv_reg(x, test_size = 0.2, n_splits = 5, random_state=None):
    """ cross- validation for regression problems """
    # ShuffleSplit
    return ss(n_splits, test_size, random_state=random_state).split(x)


def timer(class_, params, x, y):

    start = time()
    clf = class_(**params)
    clf.fit(x, y)

    return time() - start

def run_models(models_n_params, x, y, isClassification,
            test_size = 0.2, n_splits = 4, random_state=42, Oversample=True,
            scoring=None, verbose=False, n_jobs = cpu_count()-1):
    """
    runs through all model classes with their perspective hyper parameters
    ->param models_n_params: [(model class, hyper parameters),...]
    ->param isClassification (bool): classification or regression problem
    ->param scoring: default is 'accuracy' for classification; 'neg_mean_squared_error' for regression
    ->return: the best estimator, list of [(estimator, cv score),...]
    """

    
    # def cv_():
    #     return cv_clf(x, y, test_size, n_splits, random_state, Oversample) if isClassification \
    #         else cv_reg(x, test_size, n_splits, random_state)
    

    # for a time series problem it's not possible to use a regular cv schema, 
    # using an instance of Sklearn TimeSeriesSplit instead
    cv_ts = ts(n_splits=n_splits)

    res = []
    num_features = x.shape[1]
    scoring = scoring or ('accuracy' if isClassification else 'neg_mean_squared_error')
    print('Scoring criteria:', scoring)

    for i, (clf_class_, parameters) in enumerate(models_n_params):
        try:
            print('-'*15, 'model %d/%d' % (i+1, len(models_n_params)), '-'*15)
            print(clf_class_.__name__)

            if clf_class_ == KMeans:
                parameters['n_clusters'] = [len(np.unique(y))]
            elif clf_class_ in TREE_N_ENSEMBLE_MODELS:
                # ensuring a valid max_features parameter has been provided
                parameters['max_features'] = [v for v in parameters['max_features']
                                                if v is None or type(v)==str or v<=num_features]

            # warning : do change cv parameter here
            clf_search = GridSearchCV(clf_class_(), parameters, scoring, cv=cv_ts, n_jobs=n_jobs)
            clf_search.fit(x, y)

            timespent = timer(clf_class_, clf_search.best_params_, x, y)
            print('best score:', clf_search.best_score_, 'time/clf: %0.3f seconds' % timespent)
            print('best params:')
            pprint(clf_search.best_params_)

            if verbose:
                print('validation scores:', clf_search.cv_results_['mean_test_score']) 
                print('training scores:', clf_search.cv_results_['mean_train_score'])

            res.append((clf_search.best_estimator_, clf_search.best_score_, timespent))

        except Exception as e:
            print('ERROR OCCURRED')
            if verbose: print(e)
            res.append((clf_class_(), -np.inf, np.inf))


    print('='*60)
    print(tabulate([[m.__class__.__name__, '%.3f'%s, '%.3f'%t] for m, s, t in res], headers=['Model', scoring, 'Time/clf (s)']))
    winner_ind = np.argmax([v[1] for v in res])
    winner = res[winner_ind][0]
    print('='*60)
    print('The winner is: %s with score %0.3f.' % (winner.__class__.__name__, res[winner_ind][1]))

    return winner, res
