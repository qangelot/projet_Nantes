import numpy as np


# generic parameters common across multiple models

# linear
penalty_12 = {'penalty': ['l1', 'l2']}
penalty_12none = {'penalty': ['l1', 'l2', None]}
penalty_12e = {'penalty': ['l1', 'l2', 'elasticnet']}
penalty_all = {'penalty': ['l1', 'l2', None, 'elasticnet']}
max_iter = {'max_iter': [100, 300, 1000]}
max_iter_inf = {'max_iter': [100, 300, 500, 1000, np.inf]}
max_iter_inf2 = {'max_iter': [100, 300, 500, 1000, -1]}
tol = {'tol': [1e-4, 1e-3, 1e-2]}
warm_start = {'warm_start': [True, False]}
alpha = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]}
alpha_small = {'alpha': [1e-5, 1e-2, 0.1, 1]}
n_iter = {'n_iter': [5, 10, 20]}

# kernel
eta0 = {'eta0': [1e-4, 1e-3, 1e-2, 0.1]}
C = {'C': [1e-2, 0.1, 1, 5, 10]}
C_small = {'C': [ 0.1, 1, 5]}
epsilon = {'epsilon': [1e-3, 1e-2, 0.1, 0]}
normalize = {'normalize': [True, False]}
kernel = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
degree = {'degree': [1, 2, 3, 4, 5]}
gamma = {'gamma': list(np.logspace(-9, 3, 6)) + ['auto']}
gamma_small = {'gamma': list(np.logspace(-6, 3, 3)) + ['auto']}
coef0 = {'coef0': [0, 0.1, 0.3, 0.6, 1]}
coef0_small = {'coef0': [0, 0.4, 0.7, 1]}
shrinking = {'shrinking': [True, False]}
nu = {'nu': [1e-4, 1e-2, 0.1, 0.4, 0.7, 0.9]}
nu_small = {'nu': [1e-2, 0.1, 0.5, 0.9]}

# neighbors
n_neighbors = {'n_neighbors': [5, 10, 20]}
neighbor_algo = {'algorithm': ['ball_tree', 'kd_tree']}
neighbor_leaf_size = {'leaf_size': [ 2, 10, 50, 100]}
neighbor_metric = {'metric': ['euclidean', 'l1', 'manhattan']}
neighbor_radius = {'radius': [1e-2, 0.1, 1, 10]}

# trees
n_estimators = {'n_estimators': [25, 50, 100, 200, 500]}
n_estimators_small = {'n_estimators': [50, 200, 500]}
max_features = {'max_features': [10, 25, 'auto', 'log2', None]}
max_features_small = {'max_features': [15, 'auto', 'log2', None]}
max_depth = {'max_depth': [None, 3, 6, 10]}
max_depth_small = {'max_depth': [None, 5, 10]}
min_samples_split = {'min_samples_split': [2, 5, 10]}
min_impurity_split = {'min_impurity_split': [1e-6, 1e-4, 1e-3]}
tree_learning_rate = {'learning_rate' : [0.01, 0.1, 1]}
min_samples_leaf = {'min_samples_leaf': [2]}

# neural nets
learning_rate = {'learning_rate': ['constant', 'invscaling', 'adaptive']}
learning_rate_small = {'learning_rate': ['invscaling', 'adaptive']}


