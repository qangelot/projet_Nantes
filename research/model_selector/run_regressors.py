import warnings
warnings.filterwarnings('ignore')
from multiprocessing import cpu_count

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, \
    Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, \
    SGDRegressor, PassiveAggressiveRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from .utilities import *
from .universal_params import *


# specific params for different regression models

linear_models_n_params = [
    (LinearRegression, normalize),

    (Ridge,
     {**alpha, **normalize, **tol,
      'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
      }),

    (Lasso,
     {**alpha, **normalize, **tol, **warm_start
      }),

    (ElasticNet,
     {**alpha, **normalize, **tol,
      'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
      }),

    (Lars,
    {**normalize,
      'n_nonzero_coefs': [100, 300, 500, np.inf],
      }),

    (LassoLars,
     {**normalize, **max_iter_inf, **alpha
      }),

    (OrthogonalMatchingPursuit,
    {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
      **tol, **normalize
      }),

    (BayesianRidge,
    {
        'n_iter': [100, 300, 1000],
        **tol, **normalize,
        'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
        'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
        'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
        'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
    }),

    (SGDRegressor,
    {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
      **penalty_12e, **n_iter, **epsilon, **eta0,
      'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
      'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
      'learning_rate': ['constant', 'optimal', 'invscaling'],
      'power_t': [0.1, 0.25, 0.5]
      }),

    (PassiveAggressiveRegressor,
     {**C, **epsilon, **n_iter, **warm_start,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
      }),

    (HuberRegressor,
    { 'epsilon': [1.1, 1.35, 1.5, 2],
       **max_iter, **alpha, **warm_start, **tol
      }),

    (KernelRidge,
     {**alpha, **degree, **gamma, **coef0
      })
]

linear_models_n_params_small = [
    (LinearRegression, normalize),

    (Ridge,
     {**alpha_small, **normalize
      }),

    (Lasso,
     {**alpha_small, **normalize
      }),

    (ElasticNet,
     {**alpha, **normalize,
      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
      }),

    (Lars,
    {**normalize,
      'n_nonzero_coefs': [100, 300, 500, np.inf],
      }),

    (LassoLars,
     {**normalize, **max_iter_inf, **normalize, **alpha_small
      }),

    (OrthogonalMatchingPursuit,
    {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
      **normalize
      }),

    (BayesianRidge,
    { 'n_iter': [100, 300, 1000],
      'alpha_1': [1e-6, 1e-3],
      'alpha_2': [1e-6, 1e-3],
      'lambda_1': [1e-6, 1e-3],
      'lambda_2': [1e-6, 1e-3],
      **normalize
      }),

    (SGDRegressor,
    {'loss': ['squared_loss', 'huber'],
      **penalty_12e, **n_iter,
      'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
      }),

    (PassiveAggressiveRegressor,
     {**C, **n_iter,
      }),

    (HuberRegressor,
     { **max_iter, **alpha_small
      }),

    (KernelRidge,
     {**alpha_small, **degree
      })
]

svm_models_n_params_small = [

    (LinearSVR,
     {**C_small, **epsilon,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
      'intercept_scaling': [0.1, 1, 10]
      })
]

svm_models_n_params = [
    (SVR,
     {**C, **epsilon, **kernel, **degree, **gamma, **tol
      }),

    (NuSVR,
     {**C, **nu, **kernel, **degree, **gamma, **tol
      }),

    (LinearSVR,
     {**C, **epsilon, **tol,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
      'intercept_scaling': [0.1, 1, 5, 10]
      })
]

neighbor_models_n_params = [

    (RadiusNeighborsRegressor,
    {**neighbor_radius, **neighbor_algo, **neighbor_leaf_size, **neighbor_metric,
      'weights': ['uniform', 'distance'],
      }),

    (KNeighborsRegressor,
     {**n_neighbors, **neighbor_algo, **neighbor_leaf_size, **neighbor_metric,
      'weights': ['uniform', 'distance'],
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessRegressor,
    {'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      'n_restarts_optimizer': [3],
      'alpha': [1e-10, 1e-5],
      'normalize_y': [True, False]
      })
]

nn_models_n_params = [
    (MLPRegressor,
    { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 64)],
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      **alpha, **learning_rate, **tol, **warm_start,
      'batch_size': ['auto', 50],
      'max_iter': [1000],
      'early_stopping': [True, False],
      'epsilon': [1e-8, 1e-5]
      })
]

nn_models_n_params_small = [
    (MLPRegressor,
    { 'hidden_layer_sizes': [(64,), (32, 64)],
      'activation': ['identity', 'tanh', 'relu'],
      'max_iter': [500],
      'early_stopping': [True],
      **learning_rate_small
      })
]

tree_models_n_params = [
    (DecisionTreeRegressor,
     {**max_features, **max_depth, **min_samples_split, **min_samples_leaf, **min_impurity_split,
      'criterion': ['mse', 'mae']}),

    (ExtraTreesRegressor,
     {**n_estimators, **max_features, **max_depth, **min_samples_split, **min_samples_leaf, **min_impurity_split,
      'criterion': ['mse', 'mae']}),

    (RandomForestRegressor,
     {**n_estimators, **max_features, **max_depth, **min_samples_split,
      'criterion': ['mse', 'mae']})
]

tree_models_n_params_small = [
    (DecisionTreeRegressor,
      {**max_features_small, **max_depth_small, **min_samples_split, **min_samples_leaf,
      'criterion': ['mse', 'mae']}),

    (ExtraTreesRegressor,
     {**n_estimators_small, **max_features_small, **max_depth_small}),

    (RandomForestRegressor,
     {**n_estimators_small, **max_features_small, **max_depth_small})
]


def run_linear_models(x, y, small = True, normalize_x = True):
    return run_models(linear_models_n_params_small if small else linear_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_svm_models(x, y, small = True, normalize_x = True):
    return run_models(svm_models_n_params_small if small else svm_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_neighbor_models(x, y, normalize_x = True):
    return run_models(neighbor_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_gaussian_models(x, y, normalize_x = True):
    return run_models(gaussianprocess_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_nn_models(x, y, small = True, normalize_x = True):
    return run_models(nn_models_n_params_small if small else nn_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_tree_models(x, y, small = True, normalize_x = True):
    return run_models(tree_models_n_params_small if small else tree_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_tree_models(x, y, small = True, normalize_x = True):
    return run_models(tree_models_n_params_small if small else tree_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)


def run_all(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, scoring=None):

    all_params = (linear_models_n_params_small if small else linear_models_n_params) + \
                (nn_models_n_params_small if small else nn_models_n_params) + \
                ([] if small else gaussianprocess_models_n_params) + \
                neighbor_models_n_params + \
                (svm_models_n_params_small if small else svm_models_n_params) + \
                (tree_models_n_params_small if small else tree_models_n_params)

    return run_models(all_params, scoring=scoring,
                    StandardScaler().fit_transform(x) if normalize_x else x, y,
                    isClassification=False, n_jobs=n_jobs)
