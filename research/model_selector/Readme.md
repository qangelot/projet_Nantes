# Model-selection process 

Goal is to find what kind of model perform best for a given dataset (linear, instance based, trees, with kernel...)

This runs diverse models, with different sets of hyperparameters and evaluate them using cross-validation. This ensure a robust result.


# Categories

* Generalized Linear Models
* Kernel Ridge (allow linear models to learn non linear patterns)
* Support Vector Machines
* Nearest Neighbors
* Gaussian Processes
* Naive Bayes
* Trees 
* Neural Networks (basic ones)
* Ensemble methods


# Running it

1. Feed in X and y as numpy arrays or pandas dataframe.
2. One can import only sub functions that run a specific kind of model (e.g. linear, tree based...) or the run_all functions that runs every models.


# There is more

* Default metrics are accuracy and neg MSE, respectively for classification and regression (can be changed in the run_all function by setting any `sklearn` scoring metrics)

* Can use only a subset of hyperparameters for faster execution using the `small` parameter of the `run_all` functions. Small also remove some slow models on big datasets (such as SVR/SVC).
  
* Depending on the nature of the problem and the performed EDA, certain categories of models work better than others. There is sub methods in `run_regressors.py` and `run_classifiers.py` for each categories.



