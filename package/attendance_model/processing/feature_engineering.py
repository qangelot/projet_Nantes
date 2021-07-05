from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..config.core import config


class DatetimeVariableEstimator(BaseEstimator, TransformerMixin):
    """Derived features from date"""

    def __init__(self, date_variable: str):

        if not isinstance(date_variable, str):
            raise ValueError("date_variable should be a string of format 'yyyy-mm-dd' ")

        self.date_variable = date_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # so that we do not over-write the original dataframe
        X = X.copy()
        X[self.date_variable] = pd.to_datetime(X[self.date_variable])

        # creating date based features
        X["year"] = X[self.date_variable].dt.year
        X["day_of_week_sin"] = np.sin(
            2
            * np.pi
            * X[self.date_variable].dt.dayofweek
            / max(X[self.date_variable].dt.dayofweek)
        )
        X["day_of_year_sin"] = np.sin(
            2
            * np.pi
            * X[self.date_variable].dt.dayofyear
            / max(X[self.date_variable].dt.dayofyear)
        )
        X["day_of_year_cos"] = np.cos(
            2
            * np.pi
            * X[self.date_variable].dt.dayofyear
            / max(X[self.date_variable].dt.dayofyear)
        )
        X["week"] = X[self.date_variable].dt.isocalendar().week
        X.set_index("date", inplace=True)

        return X


class StatisticalVariableEstimator(BaseEstimator, TransformerMixin):
    """Derived statistical features"""

    def __init__(self, prevision: str, effectif: str):

        if not isinstance(prevision, str):
            raise ValueError("prevision should be a string")

        if not isinstance(effectif, str):
            raise ValueError("effectif should be a string")

        # while concatenating X and y, y column gets rename 0
        self.reel = config.model_config.target
        self.prevision = prevision
        self.effectif = effectif

    def fit(self, X: pd.DataFrame, y: pd.Series):
        
        # to avoid data leakage we compute the relevant statistics on train set only
        y.index = X.index
        self.y = y

        X = X.copy()
        train = pd.concat((X, self.y), axis=1)

        # aggregate data significantly (canteen, scholar year and week level) and compute statistics
        train["freq_predicted_%"] = train[self.prevision] / train[self.effectif]
        train["freq_reel_%"] = train[self.reel] / train[self.effectif]
        grouped_train = train.groupby(["cantine_nom", "week", "annee_scolaire"])
        grouped_train_m = (
            grouped_train[["freq_predicted_%", "freq_reel_%", self.prevision, self.reel]]
            .mean()
            .reset_index()
        )
        grouped_train_s = (
            grouped_train[["freq_predicted_%", "freq_reel_%", self.prevision, self.reel]]
            .std()
            .reset_index()
        )

        # then we average on the years, to have a single number per canteen and week
        # in order to be able to spread it to the test set
        agg_mean = grouped_train_m.groupby(["cantine_nom", "week"])
        self.agg_mean = agg_mean[["freq_predicted_%", "freq_reel_%"]].mean()
        agg_std = grouped_train_s.groupby(["cantine_nom", "week"])
        self.agg_std = agg_std[["freq_predicted_%", "freq_reel_%"]].mean()

        self.agg_std.rename(
            columns={
                "freq_predicted_%": "freq_predicted_%_std",
                "freq_reel_%": "freq_reel_%_std",
            },
            inplace=True,
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.merge(
            self.agg_mean, left_on=["cantine_nom", "week"], right_index=True, how="left"
        )

        X = X.merge(
            self.agg_std, left_on=["cantine_nom", "week"], right_index=True, how="left"
        )

        X.drop(["freq_predicted_%", "freq_predicted_%_std"], axis=1, inplace=True)
        X.reset_index(drop=True, inplace=True)

        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer"""

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):
            raise ValueError("Variables should be in a list")

        self.variables = variables

    def fit(self, X, y: pd.Series = None):
        # persist median in a dictionary
        self.imputer_dict_ = {}

        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].median()
        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
