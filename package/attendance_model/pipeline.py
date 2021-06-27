
import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBRegressor

from attendance_model.config.core import config
from attendance_model.processing import feature_engineering as fe

freq_pipeline = Pipeline(
    [
        # ===== ADDITIONAL FEATURES =====
        (
            "date_features",
            fe.DatetimeVariableEstimator(
                date_variable=config.model_config.split_date
            ),
        ),
        (   "statistical_features",
            fe.StatisticalVariableEstimator(
                prevision='prevision',
                effectif='effectif'
            )
        )

        # ===== IMPUTATION =====
        # impute prevision variable with interpolate method
        (
            "interpolate_imputation",
            fe.InterpolateImputer(
                prevision=config.model_config.num_vars_with_na_interpolate,
            ),
        ),
        (
            "median_imputation",
            fe.NumericalImputer( 
                variables=config.model_config.numerical_vars_with_na,
            ),
        ),
        (
            "categ_imputation",
            ce.target_encoder.TargetEncoder(
                cols=config.model_config.categorical_vars_with_na,
                smoothing=10,
            ),
        ),
        
        # ===== SCALING =====
        ("scaler", StandardScaler()),

        # ===== MODELING =====
        (
            "LGBR",
            LGBRegressor(
                max_depth=config.model_config.max_depth,
                n_estimators=config.model_config.n_estimators,
                num_leaves=config.model_config.num_leaves,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)