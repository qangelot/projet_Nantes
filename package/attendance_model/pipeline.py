import category_encoders as ce
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from attendance_model.config.core import config
from attendance_model.processing import feature_engineering as fe

freq_pipeline = Pipeline(
    [
        # ===== ADDITIONAL FEATURES =====
        (
            "date_features",
            fe.DatetimeVariableEstimator(date_variable="date"),
        ),
        (
            "statistical_features",
            fe.StatisticalVariableEstimator(prevision="prevision", effectif="effectif"),
        ),
        # ===== IMPUTATION =====
        # impute prevision variable with interpolate method
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
            LGBMRegressor(
                max_depth=config.model_config.max_depth,
                n_estimators=config.model_config.n_estimators,
                num_leaves=config.model_config.num_leaves,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
