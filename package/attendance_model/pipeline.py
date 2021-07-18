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
            fe.TargetEncoder(
                variables=config.model_config.categorical_vars_with_na,
            ),
        ),
        # ===== SCALING =====
        ("scaler", StandardScaler()),
        # ===== MODELING =====
        (
            "LGBR",
            LGBMRegressor(
                alpha=config.model_config.alpha,
                objective=config.model_config.objective,
                metric=config.model_config.metric,
                max_depth=config.model_config.max_depth,
                n_estimators=config.model_config.n_estimators,
                num_leaves=config.model_config.num_leaves,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
