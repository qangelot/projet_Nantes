import typing as t

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

from attendance_model import __version__ as _version
from attendance_model.config.core import config
from attendance_model.processing.data_manager import load_pipeline
from attendance_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_freq_pipeline = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data) 
    if "reel" in data.columns:
        y_test = data.reel

    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:

        predictions = _freq_pipeline.predict(X=validated_data)
        results = {
            "predictions": [pred ** 2 for pred in predictions],
            "version": _version,
            "errors": errors,
        }

    try:
        print(f"Root Mean Square Error: {str(np.sqrt(mean_squared_error(y_test, results['predictions'])))}")
        print(f"Mean Absolute Error: {str(mean_absolute_error(y_test, results['predictions']))}")
        print(f"Median Absolute Error: {str(median_absolute_error(y_test, results['predictions']))}")

    except:
        pass

    return results
