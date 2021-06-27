import typing as t

import numpy as np
import pandas as pd

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
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:

        # removing strike days if any because we focus on predicting normal period
        validated_data = validated_data.loc[(validated_data["greve"] != 1)]
        validated_data.drop(["greve"], axis=1, inplace=True)

        predictions = _freq_pipeline.predict(X=validated_data)
        results = {
            "predictions": [pred ** 2 for pred in predictions],
            "version": _version,
            "errors": errors,
        }

    return results
