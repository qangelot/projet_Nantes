from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from attendance_model.config.core import config
from attendance_model.processing.data_manager import load_dataset
from attendance_model.train_pipeline import timeseries_train_test_split, zscore_outliers

from ..main import app


@pytest.fixture(scope="module")
def test_data()-> pd.DataFrame:

    # read data
    data = load_dataset(file_name=config.app_config.data_file)
    
    # removing outliers that aren't bringing valuable information in regard
    # to the framing of the problem
    data = zscore_outliers(data, config.model_config.target, 2)
    data = data.loc[
        (data["greve"] != 1)
        & (data["reel"] != 0)
        & (data["upper_outlier"] != 1)
        & (data["lower_outlier"] != 1)
    ]
    data.drop(["greve", "upper_outlier", "lower_outlier"], axis=1, inplace=True)

    data = data.dropna(axis=0, subset=[config.model_config.target])

    # divide train and test
    X_train, y_train, X_test, y_test = timeseries_train_test_split(
        data, split_date=config.model_config.split_date
    )

    test = pd.concat((X_test, y_test), axis=1)
    
    return test


# pattern to set up a fastapi test client to allow for api calls
@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}