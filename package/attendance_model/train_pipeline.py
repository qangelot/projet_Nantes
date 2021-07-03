import pandas as pd

from attendance_model.config.core import config
from attendance_model.pipeline import freq_pipeline
from attendance_model.processing.data_manager import load_dataset, save_pipeline


def timeseries_train_test_split(data, split_date):
    """
    Perform train-test split with respect to time series structure
    """

    split_date = pd.to_datetime(split_date)
    data["date"] = pd.to_datetime(data["date"])

    # create train test partition
    X_train = data.loc[data["date"] < split_date]
    X_test = data.loc[data["date"] >= split_date]

    y_train = X_train.reel
    X_train = X_train.drop(["reel"], axis=1)
    y_test = X_test.reel
    X_test = X_test.drop(["reel"], axis=1)

    return X_train, y_train, X_test, y_test


def zscore_outliers(data, column, n):
    """
    Tag outliers in dataset in regard from a specific column
    As an empirical rule, often 3 std from the mean is consider as an outlier
    Return two boolean columns : upper_outlier and lower_outlier
    """

    outliers = data[(data[column] != 0)]

    # we aggregate value by cantine and annee in order to have
    # a meaningfull outlier detection system and compare what's comparable
    outliers = outliers.groupby(["cantine_nom", "annee_scolaire"])
    outliers = outliers[column].agg(["mean", "std"])

    outliers["lower_bound"] = outliers["mean"] - (n * outliers["std"])
    outliers["upper_bound"] = outliers["mean"] + (n * outliers["std"])

    data = data.merge(
        outliers,
        left_on=["cantine_nom", "annee_scolaire"],
        right_index=True,
        how="left",
    )

    data["upper_outlier"] = data[column] > data["upper_bound"]
    data["lower_outlier"] = data[column] < data["lower_bound"]

    data.drop(["upper_bound", "lower_bound", "mean", "std"], axis=1, inplace=True)

    return data


def run_training() -> None:
    """Train the attendance model."""

    # read data
    data = load_dataset(file_name=config.app_config.data_file)

    # here i consider as an outlier any data point that sits
    # 2 std away from the mean, this rigourous choice is based on the fact
    # that Nantes metropole wants a model to predict "normal periods" first
    # most of the zeros are due to mistakes and are not meaningful values
    # strikes are not predictible by nature
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

    # transform target
    y_train = y_train ** (1 / 2)

    # fit model
    freq_pipeline.fit(X_train, y_train)

    print(f"Score: {freq_pipeline.score(X_train, y_train)}")

    # persist trained model
    save_pipeline(pipeline_to_persist=freq_pipeline)


if __name__ == "__main__":
    run_training()
