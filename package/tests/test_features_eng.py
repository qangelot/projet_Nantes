import math

from attendance_model.config.core import config
from attendance_model.processing.feature_engineering import DatetimeVariableEstimator, StatisticalVariableEstimator, NumericalImputer


def test_datetime_variable_estimator(sample_input_data):

    transformer = DatetimeVariableEstimator(date_variable="date")

    # testing that the first data point from test set is the one we expect
    assert str(sample_input_data["date"].iat[0]) == "2018-09-03 00:00:00"

    # then transforming the test data date variable
    temp = transformer.fit_transform(sample_input_data)

    # finally, testing that the transformations for that data point are the ones we expect
    assert temp["year"].iat[0] == 2018
    assert temp["day_of_week_sin"].iat[0] == 0.0
    assert math.isclose(temp["day_of_year_sin"].iat[0], -0.932097, abs_tol=0.1)
    assert math.isclose(temp["day_of_year_cos"].iat[0], -0.36221, abs_tol=0.1) 
    assert temp["week"].iat[0] == 36


def test_statistical_variable_estimator(sample_input_data):

    # this test required some variable generated in DatetimeVariableEstimator
    transformer = DatetimeVariableEstimator(date_variable="date")
    transformer_stats = StatisticalVariableEstimator(prevision="prevision", effectif="effectif")

    # that class required y_test to be passed explicitly 
    # while concatenating X and y in conftest.py, y column gets rename 0
    temp = transformer.fit_transform(sample_input_data)    

    y_test = temp[0]
    X_test = temp.drop([0], axis=1)

    # testing that the first data point from test set is the one we expect
    assert X_test["prevision"].iat[0] == 126.0
    assert X_test["effectif"].iat[0] == 121


def test_numerical_imputer(sample_input_data):

    imputer = NumericalImputer(variables=['prix_quartier_detail_m2_appart', 'prix_moyen_m2_appartement', 
    'prix_moyen_m2_maison', 'longitude', 'latitude'])

    # testing that the first data point from test set is the one we expect
    assert sample_input_data["prix_quartier_detail_m2_appart"].iat[0] == 4200.0
    assert sample_input_data["prix_moyen_m2_appartement"].iat[0] == 4363.0
    assert sample_input_data["prix_moyen_m2_maison"].iat[0] == 5869.0	
    assert math.isclose(sample_input_data["longitude"].iat[0], 1.5596, abs_tol=0.001)
    assert math.isclose(sample_input_data["latitude"].iat[0], 47.2194	, abs_tol=0.001)

    # then transforming the test data date variable
    temp = imputer.fit_transform(sample_input_data)

    # finally, testing that the imputation works as expected
    assert temp[['prix_quartier_detail_m2_appart', 'prix_moyen_m2_appartement', 
    'prix_moyen_m2_maison', 'longitude', 'latitude']].isna().sum().sum() == 0