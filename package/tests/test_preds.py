import math

import numpy as np

from attendance_model.predict import make_prediction


def test_make_prediction(sample_input_data):

    res = make_prediction(input_data=sample_input_data)

    preds = res.get("predictions")

    # testing expected types, absence of errors,
    # len of preds and value of first pred
    assert isinstance(preds, list)
    assert isinstance(preds[0], np.float64)
    assert res.get("errors") is None
    assert len(preds) == 12077
    assert math.isclose(preds[0], 116.99946577306487, abs_tol=1)
