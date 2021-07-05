import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:

    payload = {
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # testing api response code, presence of preds, 
    # absence of errors and value of first pred
    assert response.status_code == 200
    preds = response.json()
    assert preds["predictions"]
    assert preds["errors"] is None
    assert math.isclose(preds[0], 116.99946577306487, abs_tol=1)