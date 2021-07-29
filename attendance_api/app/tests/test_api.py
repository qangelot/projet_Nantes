import math
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:

    test_data['date'] = test_data['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
    inputs = test_data.replace({np.nan: None}).to_dict(orient="records")
    
    payload = {
                "inputs": [
                    {
                    "date": "2018-09-03 00:00:00",
                    "prevision": 189,
                    "cantine_nom": "MAISDON PAJOT",
                    "annee_scolaire": "2018-2019",
                    "effectif": 201,
                    "quartier_detail": "Zola",
                    "prix_quartier_detail_m2_appart": 3424,
                    "prix_moyen_m2_appartement": 3553,
                    "prix_moyen_m2_maison": 4490,
                    "longitude": 1.5848,
                    "latitude": 47.2183,
                    "depuis_vacances": 1,
                    "depuis_ferie": 19,
                    "depuis_juives": 43,
                    "ramadan_dans": 245,
                    "depuis_ramadan": 81
                    }
                ]
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
    assert math.isclose(preds["predictions"][0], 189, abs_tol=1)