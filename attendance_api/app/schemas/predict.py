from typing import Any, List, Optional

from pydantic import BaseModel
from attendance_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleDataInputs(BaseModel):
    # making use of the validation schema imported from the package
    # for example, if providing "prevision" with a number in a string format,
    # fastapi will convert it to integer under the hood
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "date": '2018-09-03 00:00:00',
                        "prevision": 189.0,
                        "cantine_nom": 'MAISDON PAJOT',
                        "annee_scolaire": '2018-2019',
                        "effectif": 201,
                        "quartier_detail": 'Zola',
                        "prix_quartier_detail_m2_appart": 3424.0,
                        "prix_moyen_m2_appartement": 3553.0,
                        "prix_moyen_m2_maison": 4490.0,
                        "longitude": 1.5848,
                        "latitude": 47.2183,
                        "depuis_vacances": 1,
                        "depuis_ferie": 19,
                        "depuis_juives": 43,
                        "ramadan_dans": 245,
                        "depuis_ramadan": 81,
                    }
                ]
            }
        }