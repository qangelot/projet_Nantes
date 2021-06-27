from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from attendance_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Check model inputs for na values and filter them
    based on the ones the model knows how to handle.
    """

    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.num_vars_with_na_interpolate
        + config.model_config.categorical_vars_with_na
        + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # pydantic cannot validate inputs with np.nan, so we replace them with None
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    date: Optional[str]
    prevision: Optional[int]
    cantine_nom: Optional[str]
    annee_scolaire: Optional[str]
    effectif: Optional[int]
    quartier_detail: Optional[str]
    prix_quartier_detail_m2_appart: Optional[int]
    prix_moyen_m2_appartement: Optional[int]
    prix_moyen_m2_maison: Optional[int]
    longitude: Optional[float]
    latitude: Optional[float]
    depuis_vacances: Optional[int]
    depuis_ferie: Optional[int]
    depuis_juives: Optional[int]
    ramadan_dans: Optional[int]
    depuis_ramadan: Optional[int]
    greve: Optional[bool]


class MultipleDataInputs(BaseModel):
    """In case, mutliple inputs are provided."""
    inputs: List[DataInputSchema]