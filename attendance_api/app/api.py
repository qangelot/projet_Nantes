import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from attendance_model import __version__ as model_version
from attendance_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """ Returns the status of the service. """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """ Make Canteen Attendance predictions with the attendance-model package. """

    # the input data is following the MultipleDataInputs schema defined in the package
    # jsonable_encoder handles loading the pydantic data into json format
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # todo: improving API performance by rewriting the
    # make prediction function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")

    # pydantic does not handle np.nan, so we replace them by None
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    # if errors we throw a 400 (with details) to the client
    if results["errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return results