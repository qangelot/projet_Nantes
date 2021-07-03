import os
import sqlite3 as sql
import typing as t

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from attendance_model import __version__ as _version
from attendance_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Load the necessary data from the datawarehouse."""

    conn = sql.connect(os.path.join(DATASET_DIR / file_name))

    SQL_Query = pd.read_sql_query(
        """select
    Frequentation_quotidienne.date,
    Frequentation_quotidienne.prevision,
    Frequentation_quotidienne.reel,
    Dim_site.cantine_nom,
    Dim_site.annee_scolaire,
    Dim_site.effectif,
    Dim_site.quartier_detail,
    Dim_site.prix_quartier_detail_m2_appart,
    Dim_site.prix_moyen_m2_appartement,
    Dim_site.prix_moyen_m2_maison,
    Dim_site.longitude,
    Dim_site.latitude,
    Dim_temporelle.depuis_vacances,
    Dim_temporelle.depuis_ferie,
    Dim_temporelle.depuis_juives,
    Dim_temporelle.ramadan_dans,
    Dim_temporelle.depuis_ramadan,
    Dim_evenement.greve

    from Frequentation_quotidienne

    left join Dim_site               on Frequentation_quotidienne.site_id = Dim_site.site_id
    left join Dim_menu               on Frequentation_quotidienne.jour_id = Dim_menu.jour_id
    left join Dim_temporelle         on Frequentation_quotidienne.jour_id = Dim_temporelle.jour_id
    left join Dim_evenement          on Frequentation_quotidienne.jour_id = Dim_evenement.jour_id

    order by Frequentation_quotidienne.jour_site_id
    """,
        conn,
    )

    dataframe = pd.DataFrame(
        SQL_Query,
        columns=[
            "date",
            "prevision",
            "reel",
            "cantine_nom",
            "annee_scolaire",
            "effectif",
            "quartier_detail",
            "prix_quartier_detail_m2_appart",
            "prix_moyen_m2_appartement",
            "prix_moyen_m2_maison",
            "longitude",
            "latitude",
            "depuis_vacances",
            "depuis_ferie",
            "depuis_juives",
            "ramadan_dans",
            "depuis_ramadan",
            "greve",
        ],
    )

    return dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """
    Saves the versioned model, and overwrites any previous
    saved models.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    This is to ensure a simple one-to-one mapping between the package version
    and the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
