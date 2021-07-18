import sqlite3 as sql
import typing as t
import pandas as pd
import os


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Load the necessary data from the datawarehouse."""

    conn = sql.connect(file_name)

    SQL_Query = pd.read_sql_query(
        """select 
Frequentation_quotidienne.date, 
Frequentation_quotidienne.prevision, 
Frequentation_quotidienne.reel,
Dim_site.*,
Dim_menu.plats,
Dim_temporelle.vacances_dans, 
Dim_temporelle.depuis_vacances,
Dim_temporelle.ferie_dans, 
Dim_temporelle.depuis_ferie, 
Dim_temporelle.chretiennes_dans,
Dim_temporelle.depuis_chretiennes,
Dim_temporelle.juives_dans,
Dim_temporelle.depuis_juives, 
Dim_temporelle.ramadan_dans, 
Dim_temporelle.depuis_ramadan, 
Dim_temporelle.musulmanes_dans, 
Dim_temporelle.depuis_musulmanes,
Dim_evenement.chretiennes, 
Dim_evenement.juives,
Dim_evenement.ramadan, 
Dim_evenement.musulmanes, 
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
            'date', 'prevision', 'reel', 'site_type', 'cantine_nom',
            'annee_scolaire', 'effectif', 'quartier_detail',
            'prix_quartier_detail_m2_appart', 'prix_moyen_m2_appartement',
            'prix_moyen_m2_maison', 'longitude', 'latitude',
            'vacances_dans', 'depuis_vacances',
            'plats',
            'ferie_dans', 'depuis_ferie', 'chretiennes_dans', 'depuis_chretiennes',
            'juives_dans', 'depuis_juives', 'ramadan_dans', 'depuis_ramadan',
            'musulmanes_dans', 'depuis_musulmanes',
            'chretiennes', 'juives',
            'ramadan', 'musulmanes', 'greve'
        ],
    )

    return dataframe