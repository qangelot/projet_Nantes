import numpy as np
import pandas as pd 
import sqlite3 as sql
from sqlalchemy import *


########################## Storing main dataframe into a staging db ############################
# this allows data to be persisted more reliably than in multiple CSVs

def main():
    # create a connector to the db
    staging_db = create_engine('sqlite:///data/staging_db.db')

    # drop table if exist
    conn = staging_db.raw_connection()
    cursor = conn.cursor()
    command = "DROP TABLE IF EXISTS {};".format('frequentation')
    cursor.execute(command)
    conn.commit()
    cursor.close()

    # create frequentation table and index on date for faster retrieval
    meta = MetaData()

    frequentation = Table('frequentation', meta,
                            Column('index', Integer, primary_key=True,
                                    autoincrement=True),
                            Column('site_type', String(5)),
                            Column('date', Date,  index=True),
                            Column('prevision', Integer),
                            Column('reel', Integer),
                            Column('cantine_nom', String(50)),
                            Column('annee_scolaire', String(30)),
                            Column('Effectif', Integer),
                            Column('vacances_dans', Integer),
                            Column('depuis_vacances', Integer),
                            Column('ferie_dans', Integer),
                            Column('depuis_ferie', Integer),
                            Column('chretiennes_dans', Integer),
                            Column('depuis_chretiennes', Integer),
                            Column('juives_dans', Integer),
                            Column('depuis_juives', Integer),
                            Column('ramadan_dans', Integer),
                            Column('depuis_ramadan', Integer),
                            Column('musulmanes_dans', Integer),
                            Column('depuis_musulmanes', Integer),
                            Column('chretiennes', Integer),
                            Column('juives', Integer),
                            Column('ramadan', Integer),
                            Column('musulmanes', Integer),
                            Column('greve', Integer),
                            Column('Quartier_detail', String(50)),
                            Column('prix_Quartier_detail_m2_appart', Float),
                            Column('prix_moyen_m2_appartement', Float),
                            Column('prix_moyen_m2_maison', Float),
                            Column('Longitude', Float),
                            Column('Latitude', Float),
                            Column('Plat', Text)
                        )

    frequentation.create(staging_db)

    # store data in the sql table 
    data = pd.read_csv('data/data.csv')
    data.to_sql('frequentation', staging_db, if_exists='append')

    print('Data loaded in staging database.')


if __name__ == "__main__":
    main()