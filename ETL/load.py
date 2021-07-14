import sqlite3 as sql
from sqlalchemy import *
import pandas as pd


########################## Bulding the analytical database ############################
# organizing the data from the staging db in order to answer the business needs

def main():
    # create a connector to the db
    conn = sql.connect('data/staging_db.db')
    data = pd.read_sql_query("SELECT * FROM frequentation", conn)
    data["date"] = pd.to_datetime(data["date"])

    data.rename(columns={"Effectif": "effectif", "Quartier_detail": "quartier_detail", "prix_Quartier_detail_m2_appart":
                "prix_quartier_detail_m2_appart", "Longitude": "longitude", "Latitude": "latitude", "Plat": "plats"}, inplace=True)

    # creating the foreign key in fact table for dim_site
    data['site_id'] = data['cantine_nom'] + '_' + data['annee_scolaire']
    data['site_id'] = pd.Categorical(data['site_id']).codes + 1

    # creating the FK in fact table for all other dimensions
    data["date"] = pd.to_datetime(data["date"])
    data['jour_id'] = pd.Categorical(data['date']).codes + 1

    # dispatching the data into sub dfs that will feed the datawarehouse

    # fact table
    frequentation_quotidienne_df = data[[
        'jour_id', 'site_id', 'date', 'prevision', 'reel']]

    # geographic dimension
    dim_site_df = data[['site_type', 'cantine_nom', 'annee_scolaire',
                        'effectif', 'quartier_detail', 'prix_quartier_detail_m2_appart',
                        'prix_moyen_m2_appartement', 'prix_moyen_m2_maison', 'longitude', 'latitude']]

    # menus dimension 
    dim_menu_df = data[['date', 'plats']]

    # temporal dimension
    dim_temporelle_df = data[['date', 'vacances_dans', 'depuis_vacances',
                                'ferie_dans', 'depuis_ferie', 'chretiennes_dans', 'depuis_chretiennes',
                                'juives_dans', 'depuis_juives', 'ramadan_dans', 'depuis_ramadan',
                                'musulmanes_dans', 'depuis_musulmanes']]

    # events dimensions
    dim_events_df = data[['date', 'chretiennes', 'juives',
                            'ramadan', 'musulmanes', 'greve']]

    # datawarehousing allow us to drop duplicates in dimension tables to improve efficiency
    dim_site_df.drop_duplicates(inplace=True)
    dim_menu_df.drop_duplicates(inplace=True)
    dim_temporelle_df.drop_duplicates(inplace=True)
    dim_events_df.drop_duplicates(inplace=True)

    # building the sql tables that will constitute the DTWH

    # create a connector to the db
    conn = sql.connect('data/frequentation_dtwh.db')

    # drop table if exist
    cursor = conn.cursor()
    for table in [ 'Frequentation_quotidienne', 'Dim_site', 'Dim_menu', 'Dim_temporelle', 'Dim_evenement']:
        command = "DROP TABLE IF EXISTS {};".format(table)
        cursor.execute(command)
    conn.commit()


    # Create Dim_temporelle
    cursor.execute('''CREATE TABLE IF NOT EXISTS `Dim_temporelle` (
        `jour_id` INTEGER PRIMARY KEY AUTOINCREMENT,
        `date` DATE NOT NULL,
        `vacances_dans` INTEGER NULL,
        `depuis_vacances` INTEGER NULL,
        `ferie_dans` INTEGER NULL,
        `depuis_ferie` INTEGER NULL,
        `chretiennes_dans` INTEGER NULL,
        `depuis_chretiennes` INTEGER NULL,
        `musulmanes_dans` INTEGER NULL,
        `depuis_musulmanes` INTEGER NULL,
        `ramadan_dans` INTEGER NULL,
        `depuis_ramadan` INTEGER NULL,
        `juives_dans` INTEGER NULL,
        `depuis_juives` INTEGER NULL);
        ''')

    # Create Dim_site
    cursor.execute('''CREATE TABLE IF NOT EXISTS `Dim_site` (
        `site_id` INTEGER PRIMARY KEY AUTOINCREMENT,
        `site_type` VARCHAR(5) NULL,
        `cantine_nom` VARCHAR(50) NULL,
        `annee_scolaire` VARCHAR(30) NULL,
        `effectif` INTEGER NULL,
        `quartier_detail` VARCHAR(30) NULL,
        `prix_quartier_detail_m2_appart` INTEGER NULL,
        `prix_moyen_m2_appartement` INTEGER NULL,
        `prix_moyen_m2_maison` INTEGER NULL,
        `longitude` FLOAT NULL,
        `latitude` FLOAT NULL);
        ''')
            
    # Create Dim_menu
    cursor.execute('''CREATE TABLE IF NOT EXISTS `Dim_menu` (
        `jour_id` INTEGER PRIMARY KEY AUTOINCREMENT,
        `date` DATE NOT NULL,
        `plats` VARCHAR(200) NULL);
        ''')

    # Create Dim_evenement
    cursor.execute('''CREATE TABLE IF NOT EXISTS `Dim_evenement` (
        `jour_id` INTEGER PRIMARY KEY AUTOINCREMENT,
        `date` DATE NOT NULL,
        `chretiennes` TINYINT(1) NULL,
        `juives` TINYINT(1) NULL,
        `ramadan` TINYINT(1) NULL,
        `musulmanes` TINYINT(1) NULL,
        `greve` TINYINT(1) NULL);
        ''')

    # Create fact table (setting up the FKs and reference to dimensions)
    cursor.execute('''CREATE TABLE IF NOT EXISTS `Frequentation_quotidienne` (
        `jour_site_id` INTEGER PRIMARY KEY AUTOINCREMENT,
        `date` DATE NOT NULL,
        `reel` INT NULL,
        `prevision` INT NULL,
        `jour_id` INT NOT NULL,
        `site_id` INT NOT NULL,
        CONSTRAINT `jour_id`
            FOREIGN KEY (`jour_id`)
            REFERENCES `Dim_temporelle` (`jour_id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION,
        CONSTRAINT `site_id`
            FOREIGN KEY (`site_id`)
            REFERENCES `Dim_site` (`site_id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION,
        CONSTRAINT `fk_Frequentation_quotidienne_Dim_menu1`
            FOREIGN KEY (`jour_id`)
            REFERENCES `Dim_menu` (`jour_id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION,
        CONSTRAINT `fk_Frequentation_quotidienne_Dim_evenement1`
            FOREIGN KEY (`jour_id`)
            REFERENCES `Dim_evenement` (`jour_id`)
            ON DELETE NO ACTION
            ON UPDATE NO ACTION);
            ''')

    # create index on date for faster retrieval
    cursor.execute(''' CREATE INDEX date 
    ON Frequentation_quotidienne(date); ''')        
    conn.commit()

    # insertion des données dans le dtwh
    dim_site_df.to_sql('Dim_site', conn,
                        if_exists='append', index=False)
    frequentation_quotidienne_df.to_sql('Frequentation_quotidienne', 
                        conn, if_exists='append', index=False)
    dim_menu_df.to_sql('Dim_menu', conn,
                        if_exists='append', index=False)
    dim_temporelle_df.to_sql('Dim_temporelle', 
                        conn, if_exists='append', index=False)
    dim_events_df.to_sql('Dim_evenement', conn,
                        if_exists='append', index=False)
    conn.commit()

    # check first row of each tables of DTWH
    # for table in ['Frequentation_quotidienne', 'Dim_site', 'Dim_menu', 'Dim_temporelle', 'Dim_evenement']:
    #     print("------------------- {} --------------------".format(table), '\n')
    #     command = "SELECT * FROM {} limit 1".format(table)
    #     cursor.execute(command)
    #     for row in cursor.fetchall():
    #         print(row, '\n')
    # conn.commit()

    cursor.close()

    print('Datawarehouse built succesfully.')


if __name__ == "__main__":
    main()