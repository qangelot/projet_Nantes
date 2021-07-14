# ETL:
# Extract: selecting the right data and obtaining it 
# Transform: data cleansing is applied to that data while it sits in a staging area
# Loading: loading of the transformed data into the data store or a data warehouse
import numpy as np
import pandas as pd 
import re

from scripts.utils_date_features import *
from scripts.utils_religion_features import *
from scripts.utils_nlp_features import *

############################## Attendance dataframe building ################################

def main():
    # read data from main CSVs 
    effectifEcoles =  pd.read_csv('../data/effectifs_ecolesnantes.csv', header=0, sep=';')
    effectifEcoles.drop(['Début année scolaire'], axis=1, inplace=True)

    appariement = pd.read_csv('../data/appariement_ecoles_cantines.csv', header=0, sep=',')
    appariement.rename(columns={"ecole": "Ecole"}, inplace=True)

    freqJ = pd.read_csv('../data/frequentation_cantines_v2.csv', header=0, sep=',', low_memory=False)
    freqJ["date"] = pd.to_datetime(freqJ["date"])

    # drop useless duplicate columns
    freqJ.drop(['site_nom_sal', 'site_id', 'prevision_s', 'reel_s'], axis=1, inplace=True)
    freqJ.rename(columns={'site_nom':'cantine_nom'}, inplace=True)
    freqJ.sort_values(by='date', inplace=True, ascending=True)
    freqJ.reset_index(inplace=True, drop=True)

    # get aggregated headcounts by canteen
    effectifEcoles = pd.merge(effectifEcoles, appariement[['cantine_nom','Ecole']], on='Ecole')
    effectifEcoles.rename(columns={'Année scolaire':'annee_scolaire'}, inplace=True)
    effectifCantines = effectifEcoles.groupby(['annee_scolaire','cantine_nom'], as_index=False).sum()

    # join main df with the school year
    freqJ = get_school_year(freqJ, 'date', '../data/annees_scolaires.csv')

    # join main df with canteen headcounts
    freqJ = pd.merge(freqJ, effectifCantines[['annee_scolaire', 'cantine_nom', 'Effectif']], on=['annee_scolaire', 'cantine_nom'])

    ##  adding variable of interest based on insights from canteen employees
    # often parents often withdraw their children a few days before the holidays or do not return until a few days later 
    freqJ = get_distance_holidays(freqJ, 'date', '../data/vacances.csv')

    # same logic goes for public holidays 
    freqJ = get_distance_public(freqJ, 'date', '../data/jours_feries.csv')


    ############################## Religious evetns dataframe building #################################
    # As stated in the specification, we will not include strikes in the analysis - the model should help agents in predicting normal periods.
    # Moreover, most of the time, we will not have information about strikes at the time of making the prediction (2 or 3 weeks in advance)
    # That said, we keep it for analysis purposes

    greves = pd.read_csv('../data/greves_restauration_et_ou_ecoles.csv', header=0, sep=',')
    greves["date"] = pd.to_datetime(greves["date"])

    # read and standardize data from main CSVs 
    chretiennes = pd.read_csv('../data/fetes_chretiennes.csv', header=None, sep=',', encoding="ISO-8859-1")

    # reconstruct date variable with regex
    chretiennes['Année'] = chretiennes[1].apply(get_year)
    chretiennes['Mois'] = chretiennes[1].apply(get_month).map({"Janvier": "01", "Février": "02", "Mars": "03", "Avril": "04",
                                                            "Mai": "05", "Juin": "06", "Juillet": "07", "Août": "08", "Septembre": "09",
                                                            "Octobre": "10", "Novembre": "11", "Décembre": "12"})
    chretiennes['Jour'] = chretiennes[0].apply(get_day)
    chretiennes['Jour'] = chretiennes['Jour'].apply(lambda x: x.zfill(2))
    chretiennes['date'] = chretiennes['Jour'].astype(str) + '/' + chretiennes['Mois'].astype(str) + '/' + chretiennes['Année'].astype(str)
    chretiennes['date']
    chretiennes.drop([0, 1, 2, 3, 4, 5, 'Année', 'Mois', 'Jour'], axis=1, inplace=True)
    chretiennes['chretiennes'] = 1
    chretiennes["date"] = pd.to_datetime(chretiennes["date"], format="%d/%m/%Y")

    # same for jewish events
    juives = pd.read_csv('../data/fetes_juives.csv', header=None, sep=',', encoding="ISO-8859-1")
    juives['Année'] = juives[1].apply(get_year)
    juives['Mois'] = juives[1].apply(get_month).map({"Janvier": "01", "Février": "02", "Mars": "03", "Avril": "04",
                                                    "Mai": "05", "Juin": "06", "Juillet": "07", "Août": "08", "Septembre": "09",
                                                    "Octobre": "10", "Novembre": "11", "Décembre": "12"})
    juives['Jour'] = juives[0].apply(get_day)
    juives['Jour'] = juives['Jour'].apply(lambda x: x.zfill(2))
    juives['date'] = juives['Jour'].astype(str) + '/' + juives['Mois'].astype(str) + '/' + juives['Année'].astype(str)
    juives.drop([0, 1, 2, 3, 4, 5, 'Année', 'Mois', 'Jour'], axis=1, inplace=True)
    juives['juives'] = 1
    juives["date"] = pd.to_datetime(juives["date"], format="%d/%m/%Y")

    # same for muslim events
    musulm = pd.read_csv('../data/fetes_musulmanes.csv', header=None, sep=',', encoding = "ISO-8859-1")
    musulm.drop([1,2,3,4,5,6], axis=1, inplace=True)
    musulm.rename(columns={0: "date"}, inplace=True)
    musulm['musulmanes'] = 1
    musulm["date"] = musulm["date"].str.replace(" ", "")
    musulm["date"] = pd.to_datetime(musulm["date"], format="%d/%m/%Y")

    # same for ramadan
    ramad = pd.read_csv('../data/ramad.csv', header=None, sep=',')
    ramad.rename(columns={0: "date", 1: "ramadan"}, inplace=True)
    ramad["date"] = pd.to_datetime(ramad["date"], format="%d/%m/%Y")

    # merge relgion events dataframes based on the length of the time serie (freqJ)
    religion_dfs = [chretiennes, juives, musulm, ramad, greves]
    religion_df = merge_religious_events(freqJ, 'date', religion_dfs)
    religion_df.drop_duplicates(inplace=True)
    religion_df.to_csv('data/events.csv', index=False)

    # following the same logic than for holidays and adding proximity variables from religious events
    freqJ = events_in_ago(freqJ, 'date', 'data/events.csv')

    # adding the religous bolean features
    freqJ = pd.merge(freqJ, religion_df, how='left', on='date')
    freqJ.sort_index(inplace=True, ascending=True)
    freqJ.sort_values(by='date', inplace=True, ascending=True)
    freqJ.reset_index(inplace=True, drop=True)


    ############################## Geographic dataframe building #################################
    # now that we have created our main analysis table, we can go on to integrate additional data

    # read data from main CSVs 
    appariement = pd.read_csv('../data/appariement_ecoles_cantines.csv', header=0, sep=',')
    appariement.rename(columns={"ecole": "Ecole"}, inplace=True)
    appariement['Ecole'] = appariement['Ecole'].apply(lambda x: x.rsplit(' ', 1)[0])

    geo_features = pd.read_csv('../data/geo_features.csv', header=0, sep=';') 
    geo_features['nom_etab'] = geo_features['nom_etab'].apply(lambda x: x.rsplit(' ', 1)[0])
    geo_features.rename(columns={"nom_etab": "Ecole"}, inplace=True)

    # merging on canteen because the granularity of analysis is on canteen and not school level
    geo_features = pd.merge(geo_features, appariement[['cantine_nom', 'Ecole']].drop_duplicates(subset=['Ecole']), how='inner', on='Ecole')

    # subsetting to keep only the necessary date for analysis
    geo_features = geo_features[['cantine_nom', 'Quartier_detail', 'prix_Quartier_detail_m2_appart', 
                                'prix_moyen_m2_appartement', 'prix_moyen_m2_maison', 'Longitude_Latitude']]

    # extract latitude and longitude information using regex
    geo_features['Longitude_Latitude'] = geo_features['Longitude_Latitude'].apply(lambda x: re.findall('\d+\.\d+', x) )
    geo_features['Longitude'] = geo_features['Longitude_Latitude'].apply(lambda x: x[0]).astype(float).round(4)
    geo_features['Latitude'] = geo_features['Longitude_Latitude'].apply(lambda x: x[1]).astype(float).round(4)
    geo_features.drop('Longitude_Latitude', axis=1, inplace=True)

    # adding the geographic features to main df
    freqJ = pd.merge(freqJ, geo_features.drop_duplicates(subset=['cantine_nom']), how='left', on='cantine_nom')
    freqJ["date"] = pd.to_datetime(freqJ["date"])
    freqJ.sort_index(inplace=True, ascending=True)

    ############################## Menus dataframe building #################################
    # extracting and cleaning the menus in the form of a string of characters

    # read data from menu CSV
    menus = pd.read_csv('../data/menus-cantines-nantes-2011-2019.csv', header=0, sep=';')
    menus.rename(columns={"Date": "date"}, inplace=True)
    menus["date"] = pd.to_datetime(menus["date"])
    menus.sort_values(by='date', inplace=True, ascending=True)
    menus = menus.reset_index(drop=True)

    # group by date and apply transformation for further NLP
    menus = menus.groupby('date',as_index=False)['Plat'].apply(lambda text : ' '.join(parse_text(text)))

    # adding the menu feature to main df
    data = pd.merge(freqJ, menus, how='left', on='date')
    data["date"] = pd.to_datetime(data["date"])
    data.sort_index(inplace=True, ascending=True)
    data.to_csv('data/data.csv', index=False)

    print('Extract and transform steps done.')

if __name__ == "__main__":
    main()
