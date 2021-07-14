import pandas as pd
import numpy as np


def get_school_year(data, date_col, data_path):
    """ add a new column annee_scolaire with annees_scolaires.csv file to the main dataframe """

     # generate empty df with all dates between start and end
    start = data[date_col].min()
    end = data[date_col].max()
    df = pd.date_range(start, end, freq="D").to_frame(index=False, name="date")

    # read external holidays csv
    def _parser(date):
        return pd.to_datetime(date)

    holidays = pd.read_csv(f'{data_path}',
                            parse_dates=['date_debut', 'date_fin'],
                            date_parser=_parser)
    holidays = holidays[["annee_scolaire", "date_debut", "date_fin"]]
    holidays = holidays.drop_duplicates()

    # simulate an interval based left join using pandas
    # perform a cross join on temp_key
    up_bound = "date_fin"
    low_bound = "date_debut"
    df['temp_key'] = 1
    holidays['temp_key'] = 1
    crossjoindf = pd.merge(df, holidays, on=['temp_key'])

    df.drop(columns=['temp_key'], inplace=True)
    crossjoindf.drop(columns=['temp_key'], inplace=True)
    
    # filter with lower_bound & upper_bound
    conditionnal_df = crossjoindf[
        (crossjoindf["date"] >= crossjoindf[low_bound]) & (crossjoindf["date"] <= crossjoindf[up_bound])]

    # merge on the main df with all cols as keys to simulate left join
    df_col = df.columns.values.tolist()
    conditionnal_df.set_index(df_col, inplace=True)
    df = df.merge(conditionnal_df, left_on=df_col, right_index=True, how='left')

    df.set_index('date', inplace=True) 
    data = pd.merge(data, df['annee_scolaire'], on='date')
    
    return data


def get_distance_holidays(data, date_col, data_path):
    """
    add features about how close and how far we are from holidays
    """

    # generate all dates within start and end
    start = data[date_col].min()
    end = data[date_col].max()
    df = pd.date_range(start, end, freq="D").to_frame(index=False, name="date")

    # read external holidays csv
    def _parser(date):
        return pd.to_datetime(date)

    holidays = pd.read_csv(f'{data_path}',
                            parse_dates=['date_debut', 'date_fin'],
                            date_parser=_parser)
    holidays = holidays[["vacances_nom", "date_debut", "date_fin", "zone", "vacances"]]
    holidays = holidays.drop_duplicates()

    # simulate an interval based left join using pandas
    # perform a cross join on temp_key
    up_bound = "date_fin"
    low_bound = "date_debut"
    df['temp_key'] = 1
    holidays['temp_key'] = 1
    crossjoindf = pd.merge(df, holidays, on=['temp_key'])

    df.drop(columns=['temp_key'], inplace=True)
    crossjoindf.drop(columns=['temp_key'], inplace=True)
    
    # filter with lower_bound & upper_bound
    conditionnal_df = crossjoindf[
        (crossjoindf["date"] >= crossjoindf[low_bound]) & (crossjoindf["date"] <= crossjoindf[up_bound])]

    # merge on the main df with all cols as keys to simulate left join
    df_col = df.columns.values.tolist()
    conditionnal_df.set_index(df_col, inplace=True)
    df = df.merge(conditionnal_df, left_on=df_col, right_index=True, how='left')
 
    # find rows index corresponding to holidays
    holidays_index = np.where(~df['vacances_nom'].isnull())[0]

    # compute arrays of first day and last day of holidays
    holidays_min_index = []
    holidays_max_index = []
    i = 0
    while i < len(holidays_index):
        j = 0
        while i + j < len(holidays_index) and (holidays_index[i] + j) == holidays_index[i + j]:
            j += 1
        holidays_min_index.append(holidays_index[i])
        holidays_max_index.append(holidays_index[i + j - 1])
        i += j

    indexes = range(0, len(df)) 
    
    # compute for each index row the distance with the nearest upcoming holidays
    df['vacances_dans'] = [min([i - x for i in holidays_min_index if i > x], default=0) for x in indexes]
    
    # compute for each index row the distance with the latest past holidays
    df['depuis_vacances'] = [min([x - i for i in holidays_max_index if i < x], default=0) for x in indexes]
    
    # set holidays_in and holidays_ago to 0 during effective holidays
    df.loc[~df['vacances_nom'].isnull(), 'vacances_dans'] = 0
    df.loc[~df['vacances_nom'].isnull(), 'depuis_vacances'] = 0
    
    df.set_index('date', inplace=True)
    data = pd.merge(data, df[['vacances_dans', 'depuis_vacances']], on='date')
    
    return data


def get_distance_public(data, date_col, data_path):
    """"
    add features about how close and how far we are from public holidays
    """

    # generate all dates within start and end 
    start = data[date_col].min()
    end = data[date_col].max()
    df = pd.date_range(start, end, freq="D").to_frame(index=False, name="_date")

    # read external holidays csv
    def _parser(date):
        return pd.to_datetime(date)

    pub_holidays = pd.read_csv(f'{data_path}', parse_dates=['date'], date_parser=_parser)
    pub_holidays = pub_holidays[["date", "nom_jour_ferie"]]
    pub_holidays = pub_holidays.drop_duplicates()

    # simulate an interval based left join using pandas
    # perform a cross join on temp_key
    low_bound = "date"
    df['temp_key'] = 1
    pub_holidays['temp_key'] = 1
    
    crossjoindf = pd.merge(df, pub_holidays, on=['temp_key'])
    df.drop(columns=['temp_key'], inplace=True)
    crossjoindf.drop(columns=['temp_key'], inplace=True)
     
    # filter with lower_bound  
    conditionnal_df = crossjoindf[(crossjoindf['_date'] == crossjoindf[low_bound])]

    # merge on the main df with all cols as keys to simulate left join
    df_col = df.columns.values.tolist()
    conditionnal_df.set_index(df_col, inplace=True)
    df = df.merge(conditionnal_df, left_on=df_col, right_index=True, how='left')
 
    # find rows index corresponding to holidays
    pub_holidays_index = np.where(~df['nom_jour_ferie'].isnull())[0]

    # compute arrays of first day and last day of holidays
    pub_holidays_min_index = []
    pub_holidays_max_index = []
    i = 0
    while i < len(pub_holidays_index):
        j = 0
        while i + j < len(pub_holidays_index) and (pub_holidays_index[i] + j) == pub_holidays_index[i + j]:
            j += 1
        pub_holidays_min_index.append(pub_holidays_index[i])
        pub_holidays_max_index.append(pub_holidays_index[i + j - 1])
        i += j

    indexes = range(0, len(df)) 
    
    # compute for each index row the distance with the nearest upcoming public holiday
    df['ferie_dans'] = [min([i - x for i in pub_holidays_min_index if i > x], default=0) for x in indexes]
    
    # compute for each index row the distance with the latest past holidays
    df['depuis_ferie'] = [min([x - i for i in pub_holidays_max_index if i < x], default=0) for x in indexes]
    
    # set pub_holidays_in and pub_holidays_ago to 0 during effective holidays
    df.loc[~df['nom_jour_ferie'].isnull(), 'ferie_dans'] = 0
    df.loc[~df['nom_jour_ferie'].isnull(), 'depuis_ferie'] = 0
    
    # we drop date col that was just useful to define lower_bound 
    # and we rename _date to have the same key name to join both dataframes
    df.drop(columns=['date'], inplace=True)
    df.rename(columns={"_date": "date"}, inplace=True)
    df.set_index('date', inplace=True)
    data = pd.merge(data, df[['ferie_dans', 'depuis_ferie']], on='date')
    
    return data