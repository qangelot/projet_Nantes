import pandas as pd
import numpy as np
import re


def get_year(date):
    x = re.findall('([\d]{4})', date)
    if x :
        return x[0]


def get_month(date):
    x = re.findall('^[A-Z][a-zéû]+', date)
    if x:
        return x[0]


def get_day(date):
    return date[-2:]


def merge_religious_events(data, date_col, religion_dfs):
    """"
    merge religion dataframes based on the length of the whole time serie
    """

    # generate all dates between start and end
    start = data[date_col].min()
    end = data[date_col].max()
    religion_df = pd.date_range(start, end, freq="D").to_frame(index=False, name="date")

    for rel in religion_dfs:
        religion_df = pd.merge(religion_df, rel, how='left', on='date')
        
    religion_df = religion_df.fillna(0)
    
    return religion_df


def events_in_ago(data, date_col, data_path):
    """"
    add features about how close and far we are from religious events
    """

    # generate all dates within start and end 
    start = data[date_col].min()
    end = data[date_col].max()
    dfs = pd.date_range(start, end, freq="D").to_frame(index=False, name="_date")

    # read external holidays csv
    def _parser(date):
        return pd.to_datetime(date)

    events = pd.read_csv(f'{data_path}', parse_dates=['date'], date_parser=_parser)

    for col in ['chretiennes', 'juives', 'ramadan', 'musulmanes']:
        df = dfs.copy()
        event = events.copy()
        event = event[["date", col]]
        event = event[event[col] != 0]
        event = event.drop_duplicates()

        # simulate an interval based left join using pandas
        # perform a cross join on temp_key
        low_bound = "date"
        df['temp_key'] = 1
        event['temp_key'] = 1

        crossjoindf = pd.merge(df, event, on=['temp_key'])
        df.drop(columns=['temp_key'], inplace=True)
        crossjoindf.drop(columns=['temp_key'], inplace=True)

        # filter with lower_bound  
        conditionnal_df = crossjoindf[(crossjoindf['_date'] == crossjoindf[low_bound])]

        # merge on the main df with all cols as keys to simulate left join
        df_col = df.columns.values.tolist()
        conditionnal_df.set_index(df_col, inplace=True)
        df = df.merge(conditionnal_df, left_on=df_col, right_index=True, how='left')

        # find rows index corresponding to holidays
        events_index = np.where(~df[col].isnull())[0]

        # compute arrays of first day and last day of holidays
        events_min_index = []
        events_max_index = []
        i = 0
        while i < len(events_index):
            j = 0
            while i + j < len(events_index) and (events_index[i] + j) == events_index[i + j]:
                j += 1
            events_min_index.append(events_index[i])
            events_max_index.append(events_index[i + j - 1])
            i += j

        indexes = range(0, len(df)) 

        # compute for each index row the distance with the nearest upcoming public holiday
        df[col + '_dans'] = [min([i - x for i in events_min_index if i > x], default=0) for x in indexes]

        # compute for each index row the distance with the latest past holidays
        df[ 'depuis_' + col] = [min([x - i for i in events_max_index if i < x], default=0) for x in indexes]

        # set pub_holidays_in and pub_holidays_ago to 0 during effective holidays
        df.loc[~df[col].isnull(), col + '_dans'] = 0
        df.loc[~df[col].isnull(), 'depuis_' + col] = 0

        # we drop date col that was just useful to define lower_bound 
        # and we rename _date to have the same key name to join both dataframes
        df.drop(columns=['date'], inplace=True)
        df.rename(columns={"_date": "date"}, inplace=True)
        df.set_index('date', inplace=True)
        data = pd.merge(data, df[[col + '_dans', 'depuis_' + col]], on='date')
    
    return data
