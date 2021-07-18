import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3 as sql

import plotly.graph_objs as go
import plotly.colors
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scripts.data_load import load_dataset


def tempo_figures(canteen, start_date, end_date):
  """Creates figures about canteen frequentation
    Args:
        canteen (str): canteen name for filtering the data
    Returns:
        list (dict): list containing the plotly visualizations
  """

  start_date = datetime.strptime(start_date , '%d/%m/%Y').strftime("%Y-%m-%d")
  end_date = datetime.strptime(end_date , '%d/%m/%Y').strftime("%Y-%m-%d")

  conn = sql.connect('./data/frequentation_dtwh.db')

  SQL_Query = pd.read_sql_query(
    '''select 
    Frequentation_quotidienne.date, 
    Dim_site.cantine_nom,
    strftime("%w",Frequentation_quotidienne.date) as weekday,
    strftime("%m",Frequentation_quotidienne.date) as month,
    Frequentation_quotidienne.prevision, 
    Frequentation_quotidienne.reel,
    Dim_temporelle.vacances_dans, 
    Dim_temporelle.depuis_vacances,
    Dim_temporelle.ferie_dans, 
    Dim_temporelle.depuis_ferie

    from Frequentation_quotidienne
    
    left join Dim_site               on Frequentation_quotidienne.site_id = Dim_site.site_id
    left join Dim_temporelle         on Frequentation_quotidienne.jour_id = Dim_temporelle.jour_id

    order by Frequentation_quotidienne.date
    ''', conn)

  data = pd.DataFrame(SQL_Query, columns=['date', 'cantine_nom', 'weekday', 'month', 'reel', 
                                        'vacances_dans', 'depuis_vacances', 'ferie_dans', 'depuis_ferie'])
  data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
  data.sort_values("date", inplace=True)

  # handling outliers for better analysis
  data = data[data['reel'] > 0 ]
 
  # filtering data based on front end inputs
  data_filter = data.query('cantine_nom == @canteen and date >= @start_date and date <= @end_date')
  
  # create features for analysis 
  conditions = [(data_filter['vacances_dans'] < 5), (data_filter['depuis_vacances'] < 5)]
  choices = [1, 0]
  data_filter['is_close_holidays'] = np.select(conditions, choices)

  conditions = [(data_filter['ferie_dans'] < 3), (data_filter['depuis_ferie'] < 3)]
  data_filter['is_close_public'] = np.select(conditions, choices) 

  # attendance graph
  day_graph = []

  grouped_day = data_filter.groupby('weekday')['reel'].mean().reset_index()

  day_graph.append(
      go.Bar(
      x = grouped_day.weekday.tolist(),
      y = grouped_day.reel.tolist(),
      text = grouped_day.reel.round().tolist(),
      textposition='auto',
      name = 'Jours',
      marker_color="#548CA8"
      )
  )

  day_layout = dict(title = "Impact du jour de la semaine",
                  xaxis = dict(title = 'Jours', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # forcasting graph
  month_graph = []

  grouped_month = data_filter.groupby('month')['reel'].mean().reset_index()

  month_graph.append(
      go.Bar(
      x = grouped_month.month.tolist(),
      y = grouped_month.reel.tolist(),
      text = grouped_month.reel.round().tolist(),
      textposition='auto',
      name = 'Mois',
      marker_color="#9960d6"
      )
  )

  month_layout = dict(title = "Impact du mois de l'année",
                  xaxis = dict(title = 'Mois', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # global attendance graph
  holidays_graph = []

  holidays_graph.append(
      go.Box(
      x = data_filter.is_close_holidays.tolist(), 
      y = data_filter.reel.tolist(), 
      name = 'Impact des vacances',
      boxpoints="all",
      boxmean=True,
      line=dict(color="#548CA8")
      )
  )

  holidays_layout = dict(title = "Impact des vacances",
                  xaxis = dict(title = 'Proximité des vacances', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # global trend graph
  public_graph = []

  public_graph.append(
      go.Box(
      x = data_filter.is_close_public.tolist(),
      y = data_filter.reel.tolist(),
      name = 'Impact des jours fériés',
      boxpoints="all",
      boxmean=True,
      line=dict(color="#548CA8")
      )
  )

  public_layout = dict(title = "Impact des jours fériés",
                  xaxis = dict(title = 'Proximité d\'un jour férié', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )


  # append all charts
  figures = []
  figures.append(dict(data=day_graph, layout=day_layout))
  figures.append(dict(data=month_graph, layout=month_layout))
  figures.append(dict(data=holidays_graph, layout=holidays_layout))
  figures.append(dict(data=public_graph, layout=public_layout))


  return figures