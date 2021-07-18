import pandas as pd
import numpy as np
from datetime import datetime

import plotly.graph_objs as go
import plotly.colors
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scripts.data_load import load_dataset


def freq_figures(canteen, start_date, end_date):
  """Creates figures about canteen frequentation
    Args:
        canteen (str): canteen name for filtering the data
    Returns:
        list (dict): list containing the plotly visualizations
  """

  start_date = datetime.strptime(start_date , '%d/%m/%Y').strftime("%Y-%m-%d")
  end_date = datetime.strptime(end_date , '%d/%m/%Y').strftime("%Y-%m-%d")

  data = load_dataset(file_name="./data/frequentation_dtwh.db")
  data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
  data.sort_values("date", inplace=True)

  # time serie decomposition
  time_serie = data.groupby('date').reel.sum()
  time_serie_date = time_serie.index
  decomposed_full = sm.tsa.seasonal_decompose(time_serie, period=200, extrapolate_trend='freq')

  # filtering data based on front end inputs
  data_filter = data.query('cantine_nom == @canteen and date >= @start_date and date <= @end_date')


  # attendance graph
  freq_graph = []

  freq_graph.append(
      go.Scatter(
      x = data_filter.date.tolist(),
      y = data_filter.reel.tolist(),
      mode = 'lines',
      name = 'Fréquentation',
      line=dict(color="#548CA8")
      )
  )

  freq_layout = dict(title = "Fréquentation réelle",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # forecasting graph
  prev_graph = []

  prev_graph.append(
      go.Scatter(
      x = data_filter.date.tolist(),
      y = data_filter.prevision.tolist(),
      mode = 'lines',
      name = 'Prévision',
      line=dict(color="#9960d6")
      )
  )

  prev_layout = dict(title = "Prévisions de fréquentation",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # global attendance graph
  freq_global_graph = []

  freq_global_graph.append(
      go.Scatter(
      x = time_serie_date.tolist(),
      y = decomposed_full.observed.tolist(),
      mode = 'lines',
      name = 'Fréquentation globale',
      line=dict(color="#548CA8")
      )
  )

  freq_global_layout = dict(title = "Fréquentation globale réelle",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # global trend graph
  trend_global_graph = []

  trend_global_graph.append(
      go.Scatter(
      x = time_serie_date.tolist(),
      y = decomposed_full.trend.tolist(),
      mode = 'lines',
      name = 'Tendance globale',
      line=dict(color="#edb139")
      )
  )

  trend_global_layout = dict(title = "Tendance globale",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # global seasonality graph
  seasonal_global_graph = []

  seasonal_global_graph.append(
      go.Scatter(
      x = time_serie_date.tolist(),
      y = decomposed_full.seasonal.tolist(),
      mode = 'lines',
      name = 'Saisonalité globale',
      line=dict(color="#26992e")
      )
  )

  seasonal_global_layout = dict(title = "Saisonalité globale",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )

  # global residual graph
  residual_global_graph = []

  residual_global_graph.append(
      go.Scatter(
      x = time_serie_date.tolist(),
      y = decomposed_full.resid.tolist(),
      mode = 'lines',
      name = 'Résidus globaux',
      line=dict(color="#b50d0d")
      )
  )

  residual_global_layout = dict(title = "Résidus globaux",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Volume'),
                  )


  # append all charts
  figures = []
  figures.append(dict(data=freq_graph, layout=freq_layout))
  figures.append(dict(data=prev_graph, layout=prev_layout))
  figures.append(dict(data=freq_global_graph, layout=freq_global_layout))
  figures.append(dict(data=trend_global_graph, layout=trend_global_layout))
  figures.append(dict(data=seasonal_global_graph, layout=seasonal_global_layout))
  figures.append(dict(data=residual_global_graph, layout=residual_global_layout))


  return figures