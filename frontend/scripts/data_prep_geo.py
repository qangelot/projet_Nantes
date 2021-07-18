import pandas as pd
import numpy as np
from datetime import datetime

import plotly.graph_objs as go
import plotly.colors
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scripts.data_load import load_dataset


def geo_figures(canteen, start_date, end_date):
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
  data['attendance_rate'] = data['reel']/data['effectif']

  # handling outliers for analysis
  data['attendance_rate'] = np.where(data['attendance_rate'] >= 1.0 , 0.995246, data['attendance_rate'])  
    
  # filtering data based on front end inputs
  data_filter_date = data.query('date >= @start_date and date <= @end_date')
  data_filter = data.query('cantine_nom == @canteen and date >= @start_date and date <= @end_date')

  # attendance graph
  att_graph = []
    
  att_graph.append(
      go.Scatter(
      x = data_filter.date.tolist(),
      y = data_filter.attendance_rate.tolist(),
      mode = 'lines',
      name = 'Taux de présence',
      line=dict(color="#548CA8")
      )
  )

  att_layout = dict(title = "Evolution des taux de présence",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Taux de présence'),
                  )

  # global attendance graph
  att_glob_graph = []
  grouped_date = data_filter_date.groupby('date')['attendance_rate'].mean().reset_index()

  att_glob_graph.append(go.Scatter(
      x = grouped_date.date.tolist(),
      y = grouped_date.attendance_rate.tolist(),
      mode = 'lines',
      name = 'Taux de présence global',
      line=dict(color="#9960d6"))
    )
  
  att_glob_layout = dict(title = "Evolution des taux de présence globaux",
                  xaxis = dict(title = 'Année', autotick=True),
                  yaxis = dict(title = 'Taux de présence'),
                  )
  
  # attendance rate by district / real estate price
  grouped_district = data_filter_date.groupby('quartier_detail')['attendance_rate'].mean().reset_index()
  grouped_district_price = data_filter_date.groupby('quartier_detail')['prix_quartier_detail_m2_appart'].mean().reset_index()

  district_graph = go.Figure(
    data=[
        go.Bar(
      x = grouped_district.quartier_detail.tolist(),
      y = grouped_district.attendance_rate.tolist(),
      name = 'Taux de présence', yaxis='y', offsetgroup=1, 
      marker_color="#26992e"
      ),
        go.Bar(x=grouped_district_price.quartier_detail.tolist(),
            y=grouped_district_price.prix_quartier_detail_m2_appart.tolist(),
            name = 'Prix du m²', yaxis='y2', offsetgroup=2,     
            marker_color="#b50d0d" 
     )
    ],
    layout={
        'title' : {'text': "Taux de présence par quartier sur la période"},
        'yaxis': {'title': 'Taux de présence'},
        'yaxis2': {'title': 'Prix du m²', 'overlaying': 'y', 'side': 'right'}
    }
  )

  # change the bar mode
  district_graph.update_layout(barmode='group', title_x=0.5)

  # append all charts
  figures = []
  figures.append(dict(data=att_graph, layout=att_layout))
  figures.append(dict(data=att_glob_graph, layout=att_glob_layout))
  figures.append(district_graph)


  return figures