import pandas as pd
import numpy as np
from datetime import datetime

import plotly.graph_objs as go
import plotly.colors

from scripts.data_load import load_dataset


def menus_figures(canteen, start_date, end_date):
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

  # handling outliers for better analysis
  data = data[data['reel'] > 0 ]

  # filtering data based on front end inputs
  data_filter = data.query('cantine_nom == @canteen and date >= @start_date and date <= @end_date')

  data_filter['is_pig'] = np.where(data_filter['plats'].str.contains(
    'carbonara|carbonata|cassoulet|chipo|chipolatas|francfort|cordon|tartiflette| \
    croziflette|saucisse|saucisses|porc|knacks|lardons|jambon|choucroute|rosette'), 1, 0)
  
  data_filter['is_meat'] = np.where(data_filter['plats'].str.contains(
    'carbonara|carbonata|cassoulet|chipo|chipolatas|francfort|cordon|tartiflette| \
    croziflette|saucisse|saucisses|porc|knacks|lardons|jambon|choucroute|rosette| \
    roti|agneau|blanquette|boeuf|boudin|boulettes|bourguignon|bourguignonne|canard|carne| \
    chapon|colombo|couscous|dinde|escalope|farci|foie|kebab|lapin|merguez|mouton|napolitaines| \
    nuggets|paupiette|pintade|poulet|steak|stogonoff|strogonoff|tagine|tajine|veau|viande|volaile| \
    volaille|parmentier'), 1, 0)

  data_filter['is_fish'] = np.where(data_filter['plats'].str.contains(
    'poisson|meunière|sardine|colin|lieu|thon|dorade|saumon|maquereau|poissons|sardines| \
    perray|calamar|crabe|crevette|crustace|dorade|rillette'), 1, 0)

  data_filter['is_bio'] = np.where(data_filter['plats'].str.contains('bio|biologique|bios|biologiques'), 1, 0)


  # top 10 menus
  top_menus = data_filter.groupby('plats').reel.mean().sort_values(ascending=False).reset_index().head(20)
  top_menus['reel'] = top_menus['reel'].round(2)

  top_graph = go.Figure(data=[go.Table(
    header=dict(values=['Composition du menu', 'Fréquentation'],
                fill_color='#B8DFD8',
                align='left'),
    cells=dict(values=top_menus.transpose().values.tolist(),
               fill_color='lavender',
               align='left'),
    columnorder=[1,0], columnwidth=[0.75,0.25]),
  ])

  top_graph.update_layout(title=dict(text='Top 20 des menus préférés'))

  # meat graph
  meat_graph = []

  meat_graph.append(
      go.Box(
      x = data_filter.is_meat.tolist(),
      y = data_filter.reel.tolist(),
      name = 'Viande',
      boxpoints="all",
      boxmean=True,
      notched=True,
      line=dict(color="#548CA8")
      )
  )

  meat_layout = dict(title = "Impact de la présence de viande sur la fréquentation",
                  xaxis = dict(title = 'Viande', autotick=True),
                  yaxis = dict(title = 'Fréquentation'),
                  )

  # fish graph
  fish_graph = []

  fish_graph.append(
      go.Box(
      x = data_filter.is_fish.tolist(),
      y = data_filter.reel.tolist(),
      name = 'Poisson',
      boxpoints="all",
      boxmean=True,
      notched=True,
      line=dict(color="#9960d6")
      )
  )

  fish_layout = dict(title = "Impact de la présence de poisson sur la fréquentation",
                  xaxis = dict(title = 'Poisson', autotick=True),
                  yaxis = dict(title = 'Fréquentation'),
                  )

  # pig graph
  pig_graph = []

  pig_graph.append(
      go.Box(
      x = data_filter.is_pig.tolist(),
      y = data_filter.reel.tolist(),
      name = 'Porc',
      boxpoints="all",
      boxmean=True,
      notched=True,
      line=dict(color="#26992e")
      )
  )

  pig_layout = dict(title = "Impact de la présence de porc sur la fréquentation",
                  xaxis = dict(title = 'Porc', autotick=True),
                  yaxis = dict(title = 'Fréquentation'),
                  )

# organic graph
  bio_graph = []

  bio_graph.append(
      go.Box(
      x = data_filter.is_bio.tolist(),
      y = data_filter.reel.tolist(),
      name = 'Bio',
      boxpoints="all",
      boxmean=True,
      notched=True,
      line=dict(color="#b50d0d")
      )
  )

  bio_layout = dict(title = "Impact de la présence de produits bio sur la fréquentation",
                  xaxis = dict(title = 'Bio', autotick=True),
                  yaxis = dict(title = 'Fréquentation'),
  )

  # append all charts
  figures = []
  figures.append(top_graph)
  figures.append(dict(data=meat_graph, layout=meat_layout))
  figures.append(dict(data=fish_graph, layout=fish_layout))
  figures.append(dict(data=pig_graph, layout=pig_layout))
  figures.append(dict(data=bio_graph, layout=bio_layout)) 


  return figures