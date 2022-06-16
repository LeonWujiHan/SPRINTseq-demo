from pathlib import Path
import yaml
import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('config.yaml') as f:
    config = yaml.safe_load(f)

base_dir = Path('.')
data_dir = base_dir / config['data_folder']
im_shape = config['original_shape']
convert_factor = config['convert_factor']
frame = {}
for x in im_shape:
    frame[x] = im_shape[x]/convert_factor
im_bounds = [0,0,frame['lon'],frame['lat']]

cell_df = pd.read_csv(data_dir / 'cells.csv')
cell_df['Y'] = im_shape['lat'] - cell_df['Y'] - 1
cell_df['lat'] = cell_df['Y'] / convert_factor
cell_df['lon'] = cell_df['X'] / convert_factor

selected_cell_types = config['default_cell_types']

cell_types = list(set(cell_df['Type']))
with st.expander('Select cell types'):
    cell_type_checks = {}
    for cell_type in cell_types:
        cell_type_checks[cell_type] = st.checkbox(cell_type, value=cell_type in selected_cell_types)
cell_type_choices = [cell_type for cell_type in cell_types if cell_type_checks[cell_type]]
color_list = [list(x) for x in plt.get_cmap('tab20').colors]
color_list = list(map(lambda x: str([int(s * 255) for s in x]), color_list))

background_option = st.selectbox('Select background', ['Convex Hull', 'DAPI'])
background_files = config['backgrounds']
cell_layers = []
cell_layers.append(pdk.Layer(
    'BitmapLayer',
    image=config['image_repo']+background_files[background_option],
    bounds=im_bounds,
))

cell_point_size = st.slider('Point size', 1, 10, 1)
for i,cell_type in enumerate(cell_type_choices):
    df_slice = cell_df[cell_df['Type']==cell_type]
    cell_layers.append(pdk.Layer(
        'ScatterplotLayer',
        data=df_slice,
        get_position='[lon, lat]',
        get_color=color_list[i],
        get_radius=cell_point_size,
        pickable=True,
        auto_highlight=True,
    ))

init_view_state = pdk.ViewState(latitude=frame['lat']/2,longitude=frame['lon']/2,zoom=11.5,pitch=0)

st.pydeck_chart(pdk.Deck(
    map_provider=None,
    views=[pdk.View(type="MapView")],
    initial_view_state=init_view_state,
    layers=cell_layers,
    tooltip={
        "html": "<b>Y:</b> {Y} <br> <b>X:</b> {X} <br> <b>Cell Index:</b> {Cell Index} <br> <b>Type:</b> {Type}",
    },
))