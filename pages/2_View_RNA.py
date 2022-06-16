from pathlib import Path
import yaml
import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def crop_df(x_start, x_width, y_start, y_width, df):
    df = df[np.logical_and(
        y_start <= df['Y'].to_numpy(), df['Y'].to_numpy() < y_start+y_width)]
    df = df[np.logical_and(
        x_start <= df['X'].to_numpy(), df['X'].to_numpy() < x_start+x_width)]
    return df

with open('config.yaml') as f:
    config = yaml.safe_load(f)

base_dir = Path('.').parent
data_dir = base_dir / config['data_folder']
im_shape = config['original_shape']
convert_factor = config['convert_factor']
frame = {}
for x in im_shape:
    frame[x] = im_shape[x]/convert_factor
im_bounds = [0,0,frame['lon'],frame['lat']]

roi_size_factor = st.number_input('ROI size (fold)',15,45,30)
roi_size = {
    'Y': im_shape['lat'] // roi_size_factor,
    'X': im_shape['lon'] // roi_size_factor
}
roi_coordinates = {
    'Y': st.number_input('Y position', 0, im_shape['lat'], value=30000),
    'X': st.number_input('X position', 0, im_shape['lon'], value=40000)
}
roi_border = [
    roi_coordinates['X'] - roi_size['X'] // 2,
    roi_size['X'],
    roi_coordinates['Y'] - roi_size['Y'] // 2,
    roi_size['Y']
]

cell_df = pd.read_csv(data_dir / 'cells.csv')
cell_df['Y'] = im_shape['lat'] - cell_df['Y'] - 1
cell_df['lat'] = cell_df['Y'] / convert_factor
cell_df['lon'] = cell_df['X'] / convert_factor

rna_df = pd.read_csv(data_dir / 'rna_labeled.csv')
rna_df['Y'] = im_shape['lat'] - rna_df['Y'] - 1
rna_df = crop_df(*roi_border, rna_df)
rna_df['lat'] = rna_df['Y'] / convert_factor
rna_df['lon'] = rna_df['X'] / convert_factor

rna_df = rna_df.merge(cell_df[['Cell Index','Type']], on='Cell Index')

background_option = st.selectbox('Select background', ['Convex Hull', 'DAPI'])
background_files = config['backgrounds']
rna_layers = []
rna_layers.append(pdk.Layer(
    'BitmapLayer',
    image=config['image_repo']+background_files[background_option],
    bounds=im_bounds,
))

genes = set(rna_df['Gene'])
color_list = [plt.get_cmap('gist_ncar')(i/len(genes)) for i in range(len(genes))]
color_list = list(map(lambda x: str([int(s * 255) for s in x]), color_list))
cell_point_size = st.slider('Point size', 0.1, 2.0, 0.6, 0.1)
for i,gene in enumerate(genes):
    df_slice = rna_df[rna_df['Gene']==gene]
    rna_layers.append(pdk.Layer(
        'ScatterplotLayer',
        data=df_slice,
        get_position='[lon, lat]',
        get_color=color_list[i],
        get_radius=cell_point_size,
        pickable=True,
        auto_highlight=True,
    ))

init_view_state = pdk.ViewState(latitude=roi_coordinates['Y']/convert_factor,longitude=roi_coordinates['X']/convert_factor,zoom=17,pitch=0)
st.pydeck_chart(pdk.Deck(
    map_provider=None,
    views=[pdk.View(type="MapView")],
    initial_view_state=init_view_state,
    layers=rna_layers,
    tooltip={
        "html": "<b>Y:</b> {Y} <br> <b>X:</b> {X} <br> <b>Gene:</b> {Gene} <br> <b>Cell Index:</b> {Cell Index} <br> <b>Type:</b> {Type}",
    },
))