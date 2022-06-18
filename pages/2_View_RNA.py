import string
from pathlib import Path
import yaml
import streamlit as st
import pydeck as pdk
import numpy as np
from skimage.io import imread
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

scale_factor = 20
def compute_grid(y,x,scale_factor):
    y_unit, x_unit = y/scale_factor, x/scale_factor
    y_coord, x_coord = np.meshgrid(np.arange(scale_factor)*y_unit, np.arange(scale_factor)*x_unit)
    y_coord = y - (y_coord + y_unit / 2).T
    x_coord = (x_coord + x_unit / 2).T
    return y_coord, x_coord


roi_size = {
    'Y': im_shape['lat'] / scale_factor,
    'X': im_shape['lon'] / scale_factor
}
grid_coordinates = {
    'Y': im_shape['lat'] - (np.arange(scale_factor)*roi_size['Y'] + roi_size['Y']/2),
    'X': np.arange(scale_factor)*roi_size['X'] + roi_size['X']/2
}
# grid_coordinates['Y'], grid_coordinates['X'] = compute_grid(im_shape['lat'], im_shape['lon'], scale_factor)
grid_im = imread(data_dir / config['grid_image'])
grid_roi = st.text_input('Enter ROI:', value='G08')
st.image(grid_im)

alphabet_dict = dict(zip(list(string.ascii_uppercase)[:scale_factor],[i for i in range(scale_factor)]))
number_dict = dict(zip([i+1 for i in range(scale_factor)],[i for i in range(scale_factor)]))
grid_roi_dict = {}
try:
    grid_roi_dict['Y'] = alphabet_dict[grid_roi[0]]
    grid_roi_dict['X'] = int(grid_roi[1:]) - 1
except KeyError:
    st.error('Please input a valid ROI')

roi_coordinates = {
    'Y': grid_coordinates['Y'][grid_roi_dict['Y']],
    'X': grid_coordinates['X'][grid_roi_dict['X']]
}
roi_border = [
    roi_coordinates['X'] - roi_size['X'] / 2,
    roi_size['X'],
    roi_coordinates['Y'] - roi_size['Y'] / 2,
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
rna_point_size = st.slider('Point size', 0.1, 2.0, 0.6, 0.1)
for i,gene in enumerate(genes):
    df_slice = rna_df[rna_df['Gene']==gene]
    rna_layers.append(pdk.Layer(
        'ScatterplotLayer',
        data=df_slice,
        get_position='[lon, lat]',
        get_color=color_list[i],
        get_radius=rna_point_size,
        pickable=True,
        auto_highlight=True,
    ))

init_view_state = pdk.ViewState(latitude=roi_coordinates['Y']/convert_factor,longitude=roi_coordinates['X']/convert_factor,zoom=17,pitch=0,min_zoom=16)
rna_deck = pdk.Deck(
    map_provider=None,
    views=[pdk.View(type="MapView")],
    initial_view_state=init_view_state,
    layers=rna_layers,
    tooltip={
        "html": "<b>Y:</b> {Y} <br> <b>X:</b> {X} <br> <b>Gene:</b> {Gene} <br> <b>Cell Index:</b> {Cell Index} <br> <b>Type:</b> {Type}",
    },
)

st.pydeck_chart(rna_deck)