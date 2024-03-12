from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
import pandas as pd 
import numpy as np
import plotly.express as px
import itertools as it 
import folium
from folium.plugins import GroupedLayerControl
import webbrowser
import streamlit as st
from flask import Flask

def getScore(dataset,epss,min_sampless,min_score,min_num_clus):
    if epss == epsilon1:
        cluster = cluster1
    elif epss == epsilon2:
        cluster = cluster2
    elif epss == epsilon3:
        cluster = cluster3
    loop_count = 1
    #Loop through parameters from combinations
    for i, (epsilon, min_sample) in enumerate(list(it.product(epss, min_sampless))):
        #Fit model
        model = DBSCAN(eps = epsilon, min_samples = min_sample).fit(cluster)
        #Get lables
        dataset['labels'] = model.labels_
        num_clus = len(dataset['labels'].value_counts()) 
        #Get scores
        if num_clus > 2:
            score = ss(cluster, dataset['labels'])
            if score >= min_score and num_clus > min_num_clus: 
                st.write("Loop number:",loop_count,"| Parameters:", epsilon,",", min_sample,"| Score for Trafic Level 1:", score,"| Number of clusters:", num_clus)
        loop_count += 1

##### App Creation #####
st.set_page_config(layout="wide")
st.title("Traffic Analysis")

#####Side Bar
uploaded_df = st.sidebar.file_uploader("Upload Traffic File", type='csv')
#Dataframe
if uploaded_df is not None:
    original_df = pd.read_csv(uploaded_df)
new_df = original_df.drop_duplicates(subset=['Longitude','Latitude'])

levels = st.sidebar.radio('Traffic Levels', options=["Level 1","Level 2", "Level 3"])

st.sidebar.header("Parameters")

main = new_df.iloc[:,[7,8,9]]
TF_1 = main[main['Traffic Level'] == 1].iloc[:,[0,1]]
TF_2 = main[main['Traffic Level'] == 2].iloc[:,[0,1]]
TF_3 = main[main['Traffic Level'] == 3].iloc[:,[0,1]]

#Params
cluster1 = TF_1.to_numpy()
cluster2 = TF_2.to_numpy()
cluster3 = TF_3.to_numpy()
epsilon1 = np.linspace(0.0003, 0.0049, num=1000).tolist()
epsilon2 = np.linspace(0.0013, 0.0073, num=1000).tolist()
epsilon3 = np.linspace(0.001, 0.0098, num=1000).tolist()
min_samples = [3,4,5,6,7]

#Body
col1, col2, col3 = st.columns([0.7,0.15,0.15])
with col1:
    with st.container():
        if levels == "Level 1":
            sel_eps = st.sidebar.slider('Epsilon', min_value=0.00030, max_value=0.0098, step=0.00001, format="%f")
            sel_minpts = st.sidebar.slider('Minimun Points', 3, 10)
            model1 = DBSCAN(eps= sel_eps, min_samples=sel_minpts).fit(cluster1)
            TF_1['labels'] = model1.labels_
            after_fig1 = px.scatter(TF_1, 
                                x = "Latitude",
                                y = "Longitude", 
                                color=TF_1['labels'].astype(str),
                                color_discrete_map={'-1':'red', '0': '#636EFA'},
                                labels={"color" : "Cluster"},
                            ) 
            after_fig1.update_layout(   
                    title={
                    'text' : "Area with traffic level 1 with eps = 0.00296, min_samples = 7",
                    'x':0.5,
                    'xanchor': 'center',
                })
            st.plotly_chart(after_fig1)
    with st.container():
        if levels == "Level 2":
            sel_eps = st.sidebar.slider('Epsilon', min_value=0.00030, max_value=0.0098, step=0.00001, format="%f")
            sel_minpts = st.sidebar.slider('Minimun Points', 3, 10)
            model2 = DBSCAN(eps= sel_eps, min_samples=sel_minpts).fit(cluster2)
            TF_2['labels'] = model2.labels_
            after_fig2 = px.scatter(TF_2,
                                x = "Latitude" , 
                                y = "Longitude", 
                                color=TF_2['labels'].astype(str),
                                labels={"color" : "Cluster"},
                                )
            after_fig2.update_layout(
                    title={
                    'text' : "Area with traffic level 2 with eps = 0.00481, min_samples = 4",
                    'x':0.5,
                    'xanchor': 'center',
                })
            st.plotly_chart(after_fig2)
    with st.container():
        if levels == "Level 3":
            sel_eps = st.sidebar.slider('Epsilon', min_value=0.00030, max_value=0.0098, step=0.00001, format="%f")
            sel_minpts = st.sidebar.slider('Minimun Points', 3, 7)
            model3 = DBSCAN(eps= sel_eps, min_samples=sel_minpts).fit(cluster3)
            TF_3['labels'] = model3.labels_
            after_fig3 = px.scatter(TF_3,
                                x = "Latitude" , 
                                y = "Longitude", 
                                color=TF_3['labels'].astype(str),
                                labels={"color" : "Cluster"},
                                )
            after_fig3.update_layout(
                    title={
                    'text' : "Area with traffic level 3 with eps = 0.0058, min_samples = 3",
                    'x':0.5,
                    'xanchor': 'center',
                })
            st.plotly_chart(after_fig3)

with col2:
    st.write("Missing Values",new_df.isnull().sum())
with col3:    
    st.write("Datatypes",new_df.dtypes)

st.header("Find best Parameters")
sel_score = st.slider("Set Score", min_value=0.001, max_value=0.999, step=0.001, format="%f", value=0.4)
with st.container():
    if st.button("Find"):
        if levels == "Level 1":
            getScore(TF_1,epsilon1,min_samples,min_score=sel_score,min_num_clus=3)
        if levels == "Level 2":
            getScore(TF_2,epsilon2,min_samples,min_score=sel_score,min_num_clus=3)
        if levels == "Level 3":
            getScore(TF_3,epsilon3,min_samples,min_score=sel_score,min_num_clus=3)


eps1, min1 = 0.005,5
eps2, min2 = 0.005,5
eps3, min3 = 0.005,5
with st.sidebar.form("Traffic Level 1"):
    input_eps1 = st.number_input("Epsilon 1", min_value=0.00030, max_value=0.0098, step=0.00001, format="%f")
    input_min1 = st.number_input("Minimun Points 1", min_value=3, max_value=10)
    input_eps2 = st.number_input("Epsilon 2", min_value=0.00030, max_value=0.0098, step=0.00001, format="%f")
    input_min2 = st.number_input("Minimun Points 2", min_value=3, max_value=10)
    input_eps3 = st.number_input("Epsilon 3", min_value=0.00030, max_value=0.0098, step=0.00001, format="%f")
    input_min3 = st.number_input("Minimun Points 3", min_value=3, max_value=10)
    submitted = st.form_submit_button()
    if submitted:
        eps1 = input_eps1
        min1 = input_min1
        eps2 = input_eps2
        min2 = input_min2
        eps3 = input_eps3
        min3 = input_min3

model1 = DBSCAN(eps=eps1, min_samples=min1).fit(cluster1)
TF_1['labels'] = model1.labels_
model2 = DBSCAN(eps=eps2, min_samples=min2).fit(cluster2)
TF_2['labels'] = model2.labels_
model3 = DBSCAN(eps=eps3, min_samples=min3).fit(cluster3)
TF_3['labels'] = model3.labels_


def createMap(dataset1, dataset2, dataset3):
    m = folium.Map(location=[21.034388, 105.831716], zoom_start=14, tiles='cartodbpositron')
    group_tf_1 = folium.FeatureGroup("Traffic Level 1").add_to(m)
    group_tf_2 = folium.FeatureGroup("Traffic Level 2").add_to(m)
    group_tf_3 = folium.FeatureGroup("Traffic Level 3").add_to(m)
    labs1 = dataset1['labels'].to_numpy()
    labs2 = dataset2['labels'].to_numpy()
    labs3 = dataset3['labels'].to_numpy()
    long_lat1 = dataset1.iloc[:,[0,1]].to_numpy()
    long_lat2 = dataset2.iloc[:,[0,1]].to_numpy()
    long_lat3 = dataset3.iloc[:,[0,1]].to_numpy()
    for i in range (0, len(labs1)):
        if labs1[i] == 0:
            folium.CircleMarker(long_lat1[i], radius= 1, color="#636EFA",  fill_opacity=0.9).add_to(group_tf_1)
        if labs1[i] == 1:
            folium.CircleMarker(long_lat1[i], radius= 1, color="#00CC96",  fill_opacity=0.9).add_to(group_tf_1)
        if labs1[i] == 2: 
            folium.CircleMarker(long_lat1[i], radius= 1, color="#AB63FA",  fill_opacity=0.9).add_to(group_tf_1)
        if labs1[i] == -1: 
            folium.CircleMarker(long_lat1[i], radius= 1, color="#EF553B",  fill_opacity=0.9).add_to(group_tf_1)
    for i in range (0, len(labs2)):
        if labs2[i] == 0:
            folium.CircleMarker(long_lat2[i], radius= 1, color="#636EFA",  fill_opacity=0.9).add_to(group_tf_2)
        if labs2[i] == 1:
            folium.CircleMarker(long_lat2[i], radius= 1, color="#00CC96",  fill_opacity=0.9).add_to(group_tf_2)
        if labs2[i] == 2: 
            folium.CircleMarker(long_lat2[i], radius= 1, color="#AB63FA",  fill_opacity=0.9).add_to(group_tf_2)
        if labs2[i] == -1: 
            folium.CircleMarker(long_lat2[i], radius= 1, color="#EF553B",  fill_opacity=0.9).add_to(group_tf_2)
    for i in range (0, len(labs3)):
        if labs3[i] == 0:
            folium.CircleMarker(long_lat3[i], radius= 1, color="#636EFA",  fill_opacity=0.9).add_to(group_tf_3)
        if labs3[i] == 1:
            folium.CircleMarker(long_lat3[i], radius= 1, color="#00CC96",  fill_opacity=0.9).add_to(group_tf_3)
        if labs3[i] == 2: 
            folium.CircleMarker(long_lat3[i], radius= 1, color="#AB63FA",  fill_opacity=0.9).add_to(group_tf_3)
        if labs3[i] == -1: 
            folium.CircleMarker(long_lat3[i], radius= 1, color="#EF553B",  fill_opacity=0.9).add_to(group_tf_3)
    GroupedLayerControl(
        groups={'Traffic Levels': [group_tf_1, group_tf_2,group_tf_3]},
        collapsed=False,
    ).add_to(m)
    folium.LayerControl(collapsed=False)
    return m

def map_click():
    #st.write(TF_1,TF_2,TF_3)
    FinalMap=createMap(TF_1, TF_2,TF_3)
    FinalMap.save('traffic_map.html')
    webbrowser.open('traffic_map.html')

st.sidebar.button("Create Map", on_click=map_click)
