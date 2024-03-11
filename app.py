#Import libraries
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score as ss
import pandas as pd 
import numpy as np
import plotly.express as px
import itertools as it 
import folium
from folium.plugins import GroupedLayerControl
import webbrowser

#Function to create map 
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

#Function to search through Parameters
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
                print("Loop number:",loop_count,"| Parameters:", epsilon,",", min_sample,"| Score for Trafic Level 1:", score,"| Number of clusters:", num_clus)
        loop_count += 1



#Original dataframe
original_df = pd.read_csv('Street Data_BaDinh.csv')
#Drop duplicates
new_df = original_df.drop_duplicates(subset=['Longitude','Latitude'])

#Count Missing Value
print(new_df.isnull().sum(), "\n")
#Check Datatype
print(new_df.dtypes)

fig = px.bar(new_df, x="Ward", color=new_df["Traffic Level"].astype(str), color_discrete_map={"3": "red","1": "green","2": "yellow"})
#fig.show()

#Cordinates and Traffic Level
main = new_df.iloc[:,[7,8,9]]
main.head(10)
#Cordinates for Traffic Level 1 
TF_1 = main[main['Traffic Level'] == 1].iloc[:,[0,1]]
TF_1.head(10)
#Cordinates for Traffic Level 2
TF_2 = main[main['Traffic Level'] == 2].iloc[:,[0,1]]
TF_2.head(10)
#Cordinates for Traffic Level 3
TF_3 = main[main['Traffic Level'] == 3].iloc[:,[0,1]]
TF_3.head(10)

#Setting Param ranges
cluster1 = TF_1.to_numpy()
cluster2 = TF_2.to_numpy()
cluster3 = TF_3.to_numpy()
epsilon1 = np.linspace(0.0003, 0.0049, num=1000).tolist()
epsilon2 = np.linspace(0.0013, 0.0073, num=1000).tolist()
epsilon3 = np.linspace(0.001, 0.0098, num=1000).tolist()
min_samples = [3,4,5,6,7]

getScore(TF_1,epsilon1,min_samples,min_score=0.414,min_num_clus=3)
getScore(TF_2,epsilon2,min_samples,min_score=0.461,min_num_clus=3)
getScore(TF_3,epsilon3,min_samples,min_score=0.517,min_num_clus=3)

##### Training #####

#Training Model for Traffic Level 1
model1 = DBSCAN(eps=0.00296,min_samples=7).fit(cluster1)
TF_1['labels'] = model1.labels_
print((TF_1['labels']).value_counts())
score1 = ss(cluster1, TF_1['labels'])
print("Silhouette score:", score1)

#Plot for Model 1
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
#after_fig1.show()

#Training Model for Traffic Level 2
model2 = DBSCAN(0.00481, min_samples=4).fit(cluster2)
TF_2['labels'] = model2.labels_
print(TF_2['labels'].value_counts())
score2 = ss(cluster2, TF_2['labels'])
print("Silhouette score:", score2)

#Plot for Model 2
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
#after_fig2.show()

#Training Model for Traffic Level 3
model3 = DBSCAN(0.0058,min_samples=3).fit(cluster3) 
TF_3['labels']= model3.labels_
print(TF_3['labels'].value_counts())
score3 = ss(cluster3, TF_3['labels'])
print("Silhouette score:", score3)

#Plot for Model 3
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
#after_fig3.show()

##### Map #####
FinalMap = createMap(TF_1, TF_2,TF_3)  
FinalMap.save('traffic_map.html')
webbrowser.open('traffic_map.html')


