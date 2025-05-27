from sklearn.cluster import DBSCAN
import pandas as pd 
import folium
import streamlit as st
from streamlit_folium import folium_static
from heapq import heappop, heappush

#Lấy dữ liệu
uploaded_df = pd.read_csv("app_data.csv")
traffic_coords = pd.concat([uploaded_df[['s_lat', 's_long']].rename(columns={'s_lat': 'lat', 's_long': 'long'}), 
                        uploaded_df[['e_lat', 'e_long']].rename(columns={'e_lat': 'lat', 'e_long': 'long'})], 
                        axis=0).drop_duplicates().reset_index(drop=True)
model_dbscan = DBSCAN(eps= 0.0006, min_samples = 4).fit(traffic_coords[['lat', 'long']])
traffic_coords['dbscan_cluster_labels'] = model_dbscan.labels_
merged_data = uploaded_df.merge(traffic_coords, left_on=['s_lat', 's_long'], right_on=['lat', 'long'], how='left')
merged_data = merged_data.drop(columns=['lat', 'long'])
unique_street_names = merged_data['street_name'].unique().tolist()

#Tạo thuật toán A*
def heuristic(node, goal):
    node_coords = merged_data[merged_data['dbscan_cluster_labels'] == node][['s_lat', 's_long']].mean()
    goal_coords = merged_data[merged_data['dbscan_cluster_labels'] == goal][['s_lat', 's_long']].mean()
    return abs(node_coords['s_lat'] - goal_coords['s_lat']) + abs(node_coords['s_long'] - goal_coords['s_long'])

def get_neighbors(node, cluster_weights, distance_threshold=0.03):
    node_coords = merged_data[merged_data['dbscan_cluster_labels'] == node][['s_lat', 's_long']].mean()
    neighbors = []
    for cluster in cluster_weights.keys():
        if cluster != node:
            cluster_coords = merged_data[merged_data['dbscan_cluster_labels'] == cluster][['s_lat', 's_long']].mean()
            distance = abs(node_coords['s_lat'] - cluster_coords['s_lat']) + abs(node_coords['s_long'] - cluster_coords['s_long'])
            if distance <= distance_threshold:
                neighbors.append(cluster)
    return neighbors

def a_star(start, goal):
    cluster_weights = merged_data[merged_data['slow_traffic'] == 1].groupby('dbscan_cluster_labels').size().to_dict()
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in get_neighbors(current, cluster_weights):
            tentative_g_score = g_score[current] + cluster_weights.get(neighbor, 0)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in [i[1] for i in open_set]:
                    heappush(open_set, (f_score[neighbor], neighbor))
    return None  

#Tạo bản đồ
def create_map(traffic_coords, optimized_path):
    start_coords = traffic_coords[traffic_coords['dbscan_cluster_labels'] == optimized_path[0]]
    end_coords = traffic_coords[traffic_coords['dbscan_cluster_labels'] == optimized_path[-1]]
    map_folium = folium.Map(location=[start_coords['lat'].mean(), start_coords['long'].mean()], zoom_start=14)

    folium.Marker(
        location=[start_coords['lat'].mean(), start_coords['long'].mean()],
        popup="Start Cluster",
        icon=folium.Icon(color="green")
    ).add_to(map_folium)

    folium.Marker(
        location=[end_coords['lat'].mean(), end_coords['long'].mean()],
        popup="End Cluster",
        icon=folium.Icon(color="red")
    ).add_to(map_folium)

    for i in range(len(optimized_path) - 1):
        start = traffic_coords[traffic_coords['dbscan_cluster_labels'] == optimized_path[i]][['lat', 'long']].mean()
        end = traffic_coords[traffic_coords['dbscan_cluster_labels'] == optimized_path[i + 1]][['lat', 'long']].mean()
        folium.PolyLine(
            [(start['lat'], start['long']), (end['lat'], end['long'])],
            color="red",
            weight=2.5,
            tooltip="Path"
        ).add_to(map_folium)
        midpoint = [(end['lat'] + start['lat']) / 2, (end['long'] + start['long']) / 2]
        folium.RegularPolygonMarker(
            location=midpoint,
            number_of_sides=3,
            radius=6,
            rotation=0, 
            color="red",
            fill=True,
            fill_color="red"
        ).add_to(map_folium)

    for cluster in optimized_path:
        cluster_points = traffic_coords[traffic_coords['dbscan_cluster_labels'] == cluster]
        for _, row in cluster_points.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['long']],
                radius=3,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.7
            ).add_to(map_folium)
    return map_folium

#Tạo giao diện
st.set_page_config(layout="wide")
default_map = folium.Map(location=[traffic_coords['lat'].mean(), traffic_coords['long'].mean()], zoom_start=12)
start = st.sidebar.text_input("Chọn điểm bắt đầu", key="start").upper()
end = st.sidebar.text_input("Chọn điểm đến", key="end").upper()
if st.sidebar.button("Submit",key="2"):
    if start and end in [name.upper() for name in unique_street_names]:
        start = merged_data[merged_data['street_name'].str.upper() == start]['dbscan_cluster_labels'].values[0]
        end = merged_data[merged_data['street_name'].str.upper() == end]['dbscan_cluster_labels'].values[0]
        optimized_path = a_star(start, end)
        map_folium=create_map(traffic_coords, optimized_path)
        folium_static(map_folium, width=1300, height=700)
    else:
        st.write('Please select a valid street name.')
        folium_static(default_map, width=1300, height=700)
else: folium_static(default_map, width=1300, height=700)

