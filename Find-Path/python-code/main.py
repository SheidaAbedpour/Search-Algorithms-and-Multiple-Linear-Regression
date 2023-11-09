import numpy as np
import pandas as pd
import heapq
import math
from timeit import default_timer as timer
from math import radians, sin, cos, sqrt, atan2

df = pd.read_csv('Flight_Data.csv')
#print(df)



# create graph of airports
graph = {}
# stor location of each airport to compute heuristic faster
location = {}

for i, row in df.iterrows():
    source = row['SourceAirport']
    destination = row['DestinationAirport']
    if source not in graph:
        graph[source] = {}
    if destination not in graph:
        graph[destination] = {}
    graph[source][destination] = row

    if source not in location:
        location[source] = {}
    if destination not in location:
        location[destination] = {}
    location[source] = [row['SourceAirport_Latitude'],
                        row['SourceAirport_Longitude'],
                        row['SourceAirport_Altitude']]
    location[destination] = [row['DestinationAirport_Latitude'],
                             row['DestinationAirport_Longitude'],
                             row['DestinationAirport_Altitude']]



# compte weight of graph based on distance, price and flyTime
def compute_weight(distance, price, flyTime):

    w_distance = 0.7
    w_price = 0.2
    w_flyTime = 0.1

    weight = (w_distance * distance) + (w_price * price) + (w_flyTime * flyTime)
    return weight



# dijkstra algorithm
def dijkstra(graph, source, destination):

    distances = {n: float('inf') for n in graph.keys()}
    distances[source] = 0

    priority_queue = [(0, source)]  # (distance, node)

    pervious = {n: None for n in graph.keys()}

    while priority_queue:
        cur_distance, cur_node = heapq.heappop(priority_queue)
        if cur_node == destination: return cur_distance, pervious
        for neighbor, row in graph[cur_node].items():
            weight = compute_weight(row['Distance'], row['Price'], row['FlyTime'])
            new_distance = cur_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                pervious[neighbor] = cur_node
                heapq.heappush(priority_queue, (new_distance, neighbor))



# get path from source to destination
def get_path(pervious, destination):
    path = []
    current = destination
    while current is not None:
        path.insert(0, current)
        current = pervious[current]
    return path


def print_info(path):
    total_distance = 0
    total_price = 0
    total_fly_time = 0
    c_row = None

    for i in range(len(path)):

        if (i < len(path) - 1):
            c_row = (df[((df['SourceAirport'] == path[i]) & (df['DestinationAirport'] == path[i + 1]))].iloc[0])
            print(c_row)

            total_distance += c_row['Distance']
            total_price += c_row['Price']
            total_fly_time += c_row['FlyTime']

    print("distance: ", total_distance, "  price: ", total_price, "  fly_time: ", total_fly_time)


# test dijkstra algorithm
airport_source = 'Imam Khomeini International Airport'
airport_destination = 'Raleigh Durham International Airport'

start_time = timer()
min_distance, pervious = dijkstra(graph, airport_source, airport_destination)
end_time = timer()

path = get_path(pervious, airport_destination)
print_info(path)

print()
print(" > ".join(path))
time = end_time - start_time
print("time : ", time)


# ----------------------------------------------------------------------------------------------------------------------
# A* algorithm



def calculate_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    # Convert coordinates from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Earth radius in kilometers
    earth_radius = 6371.0

    # Calculate differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    dalt = alt2 - alt1

    # Haversine formula
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Calculate distance
    distance = earth_radius * c

    # Add altitude difference
    distance += abs(dalt)

    return distance


def heurestic_distance(source, destination):
    d1_source = location[source][0]
    d2_source = location[source][1]
    d3_source = location[source][2]

    d1_destination = location[destination][0]
    d2_destination = location[destination][1]
    d3_destination = location[destination][2]

    heuristic = calculate_distance(d1_source, d2_source, d3_source, d1_destination, d2_destination, d3_destination)

    # price
    # flyTime
    # weight

    return heuristic


def a_star(graph, source, destination):
    g = {n: float('inf') for n in graph.keys()}
    f = {n: float('inf') for n in graph.keys()}

    priority_queue = [(0, source)]
    pervious = {n: None for n in graph.keys()}

    g[source] = 0
    f[source] = heurestic_distance(source, destination)

    while priority_queue:
        cur_distance, cur_node = heapq.heappop(priority_queue)
        if cur_node == destination:
            return cur_distance, pervious
        for neighbor, row in graph[cur_node].items():
            weight = row['Distance']
            new_distance = g[cur_node] + weight
            if new_distance < g[neighbor]:
                g[neighbor] = new_distance
                f[neighbor] = g[neighbor] + heurestic_distance(neighbor, destination)
                pervious[neighbor] = cur_node
                heapq.heappush(priority_queue, (f[neighbor], neighbor))


# test A* algorithm
airport_source = 'Imam Khomeini International Airport'
airport_destination = 'Raleigh Durham International Airport'

start_time = timer()
min_distance, pervious = dijkstra(graph, airport_source, airport_destination)
end_time = timer()

path = get_path(pervious, airport_destination)
print_info(path)

print()
print(" > ".join(path))
time = end_time - start_time
print("time : ", time)
