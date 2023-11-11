import numpy as np
import pandas as pd
import heapq
import math
import time
from math import radians, sin, cos, sqrt, atan2


airport_source = 'Imam Khomeini International Airport'
airport_destination = 'Raleigh Durham International Airport'

#df = pd.read_csv('Flight_Data.csv')
df = pd.read_csv('Dataset.csv')


class Node:
    def __init__(self, name, city, country, latitude, longitude, altitude):
        self.name = name
        self.city = city
        self.country = country
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.neighbors = {}

    def add_neighbor(self, neighbor, airline, distance, price, fly_time):
        self.neighbors[neighbor] = {'Airline': airline, 'Distance': distance, 'Price': price, 'FlyTime': fly_time}


nodes = {}
for _,row in df.iterrows():
    source = row['SourceAirport']
    destination = row['DestinationAirport']

    if source not in nodes:
        nodes[source] = Node(row['SourceAirport'],
                             row['SourceAirport_City'], row['SourceAirport_Country'],
                             row['SourceAirport_Latitude'], row['SourceAirport_Longitude'],
                             row['SourceAirport_Altitude'])
    if destination not in nodes:
        nodes[destination] = Node(row['DestinationAirport'],
                                  row['DestinationAirport_City'], row['DestinationAirport_Country'],
                                  row['DestinationAirport_Latitude'], row['DestinationAirport_Longitude'],
                                  row['DestinationAirport_Altitude'])

    nodes[source].add_neighbor(nodes[destination], row['Airline'], row['Distance'], row['Price'], row['FlyTime'])


def compute_weight(distance, price, flyTime):
    w_distance = 1.7
    w_price = 1.2
    w_flyTime = 1.1

    weight = (w_distance * distance) + (w_price * price) + (w_flyTime * flyTime)
    return weight

def dijkstra(source_name, destination_name):
    source = nodes[source_name]
    destination = nodes[destination_name]

    if source.name == destination.name:
        return {source: None}

    pervious = {n: None for n in nodes.values()}
    distances = {n: float('inf') for n in nodes.values()}

    distances[source] = 0
    priority_queue = [(0, source)]

    counter_dijkstra = 0
    while priority_queue:
        counter_dijkstra += 1
        cur_distance, cur_node = heapq.heappop(priority_queue)

        if cur_node.name == destination.name:
            return cur_distance, pervious

        for neighbor, edge in cur_node.neighbors.items():
            weight = compute_weight(edge['Distance'], edge['Price'], edge['FlyTime'])
            new_distance = cur_distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                pervious[neighbor] = cur_node
                heapq.heappush(priority_queue, (new_distance, neighbor))


def get_path(pervious, destination):
    path = []
    current = nodes[destination]
    while current is not None:
        path.insert(0, current)
        current = pervious[current]
    return path

def print_info(path):
    total_distance = 0
    total_price = 0
    total_fly_time = 0
    result_str = ""

    for i in range(len(path)):
        node = path[i]
        if i < len(path) - 1:
            next_node = path[i + 1]
            edge = node.neighbors[next_node]
            result_str += ("Flight #"+str(i + 1)+"("+node.neighbors[next_node]['Airline']+")"+"\n")
            result_str += ("From: "+node.name+" - "+node.city+", "+ node.country+"\n")
            result_str += ("To: "+next_node.name+" - "+next_node.city+", "+next_node.country+"\n")
            result_str += ("Duration: {:.2f}km".format(edge['Distance'])+"\n")
            result_str += ("Time: {:.2f}h".format(edge['FlyTime'])+"\n")
            result_str += ("Price: {:.2f}$".format(edge['Price'])+"\n")
            result_str += ("----------------------------"+"\n")

            total_distance += edge['Distance']
            total_price += edge['Price']
            total_fly_time += edge['FlyTime']

    result_str += ("Total Price: {:.2f}$".format(total_price)+"\n")
    result_str += ("Total Duration: {:.2f}km".format(total_distance)+"\n")
    result_str += ("Total Time: {:.2f}h".format(total_fly_time)+"\n")
    return result_str


start_time = time.time()
num_visited_nodes, pervious = dijkstra(airport_source, airport_destination)
elapsed_time = time.time()  - start_time

str_path_dijkstra = ("Dijkstra Algorithm"+"\n")
minutes, seconds = divmod(elapsed_time, 60)
str_path_dijkstra += ("Execution Time: {:.0f}m{:.5f}s".format(minutes, seconds)+"\n")
str_path_dijkstra += (".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-"+"\n")

path = get_path(pervious, airport_destination)
str_path_dijkstra += print_info(path)
print(str_path_dijkstra)
print("number of visited nodes: ", num_visited_nodes, "\n")

print("***********************")
# ----------------------------------------------------------------------------------------------------------------------

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
    d1_source = source.latitude
    d2_source = source.longitude
    d3_source = source.altitude

    d1_destination = destination.latitude
    d2_destination = destination.longitude
    d3_destination = destination.altitude

    heuristic = math.sqrt(((d1_source - d1_destination) ** 2) + ((d2_source - d2_destination) ** 2) +
                          ((d3_source - d3_destination) ** 2))
    #heuristic = calculate_distance(d1_source, d2_source, d3_source, d1_destination, d2_destination, d3_destination)
    return heuristic

def a_star(source_name, destination_name):

    source = nodes[source_name]
    destination = nodes[destination_name]

    g = {n: float('inf') for n in nodes.values()}
    f = {n: float('inf') for n in nodes.values()}
    pervious = {n: None for n in nodes.values()}
    priority_queue = [(0, source)]

    g[source] = 0
    f[source] = heurestic_distance(source, destination)

    counter_a_star = 0
    while priority_queue:
        counter_a_star += 1
        cur_distance, cur_node = heapq.heappop(priority_queue)

        if cur_node.name == destination.name:
            return counter_a_star, pervious

        for neighbor, edge in cur_node.neighbors.items():
            weight = edge['Distance']
            new_distance = g[cur_node] + weight
            if new_distance < g[neighbor]:
                g[neighbor] = new_distance
                f[neighbor] = g[neighbor] + heurestic_distance(neighbor, destination)
                pervious[neighbor] = cur_node
                heapq.heappush(priority_queue, (f[neighbor], neighbor))


start_time = time.time()
num_visited_nodes, pervious = a_star(airport_source, airport_destination)
elapsed_time = time.time()  - start_time

str_path_a_star = ("A* Algorithm"+"\n")
minutes, seconds = divmod(elapsed_time, 60)
str_path_a_star += ("Execution Time: {:.0f}m{:.5f}s".format(minutes, seconds)+"\n")
str_path_a_star += (".-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-"+"\n")

path = get_path(pervious, airport_destination)
str_path_a_star += print_info(path)
print(str_path_a_star)
print("number of visited nodes: ", num_visited_nodes)

# Open the file for writing
with open('17-UIAI4021-PR1-Q1(A-Star).txt', 'w', encoding='utf-8') as file:
    file.write(str_path_a_star)
