import numpy as np
import pandas as pd
import heapq
import math
from timeit import default_timer as timer
from math import radians, sin, cos, sqrt, atan2

#load dataset
df = pd.read_csv('Flight_Data.csv')


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


# compte weight of graph based on distance, price and flyTime
def compute_weight(distance, price, flyTime):

    w_distance = 1.7
    w_price = 1.2
    w_flyTime = 1.1

    weight = (w_distance * distance) + (w_price * price) + (w_flyTime * flyTime)
    return weight



# dijkstra algorithm
def dijkstra(source_name, destination_name):
    """
        Dijkstra's algorithm to find the shortest path in a graph from a source node to a destination node.

        Args:
            source (str): The source node.
            destination (str): The destination node.

        Returns:
            tuple: A tuple containing the shortest distance and the path from source to destination.

        Raises:
            ValueError: If the source or destination node is invalid.

        """

    source = nodes[source_name]
    destination = nodes[destination_name]

    # Previous nodes to track the path
    pervious = {n: None for n in nodes.values()}

    #if source not in nodes or destination not in nodes:
     #   return 0, pervious

    if source.name == destination.name:
        return 0, {source: None}

    #distance from each airport to destination
    distances = {n: float('inf') for n in nodes.values()}
    distances[source] = 0
    # Priority queue initialized with (cost, node)
    priority_queue = [(0, source)]

    while priority_queue:
        #pop airport with minmum distance
        cur_distance, cur_node = heapq.heappop(priority_queue)
        if cur_node.name == destination.name: return cur_distance, pervious
        # Explore neighbors of the current node
        for neighbor, edge in cur_node.neighbors.items():
            weight = compute_weight(edge['Distance'], edge['Price'], edge['FlyTime'])
            new_distance = cur_distance + weight
            # Update distance of neighbors and previous node if a shorter path is found
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                pervious[neighbor] = cur_node
                heapq.heappush(priority_queue, (new_distance, neighbor))


# get path from source to destination
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

    for i in range(len(path)):
        node = path[i]
        print(node.name)
        if i < len(path) - 1:
            edge = node.neighbors[path[i + 1]]
            total_distance += edge['Distance']
            total_price += edge['Price']
            total_fly_time += edge['FlyTime']

    print("distance: ", total_distance, "  price: ", total_price, "  fly_time: ", total_fly_time)


# test dijkstra algorithm
airport_source = 'Imam Khomeini International Airport'
airport_destination = 'Raleigh Durham International Airport'

start_time = timer()
min_distance, pervious = dijkstra(airport_source, airport_destination)
end_time = timer()

print("dijkstra")

path = get_path(pervious, airport_destination)
print_info(path)

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
    d1_source = source.latitude
    d2_source = source.longitude
    d3_source = source.altitude

    d1_destination = destination.latitude
    d2_destination = destination.longitude
    d3_destination = destination.altitude

    heuristic = calculate_distance(d1_source, d2_source, d3_source, d1_destination, d2_destination, d3_destination)

    # price
    # flyTime
    # weight

    return heuristic


def a_star(source_name, destination_name):

    source = nodes[source_name]
    destination = nodes[destination_name]

    g = {n: float('inf') for n in nodes.values()}
    f = {n: float('inf') for n in nodes.values()}

    priority_queue = [(0, source)]
    pervious = {n: None for n in nodes.values()}

    g[source] = 0
    f[source] = heurestic_distance(source, destination)

    while priority_queue:
        cur_distance, cur_node = heapq.heappop(priority_queue)
        if cur_node.name == destination.name:
            return cur_distance, pervious
        for neighbor, edge in cur_node.neighbors.items():
            weight = edge['Distance']
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
min_distance, pervious = a_star(airport_source, airport_destination)
end_time = timer()

print("a*")

path = get_path(pervious, airport_destination)
print_info(path)

time = end_time - start_time
print("time : ", time)
