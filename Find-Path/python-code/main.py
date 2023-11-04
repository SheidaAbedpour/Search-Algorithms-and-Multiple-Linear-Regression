import numpy as np
import pandas as pd
import heapq
import math

df = pd.read_csv('Flight_Data.csv')
#print(df)


# create graph of airports
graph = {}
for i, row in df.iterrows():
    source = row['SourceAirport']
    destination = row['DestinationAirport']
    if source not in graph:
        graph[source] = {}
    if destination not in graph:
        graph[destination] = {}
    graph[source][destination] = i



# dijkstra algorithm
def dijkstra(graph, source, destination):

    # store distances from source to every node
    distances = {n: float('inf') for n in graph.keys()}
    distances[source] = 0

    priority_queue = [(0, source)]  # (distance, node)

    # store the path
    pervious = {n: None for n in graph.keys()}

    while priority_queue:

        # get node with minimum distance
        cur_distance, cur_node = heapq.heappop(priority_queue)

        if cur_node == destination:
            return cur_distance, pervious

        # update distance of neighbors
        for neighbor, index in graph[cur_node].items():
            weight = df.loc[index, 'Distance']
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


# test dijkstra algorithm
airport_source = 'Imam Khomeini International Airport'
airport_destination = 'Raleigh Durham International Airport'

min_distance, pervious = dijkstra(graph, airport_source, airport_destination)
path = get_path(pervious, airport_destination)

print(min_distance)
print(" > ".join(path))


# ----------------------------------------------------------------------------------------------------------------------

# A* algorithm

def heurestic(source, destination):

    # if there is no path from source to destination return
    if df[(df['SourceAirport'] == source) & (df['DestinationAirport'] == destination)].empty:
        return 0;

    # get row in df
    row_source = df[df['SourceAirport'] == source].iloc[0]
    row_destination = df[df['DestinationAirport'] == destination].iloc[0]

    #get index of row
    index_source = df[df.eq(row_source).all(axis=1)].index[0]
    index_destination = df[df.eq(row_destination).all(axis=1)].index[0]

    # get location of source
    d1_source = df.loc[index_source, 'SourceAirport_Latitude']
    d2_source = df.loc[index_source, 'SourceAirport_Longitude']
    d3_source = df.loc[index_source, 'SourceAirport_Altitude']

    # get location of destination
    d1_destination = df.loc[index_destination, 'DestinationAirport_Latitude']
    d2_destination = df.loc[index_destination, 'DestinationAirport_Longitude']
    d3_destination = df.loc[index_destination, 'DestinationAirport_Altitude']

    # compute distance
    d1 = d1_source - d1_destination
    d2 = d2_source - d2_destination
    d3 = d3_source - d3_destination

    heurestic = math.sqrt(d1 ** 2 + d2 ** 2 + d3 ** 2)
    return heurestic


def a_star(graph, source, destination):

    # define g(x) and f(x)
    g = {n: float('inf') for n in graph.keys()}
    f = {n: float('inf') for n in graph.keys()}

    priority_queue = [(0, source)]
    pervious = {n: None for n in graph.keys()}

    g[source] = 0
    f[source] = heurestic(source, destination)

    while priority_queue:

        # get the node with min distance
        cur_distance, cur_node = heapq.heappop(priority_queue)

        if cur_node == destination:
            return cur_distance, pervious

        # update distance of neighbors
        for neighbor, index in graph[cur_node].items():
            weight = df.loc[index, 'Distance']
            new_distance = g[cur_node] + weight
            if new_distance < g[neighbor]:
                g[neighbor] = new_distance
                f[neighbor] = g[neighbor] + heurestic(neighbor, destination)
                pervious[neighbor] = cur_node
                heapq.heappush(priority_queue, (f[neighbor], neighbor))


# test A* algorithm
min_d, pervious = a_star(graph, airport_source, airport_destination)
path = get_path(pervious, airport_destination)

print(min_d)
print(" > ".join(path))




