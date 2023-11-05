import numpy as np
import pandas as pd
import heapq
import math
from timeit import default_timer as timer

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


# test dijkstra algorithm
airport_source = 'Imam Khomeini International Airport'
airport_destination = 'Raleigh Durham International Airport'

start_time = timer()
min_distance, pervious = dijkstra(graph, airport_source, airport_destination)
end_time = timer()

path = get_path(pervious, airport_destination)

print(min_distance)
print(" > ".join(path))

time = end_time - start_time
print("time : ", time)


# ----------------------------------------------------------------------------------------------------------------------

# A* algorithm

def heurestic_distance(source, destination):

    d1_source = location[source][0]
    d2_source = location[source][1]
    d3_source = location[source][2]

    d1_destination = location[destination][0]
    d2_destination = location[destination][1]
    d3_destination = location[destination][2]

    d1 = d1_source - d1_destination
    d2 = d2_source - d2_destination
    d3 = d3_source - d3_destination

    distance = math.sqrt(d1 ** 2 + d2 ** 2 + d3 ** 2)
    # price = row['Price']
    # flyTime = row['FlyTime']

    # weight = compute_weight(distance, price, flyTime)
    return distance


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
start_time = timer()
min_d, pervious = a_star(graph, airport_source, airport_destination)
end_time = timer()

path = get_path(pervious, airport_destination)

print(min_d)
print(" > ".join(path))
time = end_time - start_time
print("time : ", time)
