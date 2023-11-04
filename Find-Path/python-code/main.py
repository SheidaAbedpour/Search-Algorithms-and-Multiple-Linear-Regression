import numpy as np
import pandas as pd
import heapq

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

