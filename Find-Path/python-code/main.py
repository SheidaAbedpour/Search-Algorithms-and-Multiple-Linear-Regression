import numpy as np
import pandas as pd

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

