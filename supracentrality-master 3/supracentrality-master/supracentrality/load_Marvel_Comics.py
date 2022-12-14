import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

def remove_nodes(G, low_deg, low_layer, weight, *args):
    ''' remove nodes from a networkx graph G, that have too low a degree (lower than or equal to low_deg) and
    exist in too few layers (lower than or equal to low_layer). 
    The other layers (of form of networkx graphs as well) are given in args.
    For weighted graph, can use weight = 'weight'
    '''
    need_remove = []
    for node in G.nodes():
        layers = 1
        for L in args:
            if (L.has_node(node) == True):
                layers = layers + 1
        if ((G.degree[node] <= low_deg) & (layers <= low_layer)):
            need_remove.append(node)
    print(need_remove)
    G.remove_nodes_from(need_remove)

def remove_non_common(G, comm_heroes):
    node_lis = list(G.nodes)
    remove_lis = []
    for i in node_lis:
        if i not in comm_heroes:
            remove_lis += [i]
            
    G.remove_nodes_from(remove_lis)
    return G

################################################
## Load all nodes temporal network
################################################



################################################
## Load temporal network 
################################################


def load_temporal_MC_flow_graph(foldername):
    # load graph from file
    data = pd.read_csv(foldername+'/edges.csv') 
    graph = {}
    
    # temporal periods name
    graph['layer_names'] = ["silver", "bronze", "modern", "heroes"]
    graph['T'] = len(graph['layer_names'])
    
    # data cleaning
    filter1 = data["comic"].str.startswith(("A ", "A2", "A3"))
    data[filter1].drop_duplicates(subset = "comic")
    
    layers = []
    for i in ["A ", "A2", "A3"]:
        filters = data["comic"].str.startswith(i)
        layers.append(data[filters])
        
    layers[0]['issue'] = layers[0]['comic'].str[2:]
    issues1 = layers[0][~layers[0]['issue'].str.startswith("'")]
    for i in range(len(issues1)):
        issues1.iloc[i, 2] = issues1.iloc[i, 2].split("/",1)[0]
        issues1.iloc[i, 2] = issues1.iloc[i, 2].split("-",1)[0]
    
    # first construct four bipartite networks
    silver = issues1[issues1['issue'].astype(float) <= 82]
    silver = silver[["hero", "comic"]]
    silver.drop_duplicates(subset = "hero")
    
    bronze = issues1[(issues1['issue'].astype(float) > 82) & (issues1['issue'].astype(float) <= 242)]
    bronze = bronze[["hero", "comic"]]
    bronze.drop_duplicates(subset = "hero")
    
    modern = issues1[issues1['issue'].astype(float) > 242]
    modern = modern[["hero", "comic"]]
    modern = pd.concat([modern, layers[1]])
    modern.drop_duplicates(subset = "hero")
    
    heroes = pd.concat([layers[2], layers[0][layers[0]['issue'].str.startswith("'")]])
    heroes.drop_duplicates(subset = "hero")
    
    # construct adjacency matrices
    B = nx.Graph()
    B.add_nodes_from(silver['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(silver['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in silver.iterrows()])
    Gsilver = bipartite.projected_graph(B, silver['hero'])
    Gsilver_weighted = bipartite.weighted_projected_graph(B, silver['hero'].drop_duplicates())
    
    B = nx.Graph()
    B.add_nodes_from(bronze['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(bronze['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in bronze.iterrows()])
    Gbronze = bipartite.projected_graph(B, bronze['hero'])
    Gbronze_weighted = bipartite.weighted_projected_graph(B, bronze['hero'].drop_duplicates())
    
    B = nx.Graph()
    B.add_nodes_from(modern['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(modern['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in modern.iterrows()])
    Gmodern = bipartite.projected_graph(B, modern['hero'])
    Gmodern_weighted = bipartite.weighted_projected_graph(B, modern['hero'].drop_duplicates())
    
    B = nx.Graph()
    B.add_nodes_from(heroes['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(heroes['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in heroes.iterrows()])
    Gheroes = bipartite.projected_graph(B, heroes['hero'])
    Gheroes_weighted = bipartite.weighted_projected_graph(B, heroes['hero'].drop_duplicates())
    
    networks = {'silver': Gsilver_weighted, 'bronze': Gbronze_weighted, 'modern': Gmodern_weighted, 'heroes': Gheroes_weighted}
    
    # find common heros
    comm_heroes = list(set(silver['hero']).intersection(bronze['hero']))
    comm_heroes = list(set(comm_heroes).intersection(modern['hero']))
    comm_heroes = list(set(comm_heroes).intersection(heroes['hero']))
    
        
    # a tensor containing network adjacency matrix at each time
    graph['A_tensor'] = []
    for layer in graph['layer_names']:
        network = remove_non_common(networks[layer], comm_heroes)
        network_matrix = nx.to_numpy_array(network, sorted(list(network.nodes())))
        print(sorted(list(network.nodes())))
        graph['nodenames'] = sorted(list(network.nodes()))
        graph['N'] = len(graph['nodenames'])
        graph['A_tensor'].append(network_matrix)

    
    return graph