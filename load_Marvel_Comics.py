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


def load_temporal_MC_flow_graph(foldername, weighted = True, size = "common", outnet = False):
    # load graph from file
    data = pd.read_csv(foldername+'/edges.csv') 
    graph = {}
    
    # temporal periods name
    graph['layer_names'] = ["silver", "bronze", "modern", "heroes"]
    graph['T'] = len(graph['layer_names'])
    
    # data cleaning
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
    
    bronze = issues1[(issues1['issue'].astype(float) > 82) & (issues1['issue'].astype(float) <= 242)]
    bronze = bronze[["hero", "comic"]]
    
    modern = issues1[issues1['issue'].astype(float) > 242]
    modern = modern[["hero", "comic"]]
    modern = pd.concat([modern, layers[1]])
    
    heroes = pd.concat([layers[2], layers[0][layers[0]['issue'].str.startswith("'")]])
    
    # project the bipartite graphs onto heros
    B = nx.Graph()
    B.add_nodes_from(silver['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(silver['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in silver.iterrows()])
    if weighted == True:
        Gsilver_weighted = bipartite.weighted_projected_graph(B, silver['hero'].drop_duplicates())
    else: 
        Gsilver = bipartite.projected_graph(B, silver['hero'])
    
    B = nx.Graph()
    B.add_nodes_from(bronze['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(bronze['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in bronze.iterrows()])
    if weighted == True:
        Gbronze_weighted = bipartite.weighted_projected_graph(B, bronze['hero'].drop_duplicates())
    else:
        Gbronze = bipartite.projected_graph(B, bronze['hero'])
    
    B = nx.Graph()
    B.add_nodes_from(modern['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(modern['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in modern.iterrows()])
    if weighted == True:
        Gmodern_weighted = bipartite.weighted_projected_graph(B, modern['hero'].drop_duplicates())
    else:
        Gmodern = bipartite.projected_graph(B, modern['hero'])
    
    B = nx.Graph()
    B.add_nodes_from(heroes['hero'].drop_duplicates(), bipartite=0)
    B.add_nodes_from(heroes['comic'].drop_duplicates(), bipartite=1)
    B.add_edges_from([(row['hero'], row['comic']) for idx, row in heroes.iterrows()])
    if weighted == True:
        Gheroes_weighted = bipartite.weighted_projected_graph(B, heroes['hero'].drop_duplicates())
    else:
        Gheroes = bipartite.projected_graph(B, heroes['hero'])
      
    
    if weighted == True:
        networks = {'silver': Gsilver_weighted, 'bronze': Gbronze_weighted, 'modern': Gmodern_weighted, 'heroes': Gheroes_weighted}
    else:
        networks = {'silver': Gsilver, 'bronze': Gbronze, 'modern': Gmodern, 'heroes': Gheroes}
        
    if outnet == True:
        graph['networks'] = networks
            
    # Add or remove nodes from graph depending on "size”
    if size == "common": 
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
            
    elif size == 10: 
        # find common heros
        comm_heroes = list(set(silver['hero']).intersection(bronze['hero']))
        comm_heroes = list(set(comm_heroes).intersection(modern['hero']))
        comm_heroes = list(set(comm_heroes).intersection(heroes['hero']))
        comm_heroes = sorted(comm_heroes)
        wanted = []
        for i in [7, 20, 22, 23, 28, 33, 39, 40, 43, 44]:
            wanted.append(comm_heroes[i])


        # a tensor containing network adjacency matrix at each time
        graph['A_tensor'] = []
        for layer in graph['layer_names']:
            network = remove_non_common(networks[layer], wanted)
            network_matrix = nx.to_numpy_array(network, sorted(list(network.nodes())))
            print(sorted(list(network.nodes())))
            graph['nodenames'] = sorted(list(network.nodes()))
            graph['N'] = len(graph['nodenames'])
            graph['A_tensor'].append(network_matrix)
            
            
            
    elif size == "full":
        full_heros = list(set().union(silver['hero'], bronze['hero'], modern['hero'], heroes['hero']))
        
        # a tensor containing network adjacency matrix at each time
        graph['A_tensor'] = []
        for layer in graph['layer_names']:
            network = networks[layer]
            network.add_nodes_from(full_heros)
            network_matrix = nx.to_numpy_array(network, sorted(list(network.nodes())))
            print(sorted(list(network.nodes())))
            graph['nodenames'] = sorted(list(network.nodes()))
            graph['N'] = len(graph['nodenames'])
            graph['A_tensor'].append(network_matrix)
    
    return graph