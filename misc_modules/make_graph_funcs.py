import pandas as pd
import numpy as np
import mendeleev as me
import torch
from torch_geometric.data import Data, Batch
from itertools import combinations

def convert_to_graph(raw_mol):
    #convert_to_graph_data converts the raw data of the list into graph data
    #-Node - Features:
    ##[see below]
    #-Edge - Features:
    ##[distance, bond - type(one - hot)]
    NO_NODE_FEATURES=0
    NO_EDGE_FEATURES=0
    graph_data = []
    bond_dic = {"1": [1, 0, 0, 0, 0, 0], "2": [0, 1, 0, 0, 0, 0], "3": [0, 0, 1, 0, 0, 0], 
                "ar": [0, 0, 0, 1, 0, 0], "am": [0, 0, 0, 0, 1, 0], "n": [0, 0, 0, 0, 0, 1]}

    node_features = []
    edge_index = []
    edge_attributes = []
    
    # Construct node features
    for at in range(len(raw_mol[0])):
        temp_nf = []  # temporary node features
    
        temp_nf.append(float(raw_mol[2][at]))  # charge
    
        temp_atom_id=raw_mol[0][at] # Get the atom name/id like "C"
        temp_me_atom=me.element((temp_atom_id)) # Get the mendeleev object that holds all features
    
        temp_nf.append(temp_me_atom.atomic_number) # atomic number
        temp_nf.append(temp_me_atom.atomic_radius) # atomic radius
        temp_nf.append(temp_me_atom.atomic_volume) # atomic volume
        temp_nf.append(temp_me_atom.atomic_weight) # atomic weight
    
        temp_nf.append(temp_me_atom.covalent_radius) # covalent radius
        temp_nf.append(temp_me_atom.dipole_polarizability) # dipole polarizability
        temp_nf.append(temp_me_atom.electron_affinity) # electron affinity
        temp_nf.append(temp_me_atom.electrons) # number of electrons
    
        temp_nf.append(temp_me_atom.electrophilicity()) #electrophillicity index
        temp_nf.append(temp_me_atom.en_pauling) # electronegativity acc. to pauling
        temp_nf.append(temp_me_atom.neutrons) # number of neutrons
        temp_nf.append(temp_me_atom.protons) # number of protons
    
        temp_nf.append(temp_me_atom.vdw_radius) # van der waals radius
    
        temp_nf.append(raw_mol[1][at][0])  # x pos
        temp_nf.append(raw_mol[1][at][1])  # y pos
        temp_nf.append(raw_mol[1][at][2])  # z pos
    
        node_features.append(temp_nf)
    
    
    ### Construct edge indices and features(attributes)
    #Construct all possible edges
    edge_combs=[list(j) for j in combinations(range(len(raw_mol[0])),2)]
    #Get bond indice list
    bond_indices=[[int(j)-1 for j in k[:2]] for k in raw_mol[3]]
    #Iterate over every edge
    for edge in edge_combs:
        #print(edge)
        #print([int(j)-1 for j in raw_mol[3][0][:-1]])
        #Lists of temporary edge indices and features
        temp_ei = []  # temporary edge index
        temp_ef = []  # temporary edge features
        
        #Is the graph edge a bond between atoms
        if edge in bond_indices:
            #Find out TYPE of bond
            bond_type=bond_dic[raw_mol[3][bond_indices.index(edge)][-1]]
        #Or isn't there a bond
        if edge not in bond_indices:
            #Type is "none":"n"
            bond_type=bond_dic["n"]
            
        #Appending edge index twice and reversed for bidirectional edge
        temp_ei.append(edge)
        temp_ei.append(edge[::-1])
        
        #Get first bond position
        r1 = np.array(raw_mol[1][edge[0]])
        #Get second bond position
        r2 = np.array(raw_mol[1][edge[1]])
        
        #Appending edge features twice because of bidirectional edges
        temp_ef.append([np.linalg.norm(r1 - r2)] + bond_type)
        temp_ef.append([np.linalg.norm(r1 - r2)] + bond_type)
        
        #Appending to graph
        for ei in temp_ei:
            edge_index.append(ei)
        for ef in temp_ef:
            edge_attributes.append(ef)
    
    #Tensorize the data
    node_features=torch.tensor(node_features,dtype=torch.float)
    edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
    edge_attributes=torch.tensor(edge_attributes,dtype=torch.float)
    
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes)
    
    
    return graph


def scale_graph(latent_graph, node_mean, node_std, edge_mean, edge_std):
    #Apply zero-mean, unit variance scaling
    x_sc = latent_graph.x-node_mean
    x_sc /= node_std
    x_sc = x_sc.float()
    ea_sc = latent_graph.edge_attr-edge_mean
    ea_sc /= edge_std
    ea_sc = ea_sc.float()
    scaled_graph = Data(x=x_sc,edge_index=latent_graph.edge_index,edge_attr=ea_sc)
    
    return scaled_graph