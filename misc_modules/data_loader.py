import torch
from torch_geometric.data import Batch, Data


def data_loader(graph1, graph2, graphm, ratio1, ratio2, ratiom, temperature):
    #Input: graph1, graph2, graphm: graphs of the molecules 1 and 2 and the membrane
    #       ratio1, ratio2, ratiom: mol% ratios of the molecules in the membrane
    #       temperature: temperature the prediction shall be made
    
    #Create Batch objects
    B1 = Batch()
    B2 = Batch()
    BM = Batch()
    
    #Create Temperature Tensor
    B_Temperature = torch.tensor([temperature])
    
    #Fill batch objects with data from graphs
    #Molecule 1
    B1.x = graph1.x
    B1.edge_index = graph1.edge_index
    B1.edge_attr = graph1.edge_attr
    B1.y = torch.Tensor([ratio1])
    B1.batch = torch.full([graph1.x.size()[0]], fill_value=0, dtype=torch.long)
    #Molecule 2
    B2.x = graph2.x
    B2.edge_index = graph2.edge_index
    B2.edge_attr = graph2.edge_attr
    B2.y = torch.Tensor([ratio2])
    B2.batch = torch.full([graph2.x.size()[0]], fill_value=0, dtype=torch.long)
    #Membrane
    BM.x = graphm.x
    BM.edge_index = graphm.edge_index
    BM.edge_attr = graphm.edge_attr
    BM.y = torch.Tensor([ratiom])
    BM.batch = torch.full([graphm.x.size()[0]], fill_value=0, dtype=torch.long)
    
    return [B1, B2, BM, B_Temperature]

def load_graph_data(path, name):
    x = torch.load(path + "/" + name + "_x.pt")
    ei = torch.load(path + "/" + name + "_ei.pt")
    ea = torch.load(path + "/" + name + "_ea.pt")

    graph = Data(x=x,edge_index=ei,edge_attr=ea)
    return graph