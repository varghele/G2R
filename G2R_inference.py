#_______________________ Explanation #________________________________________
# G2R inference script 
# predicts 2H NMR order parameters of PC membranes in
# presence of up two molecules
# Input:
# MOL1 - location of sybil mol2 file of first molecule OR None
# MOL1_ratio - molecular ratio of MOL1 in mixture (float)
# MOL2 - location of sybil mol2 file of second molecule OR "CHL", "POPE", "POPG" OR None
# MOL2_ratio - molecular ratio of MOL2 in mixture (float)
# MOLM_ratio - molecular ratio of membrane in mixture (float)
# TEMPERATURE - temperature measurement was taken at
#_____________________________________________________________________________





#_______________________ Importing Libraries #________________________________
print("IMPORTING LIBRARIES")
import os
import sys

import torch
import numpy as np

from misc_modules.prepare_mol2_funcs import *
from misc_modules.make_graph_funcs import *
from misc_modules.data_loader import *
from misc_modules.plotter import *

from model_and_weights.graph_net_B_LN import GNN_FULL_CLASS as G2R
#_____________________________________________________________________________





#________________________ Funcs for load and eval #___________________________
print("DEFINE FUNCS")
##Function to get the molecule(if uploaded) and turn it to a scaled graph
def get_molecule_and_return_scaled_graph(mol2_loc, node_mean, node_std, edge_mean, edge_std):
    #Input: mol_loc: location of mol2 file
    #Output: 
    
    #Import the raw molecule and rotate it to KOS
    raw_mol = load_data(mol2_loc)
    mol2_atom_lines = helper_for_KOS(mol2_loc)
    new_mol2_atom_lines = helper_rotate_and_transpose_to_KOS(mol2_atom_lines)
    
    #Apply the rotation to the loaded molecule
    cooked_mol = raw_mol.copy()
    for m in range(len(cooked_mol[1])):
        cooked_mol[1][m] = new_mol2_atom_lines[m][2:5]
    
    #Convert the molecule into a graph
    g = convert_to_graph(cooked_mol)
    scaled_g = scale_graph(g, node_mean, node_std, edge_mean, edge_std)
    
    return scaled_g

##Evaluate function for the model
def evaluate(model_input, model, state_dict_locs, device):
    #Set model to evaluation
    model.eval()
    
    #Establish list of yhats
    yhat_list = []
    
    with torch.no_grad():
        #Push input to Device
        for m in model_input:
            m.to(device)
        
        for state in [i for i in os.listdir(state_dict_locs) if "statedict" in i]:
            #print(state_dict_locs+"/"+state)
            #Load state dict
            model.load_state_dict(torch.load(state_dict_locs+"/"+state, map_location=torch.device(device)))
            
            #Run inference
            yhat = model(model_input).detach().cpu().numpy()
            #Important fix! Model predicts between 0 and 10 due to training
            yhat_list.append(yhat/10)
            
            #Empty CUDA cache
            torch.cuda.empty_cache()
            
    return np.asarray(yhat_list)
#_____________________________________________________________________________





#__________________________ Get inputs to script #____________________________
print("GET INPUTS")
if len(sys.argv) < 6:
    print( "\nUSAGE:", sys.argv[0], "PTH_MOL1 MOL1_RATIO PTH_MOL2 MOL2_RATIO MOLM_RATIO TEMPERATURE \n\n")
    exit(1)

MOL1 = str(sys.argv[1])
MOL1_ratio = float(sys.argv[2])
MOL2 = str(sys.argv[3])
MOL2_ratio = float(sys.argv[4])
MOLM_ratio = float(sys.argv[5])
TEMPERATURE = float(sys.argv[6])
#_____________________________________________________________________________





#__________________________ Load Scaling values #_____________________________
print("LOAD SCALING VALUES")
#Load Scaling values
node_mean = torch.from_numpy(np.loadtxt("node_and_edge_scaling_tensors/scale_node_mean.txt"))
node_std = torch.from_numpy(np.loadtxt("node_and_edge_scaling_tensors/scale_node_std.txt"))
edge_mean = torch.from_numpy(np.loadtxt("node_and_edge_scaling_tensors/scale_edge_mean.txt"))
edge_std = torch.from_numpy(np.loadtxt("node_and_edge_scaling_tensors/scale_edge_std.txt"))
#_____________________________________________________________________________





#____________________________Construct model input #__________________________
print("CONSTRUCT MODEL INPUT")
##Load MOL 1
if MOL1=="none" or MOL1_ratio==0:
    G1 = load_graph_data("latent_graphs_scaled","0")
else:
    G1 = get_molecule_and_return_scaled_graph(MOL1, node_mean, node_std, edge_mean, edge_std)

##Load MOL 2
if MOL2=="none" or MOL2_ratio==0:
    G2 = load_graph_data("latent_graphs_scaled","0")
elif MOL2=="CHL":
    G2 = load_graph_data("latent_graphs_scaled","CHL")
elif MOL2=="POPE":
    G2 = load_graph_data("latent_graphs_scaled","POPE")
elif MOL2=="POPG":
    G2 = load_graph_data("latent_graphs_scaled","POPG")
else:
    G2 = get_molecule_and_return_scaled_graph(MOL2, node_mean, node_std, edge_mean, edge_std)

##Load MEMBRANE
GM = load_graph_data("latent_graphs_scaled","POPC")

##BATCH IT
INPUT = data_loader(G1, G2, GM, MOL1_ratio, MOL2_ratio, MOLM_ratio, TEMPERATURE)
#_____________________________________________________________________________





#______________________________ Set up model #________________________________
print("SET UP MODEL")
##Get device to run this on
#gpu = 0
#device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
#if torch.cuda.is_available():
#    torch.cuda.set_device(device)
#print(f"DEVICE: {device}")
device="cpu"

##SET UP PARAMETERS
BATCH_SIZE = 1
NO_MP_ONE = 2
NO_MP_TWO = 2

##DEFINE MODEL
model = G2R(BATCH_SIZE, NO_MP_ONE, NO_MP_TWO, device)
model.to(device)
#_____________________________________________________________________________





#______________________________ Run the model #________________________________
print("RUN MODEL")
#Run evaluation
yhats = evaluate(INPUT, model, "model_and_weights", device)

#Define path where results get saved
results_pth = "results/"

yhats = np.squeeze(yhats)
ymean = np.mean(np.squeeze(yhats), axis=0)
pc_data = np.loadtxt("model_and_weights/POPC_order_2H_T303K.dat")

#Save results
np.savetxt(results_pth+"yhats.txt", yhats.T, delimiter=",", header="G2R", comments="#Individual Predictions of the 8 distinct states of the model:")
#Save mean prediction
np.savetxt(results_pth+"prediction.txt", ymean.T, delimiter=",", header = "G2R",comments="#Mean prediction of experts council:")
#Plot results
plotter(pc_data, yhats, ymean, results_pth)
#_____________________________________________________________________________