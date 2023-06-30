import pickle
from find_graph_general import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from brokenaxes import brokenaxes

def save_dict_values_to_text(dictionaries, filename):
    with open(filename, 'w') as file:
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                file.write(f"{key}: {value}\n")
            file.write('\n')  # Add a blank line between dictionaries

def sums_of_array_elements(arrays_dict):
    # calculate the sum of each element multiplied by the sum of its row and column indices for each array in the dictionary
    sums_dict = {}
    for key, arr in arrays_dict.items():
        nrows, ncols = arr.shape
        sums = 0
        for i in range(nrows):
            for j in range(ncols):
                sums += arr[i,j] * (i + j)
        sums_dict[key] = sums
    return sums_dict

def scale_array_elements(arrays_dict, factors_dict):
    scale_dict = {}
    # scale each element in the dictionary of arrays by its corresponding scaling factor
    for key, arr in arrays_dict.items():
        factor = 1/factors_dict[key]
        nrows, ncols = arr.shape
        arr_scaled = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                arr_scaled[i, j] = factor * arr[i, j] * (i + j)
        scale_dict[key] = arr_scaled
    return scale_dict

# concentrations = ['0.28','0.5','1','2','4','7','10','12','15','21']
concentrations = ['1']
cluster_concentration = {}
counters = {}

config, args = read_parameters('config.ini')

# directory of the trajectory and tpr
args.dir = config.get("params", "directory")
args.tpr = os.path.join(args.dir,config.get("params", "tpr_file_name"))
args.trr = os.path.join(args.dir,config.get("params", "trajectory_file_name"))
args.mdp = os.path.join(args.dir,config.get("params", "mdp_file_name"))

# calculate averaged box size
box_volume = get_box_size(args)

for c in concentrations:
    cluster_concentration[c], counters[c], _ = get_cluster_population(args, config)

# calculate 2D concentrations
normalized_cluster_concentration = scale_array_elements(cluster_concentration, sums_of_array_elements(cluster_concentration))

## plot 2D cluster distribution
plt.rcParams['font.size'] = 14
# plot three subplots for three different concentrations
# Create a figure with three subplots
fig, axs = plt.subplots(1, 1, figsize=(8,8))
gs = gridspec.GridSpec(1, 1, wspace=0.15, hspace=0.15)
# Determine the overall minimum and maximum values of the data
data_min = np.min([np.min(normalized_cluster_concentration[key]) for key in normalized_cluster_concentration])
data_max = np.max([np.max(normalized_cluster_concentration[key]) for key in normalized_cluster_concentration])
# Plot the data on each subplot
im1 = axs.imshow(normalized_cluster_concentration['1'][:5,:5].T, cmap='viridis')
# Set the titles for the subplots
axs.set_title('1 M')

axs.set_ylabel('${N_{TFSI^-}}$ (#)')
axs.set_xlabel('${N_{Li^+}}$ (#)')
axs.set_xticks(np.arange(0, 5, 1))
axs.set_yticks(np.arange(0, 5, 1))
axs.invert_yaxis()
# Set the same clim values for all subplots
im1.set_clim(data_min, data_max)
# Add a single color bar for all three subplots
cax = fig.add_axes([0.99, 0.3, 0.04, 0.4])  # [left, bottom, width, height]
cbar = fig.colorbar(im1, cax=cax)
cbar.set_label('(%)')
# Show the plot
plt.tight_layout()
plt.show()
