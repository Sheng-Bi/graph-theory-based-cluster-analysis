import networkx as nx
import matplotlib.pyplot as plt
from cgi import print_form
import pandas as pd
import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from MDAnalysis.coordinates.XTC import XTCReader
from MDAnalysis.analysis import lineardensity as lin
import MDAnalysis.analysis.msd as msd
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)
import nglview as nv
import matplotlib.pyplot as plt
import argparse
from scipy.io import FortranFile
import sys
import math
import argparse
import configparser
from scipy.signal import find_peaks
from MDAnalysis.lib.nsgrid import FastNS
from collections import Counter
import matplotlib.animation as animation
from MDAnalysis.analysis.dihedrals import Dihedral
from itertools import groupby
from operator import itemgetter
from graphTheoryCluster import *
import gromacs
import fnmatch
import os
import glob
from scipy.stats import linregress

def read_parameters(filename='config.ini'):
    ## main function
    # input parameters
    my_arg = argparse.ArgumentParser('My argument parser')
    args = my_arg.parse_args('')
    config = configparser.ConfigParser()
    config.read(filename)

    # directory of the trajectory and tpr
    args.dir = config.get("params", "directory")
    args.tpr = os.path.join(args.dir,config.get("params", "tpr_file_name"))
    args.trr = os.path.join(args.dir,config.get("params", "trajectory_file_name"))
    args.mdp = os.path.join(args.dir,config.get("params", "mdp_file_name"))

    # get temperature from mdp file
    mdp = gromacs.fileformats.mdp.MDP(args.mdp, autoconvert=False)
    args.temperature = float(mdp.get('ref_t'))
    args.dt = float(mdp.get('dt'))

    # parameters regarding cluster size
    args.cutoff = config.getfloat("params", "cutoff") #4.5 
    args.cluster_limit = config.getint("params", "cluster_limit") #50
    args.small_cluster = config.getint("params", "small_cluster") #10
    args.threshold = config.getfloat("params", "mode_threshold") 
    # parameters regarding the system
    args.cation_name = config.get("params", "cation_name")
    args.anion_name = config.get("params", "anion_name")
    args.solvent_name = config.get("params", "solvent_name")

    args.cationIsCa2 = config.getboolean("params", "cationIsCa2")
    args.anionIsTFSI = config.getboolean("params", "anionIsTFSI")
    args.ionicLiquid = config.getboolean("params", "ionicLiquid")
    if args.anionIsTFSI:
        args.cutoff = 6 # pre cut-off for tf2n ions, this value should not affect the results if it is large enough
        args.rcutoff_cation_O_tfsi = 3 # get from rdf of cation and O in tfsi

    args.selection = 'resname '+args.cation_name+' '+args.anion_name+' '+args.solvent_name
    args.update = True

    # parameters regarding the output
    args.plotOrNot = config.getboolean("params", "plotOrNot")
    args.write_small_cluster = config.getboolean("params", "write_small_cluster")
    if args.write_small_cluster:
        args.small_cluster_ndx_filename = 'small_cluster.ndx'
        
    return config, args

def get_box_size(args):
    # calculate averaged box size
    if bool(glob.glob(os.path.join(args.dir,'mdrun_*'))):
        box_volume = []
        for path in glob.glob(os.path.join(args.dir,'mdrun_*')):
            u = mda.Universe(args.tpr, os.path.join(path,'vacf.gro'))
            box_volume.append(np.linalg.det(u.trajectory.ts.triclinic_dimensions))
        box_volume = np.array(box_volume)  
        box_volume = box_volume.mean()*1e-30 # A^3 to m^3
    else:
        u = mda.Universe(args.tpr, args.trr) # load the trajectory
        box_volume = []    
        for ts in u.trajectory[-100:]:
            print('\r Time = {:.3f}'.
                format(ts.time),end='')
            box_volume.append(np.linalg.det(ts.triclinic_dimensions))
        box_volume = np.array(box_volume)    
        box_volume = box_volume.mean()*1e-30 # A^3 to m^3
    return box_volume

def get_cluster_population(args, config):
    # calculate cluster population
    if config.getboolean("params", "fine_cluster") and bool(glob.glob(os.path.join(args.dir,'mdrun_*'))):
        args.begin = 20
        args.end =120
        args.skip = 2500
        cluster_population={}
        counter = {}
        for idx, path in enumerate(glob.glob(os.path.join(args.dir,'mdrun_*'))):
            args.trr = os.path.join(path,'traj.trr')
            cluster_population[idx], counter[idx], ns_frames = graph_theory_based_clustering(args)
        stacked_arrays = np.stack(list(cluster_population.values()), axis=0)
        cluster_population = np.mean(stacked_arrays, axis=0)
    else:
        args.begin = config.getfloat("params", "begin_time") # ps
        args.end = config.getfloat("params", "end_time") # ps
        args.skip = config.getint("params", "skip")
        cluster_population, counter, ns_frames = graph_theory_based_clustering(args)
    return cluster_population, counter, ns_frames

def get_diffusion_for_cation_anion(args, config):
    if config.get("params", "diffusion_method") == 'MSD':
        # get diffusion from MSD calculated by MDAnalysis
        if len(glob.glob(os.path.join(args.dir,'msd_mdrun_*'))):
            diff_cation, diff_anion = getDiffusionFromMDAnalysisMSD(args, 'msd_mdrun_*', 200, 5000, plotOrNot=True) 
        else:
            # get diffusion from MSD calculated by Gromacs
            filename_msd = os.path.join(args.dir,config.get("params", "diffusion_msd_file_name")+'.xvg')
            diff_cation, diff_anion = getDiffusionFromMsd(filename_msd, plotOrNot=True)
    elif config.get("params", "diffusion_method") == 'VACF':
        filename_prefix_cation = os.path.join(args.dir,config.get("params", "diffusion_cation_file_name")+'[0-9]*')
        filename_prefix_anion = os.path.join(args.dir,config.get("params", "diffusion_anion_file_name")+'[0-9]*')
        if bool(glob.glob(filename_prefix_cation)):
            vacf_cation, diff_cation = getDiffusionFromVacf(filename_prefix_cation, plotOrNot=True,fitting_time=config.getfloat("params", "fitting_time_cluster"))
            vacf_anion, diff_anion = getDiffusionFromVacf(filename_prefix_anion, plotOrNot=True,fitting_time=config.getfloat("params", "fitting_time_cluster"))
        else:
            diff_cation, diff_anion = 0, 0
    
    return diff_cation, diff_anion    

def get_diffusion_for_each_cluster(args, config, diff_cation, diff_anion):
    
    if config.get("params", "diffusion_for_cluster_method") == 'VACF':
        diff_cluster = np.zeros(args.cluster_limit)
        vacf_ave={}
        for i in range(args.cluster_limit):
            filename_prefix = os.path.join(args.dir,'vacf_small_'+str(i)+'_[0-9]*')
            if len(glob.glob(filename_prefix))>0.3*len(glob.glob(os.path.join(args.dir,'mdrun_*'))):
                weights = [1]*len(glob.glob(filename_prefix))
                for ii, filename in enumerate(glob.glob(filename_prefix)):
                    index = int(filename.rsplit('_', 1)[-1].split('.xvg')[0])
                    index_file = os.path.join(args.dir,'small_cluster_'+str(index)+'.ndx')
                    with open(index_file, 'r') as f:
                        # Initialize the flag
                        found_mode_line = False
                        indices = []
                        # Read the file line by line
                        for line in f:
                            # Check if the line is the mode line
                            if line.strip() == f'[{i}]':
                                found_mode_line = True
                            elif found_mode_line:
                                # Concatenate the indices from multiple lines
                                indices.extend(line.strip().split())
                            if not line.strip() or line.strip().startswith(';'):
                                # End of the mode block
                                if indices:
                                    # Count the number of entries
                                    num_atoms = len(indices)
                                    weights[ii] = num_atoms
                                    break
                print(i)
                vacf_ave[i], diff_cluster[i] = getWeightedDiffusionFromVacf(filename_prefix, plotOrNot=True, weight=weights, \
                                                                fitting_time=config.getfloat("params", "fitting_time_cluster"))
    elif config.get("params", "diffusion_for_cluster_method") == 'MSD':
        if len(glob.glob(os.path.join(args.dir,'msd_mdrun_*'))):
            diff_cluster = np.zeros(args.cluster_limit)
            # calculate diffusion for free ions based on segmented trajectory
            args.begin = 2000 # ps
            args.end = 20000 # ps
            args.skip = 100
            diff_free, msd_free, msd_time, free_life_distributions = \
                getDiffusionForFreeIonsBasedOnSegmentedMSD(args, 'msd_mdrun_*', expected_lifetime = 1000, skip = 100, \
                                                        beginFit = 200, endFit = 2000, plotOrNot=True)
            diff_cluster[1] = diff_free
        else:
            diff_cluster = np.zeros(args.cluster_limit)
            diff_cluster[1] = (diff_cation+diff_anion)/2
    
    return diff_cluster

def get_diffusion_for_small_large_cluster(args, config):
    filename_prefix_small = os.path.join(args.dir,config.get("params", "diffusion_small_file_name")+'[0-9]*')
    filename_prefix_large = os.path.join(args.dir,config.get("params", "diffusion_large_file_name")+'[0-9]*')
    if bool(glob.glob(filename_prefix_small)):
        diff_small = getDiffusionFromVacf(filename_prefix_small, plotOrNot=True,fitting_time=config.getfloat("params", "fitting_time_binary"))
        diff_large = getDiffusionFromVacf(filename_prefix_large, plotOrNot=True,fitting_time=config.getfloat("params", "fitting_time_binary"))
    else:
        diff_small, diff_large = 0, 0
    return diff_small, diff_large

def get_conductivities(args, config, cluster_population, box_volume, diff_cation, diff_anion, diff_cluster):
    
    # Boltzmann constant
    kb = 1.38064852e-23 # J/K
    # elementary charge
    e = 1.60217662e-19 # C
    # Avogadro's number
    Na = 6.022140857e23 # mol^-1
    # square of the scale of the charge
    e2_scale = 1 #1^2 for point charge, 0.78^2 for coarsed-grained charge
    
    u = mda.Universe(args.tpr, args.trr)
    n_cation = len(u.select_atoms('resname '+args.cation_name).residues)
    n_anion = len(u.select_atoms('resname '+args.anion_name).residues)
    charge_cation = u.select_atoms('resname '+args.cation_name).total_charge()/n_cation
    charge_anion = u.select_atoms('resname '+args.anion_name).total_charge()/n_anion

    # sigma_cne_1 is calculated based on the assumption that the diffusion of each cluster is based on the averaged diffusion of cation and anion
    # sigma_cne_2 is calculated based on the assumption that the diffusion of each cluster is directly calcualted by vacf
    sigma_cne_1 = 0
    sigma_cne_2 = 0

    sigma_contributions_1 = np.zeros((args.cluster_limit+1,args.cluster_limit+1))
    sigma_contributions_2 = np.zeros((args.cluster_limit+1,args.cluster_limit+1))

    diff_cluster_1 = np.zeros((args.cluster_limit,args.cluster_limit))
    diff_cluster_2 = np.zeros((args.cluster_limit,args.cluster_limit))

    for i in range(args.cluster_limit):
        for j in range(args.cluster_limit):
            if i+j > 0:
                diff_cluster_1[i][j] = (diff_cation*i+diff_anion*j)/(i+j)   
                
    for i in range(args.cluster_limit):
        for j in range(args.cluster_limit):
            if i+j > 0 and i+j < args.cluster_limit:
                if diff_cluster[i+j] > 0:
                    diff_cluster_2[i][j] = diff_cluster[i+j]        
                    
    sigma_contributions_1, sigma_cne_1 = calculate_sigma_cne(cluster_population, diff_cluster_1, args, u, box_volume)    
    sigma_contributions_2, sigma_cne_2 = calculate_sigma_cne(cluster_population, diff_cluster_2, args, u, box_volume)    

    # calculate conductivity by NE
    sigma_ne = (diff_cation*n_cation*charge_cation*charge_cation
            +diff_anion*n_anion*charge_anion*charge_anion)*e*e/(kb*args.temperature*box_volume)

    # calculate conductivity by ECACF
    if config.get("params", "cond_method") == 'ECACF':
        gro_dir_prefix = os.path.join(args.dir,config.get("params", "caf_gro_file_name"))
        xvg_dir_prefix = os.path.join(args.dir,'caf')
        if len(fnmatch.filter(os.listdir(args.dir), 'caf*.xvg')):
            ncases = len(fnmatch.filter(os.listdir(args.dir), 'caf[0-9]*.xvg'))
            sigma_ecacf = calCondFromECACF(ncases, gro_dir_prefix, xvg_dir_prefix, args.temperature)
    elif config.get("params", "cond_method") == 'Einstein':
        if len(glob.glob(os.path.join(args.dir,'msd_mdrun_*'))):
            args.trr = os.path.join(args.dir,config.get("params", "collective_trr_file_name"))
            _, sigma_ecacf = cal_cond_einstein(args, 'msd_mdrun_*', 0.005, 0.12,\
                            10000, 200000, args.temperature, plotOrNot=True)
        else:
            gro_dir_prefix = os.path.join(args.dir,config.get("params", "caf_gro_file_name"))
            xvg_dir_prefix = os.path.join(args.dir,'caf')
            if len(fnmatch.filter(os.listdir(args.dir), 'caf[0-9]*.xvg')):
                ncases = len(fnmatch.filter(os.listdir(args.dir), 'caf[0-9]*.xvg'))
                sigma_ecacf = calCondFromECACF(ncases, gro_dir_prefix, xvg_dir_prefix, args.temperature)
                

    print(sigma_ne, sigma_cne_1, sigma_cne_2, sigma_ecacf)
    return sigma_ne, sigma_cne_1, sigma_cne_2, sigma_ecacf

def get_spectrum_for_free_and_bound_ions(args, config, total_time = 20,dt=0.002):
    filename_prefix_anion = os.path.join(args.dir,config.get("params", "diffusion_anion_file_name")+'[0-9]*')
    if bool(glob.glob(filename_prefix_anion)):
        vacf_anion, diff_anion = getDiffusionFromVacf(filename_prefix_anion, plotOrNot=True,fitting_time=-1)

    vacf_ave={}
    i = 1
    filename_prefix = os.path.join(args.dir,'vacf_free_[0-9]*')
    if len(glob.glob(filename_prefix))>0.3*len(glob.glob(os.path.join(args.dir,'mdrun_*'))):
        weights = [1]*len(glob.glob(filename_prefix))
        for ii, filename in enumerate(glob.glob(filename_prefix)):
            index = int(filename.rsplit('_', 1)[-1].split('.xvg')[0])
            index_file = os.path.join(args.dir,'small_cluster_'+str(index)+'.ndx')
            with open(index_file, 'r') as f:
                # Initialize the flag
                found_mode_line = False
                indices = []
                # Read the file line by line
                for line in f:
                    # Check if the line is the mode line
                    if line.strip() == f'[{i}]':
                        found_mode_line = True
                    elif found_mode_line:
                        # Concatenate the indices from multiple lines
                        indices.extend(line.strip().split())
                    if not line.strip() or line.strip().startswith(';'):
                        # End of the mode block
                        if indices:
                            # Count the number of entries
                            num_atoms = len(indices)
                            weights[ii] = num_atoms
                            break
        print(i)
        print(weights)
        vacf_ave[i], _ = getWeightedDiffusionFromVacf(filename_prefix, plotOrNot=True, weight=weights, \
                                                               fitting_time=-1)
        vacf_free = vacf_ave[1]
    else:
        vacf_free = vacf_anion
    
    filename_prefix = os.path.join(args.dir,'vacf_bound_[0-9]*')
    if len(glob.glob(filename_prefix)):
        weights = [1]*len(glob.glob(filename_prefix))
        for ii, filename in enumerate(glob.glob(filename_prefix)):
            index_file = os.path.join(args.dir,'index_'+str(ii)+'.ndx')
            with open(index_file, 'r') as f:
                # Initialize the flag
                found_mode_line = False
                indices = []
                # Read the file line by line
                for line in f:
                    # Check if the line is the mode line
                    if line.strip() == '[ Li_TFS_&_!1 ]':
                        found_mode_line = True
                    elif found_mode_line:
                        # Concatenate the indices from multiple lines
                        indices.extend(line.strip().split())
                    if not line.strip() or line.strip().startswith(';'):
                        # End of the mode block
                        if indices:
                            # Count the number of entries
                            num_atoms = len(indices)
                            weights[ii] = num_atoms
                            break
        vacf_bound, _ = getWeightedDiffusionFromVacf(filename_prefix, plotOrNot=False, weight=weights, \
                                                        fitting_time=-1)
    else:
        vacf_bound = np.zeros(len(vacf_free))
        
    
    total_time = total_time * 1e-12  # ps to s
    dt = dt * 1e-12  # ps to s
    time = np.linspace(0, total_time, int(total_time/dt))
    acf = vacf_free[:len(time)]
    a_free,b_free = fft_acf(dt,total_time,acf)
    acf = vacf_bound[:len(time)]
    a_bound,b_bound = fft_acf(dt,total_time,acf)
    acf = vacf_anion[:len(time)]
    a_anion,b_anion = fft_acf(dt,total_time,acf)
        
    return vacf_free, vacf_bound, vacf_anion, a_free, b_free, a_bound, b_bound, a_anion, b_anion

def main():
    ## main function
    # input parameters
    config, args = read_parameters('config.ini')

    # calculate averaged box size
    box_volume = get_box_size(args)

    # calculate cluster population
    cluster_population, counter, ns_frames = get_cluster_population(args, config)
    
    # calculate diffusion for cation and anion
    diff_cation, diff_anion = get_diffusion_for_cation_anion(args, config)
        
    # calculate diffution for each cluster
    diff_cluster = get_diffusion_for_each_cluster(args, config, diff_cation, diff_anion)
    
    # calculate conductivity by cluster NE
    sigma_ne, sigma_cne_1, sigma_cne_2, sigma_ecacf = get_conductivities(args, config, cluster_population, \
                                                            diff_cluster, diff_cation, diff_anion, box_volume)
    
    return cluster_population, diff_cation, diff_anion, diff_cluster, sigma_ne, sigma_cne_1, sigma_cne_2, sigma_ecacf

# if __name__ == "__main__":
#     main()


# aa = np.zeros(100)
# for i in range(50):
#     for j in range(50):
#         if 2*i-j != 0:
#          aa[i+j] += cluster_population[i][j]*(i+j)
# plt.plot(aa,'o-')
# plt.xlim([0,21])
# plt.show()

# pwd1 = '/scratchbeta/bisheng/GraphTheory/LiTFSI/'
# pwd2 =  '/scratchbeta/bisheng/CaCH4_2/CaTFSI_2/'
# pwd3 = '/scratchbeta/bisheng/GraphTheory/IL/'
# LiTFSI = ['0.28','0.5','1','2','4','7','10','12','15','21']
# CaTFSI = ['0.1','0.2','0.4','0.6','0.8','1.0']
# IL = ['300K','350K','400K','450K','500K']

# dir1 = pwd1 + LiTFSI[-6] + '/'
# dir2 = pwd2 + CaTFSI[3] + '/'
# dir3 = pwd3 + IL[4] + '/'

# total_time = 20 * 1e-12  # 1 ps
# dt = 0.002 * 1e-12  # 0.002 ps

# time = np.linspace(0, total_time, int(total_time/dt))
# acf = vacf_ave[1][:len(time)]
# a_21_free,b_21_free = fft_acf(dt,total_time,acf)

# plt.plot(a_4, b_4,'r--')

# plt.plot(a_15_free, b_15_free,'b--')
# plt.plot(a_21_free, b_21_free,'g--')

# plt.plot(a_15,b_15,'k-.')

# plt.xlim([860,880])
# plt.ylim([0.1e-8,0.4e-8])
# plt.show()