import networkx as nx
import matplotlib.pyplot as plt
from ast import arg
from cgi import print_form
import pandas as pd
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.formats.libmdaxdr import XTCFile
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
# from fft_function import*
from scipy.signal import find_peaks
from MDAnalysis.lib.nsgrid import FastNS
from collections import Counter
import matplotlib.animation as animation
from MDAnalysis.analysis.dihedrals import Dihedral
from itertools import groupby
from operator import itemgetter
from scipy import integrate
from scipy.ndimage.filters import uniform_filter1d
import glob
import re
from scipy.stats import linregress
import os
# %matplotlib inline

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angel_in_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    return angel_in_rad

def getUniqueItems(iterable):
    seen = set()
    result = []
    for item in iterable:
        curr = frozenset(item)
        if curr not in seen:
            seen.add(curr)
            result.append(item)
    return result

def plot_ns_result(res_count):
    name = ['-'.join(value) for value, count in res_count]
    name = name[::-1]
    count = [count for value, count in res_count]
    count = count[::-1]
    total = sum(count)
    count = [a/total*100 for a in count]

    ax.clear()
    ax.barh(name,count)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.xaxis.set_ticks_position('top')
    #添加标注
    for i, (value, name) in enumerate(zip(count,name)):
        #enumerate枚举对象，一个索引序列，同时列出数据和数据下标
        ax.text(value,i,name,size=10,weight=600, color='black',ha='right', va='bottom')  #国家名称
        ax.text(value, i-0.5,'{:.2f}%'.format(value),size=12,color='red', ha='right', va='baseline')   # GDP值/10^13

def nextpow2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def fft_acf(dt, total_time, acf):
    fs = 1 / dt
    L = int(total_time / dt)
    time = np.linspace(0, total_time, L)
    y = acf[0:L]
    n = range(0, L)
    t = [x / fs for x in n]
    b = 2 ^ nextpow2(L)
    B = np.fft.fft(y, b)
    frequency = [x * fs / b for x in n]
    Frequency = frequency[0:int(b / 2)]
    wavenumber = [x / 1e12 * 33.356 for x in Frequency]
    real_part = B.real * dt
    real_part_slice = real_part[0:int(b / 2)]
    # plt.figure()
    # plt.plot(time, acf[:L] / acf[0])
    # # plt.set_xlim([0, 5e-13])
    # plt.figure()
    # plt.plot([x / 1e12 for x in Frequency], real_part_slice)  # THz
    # plt.set_xlim([0, 30])
    return wavenumber, real_part_slice

def cal_neighbour_search(u, atomgroup,  rcutoff):

    box = u.trajectory.ts.dimensions

    # if atomgroup.ts.frame%10==0:
    print('\r Time = {:.3f}'.
    format(u.atoms.ts.frame*u.atoms.ts.dt),end='')

    res_com_all = []
    for i_res in u.residues:
        if i_res.resname == 'tf2':
            # for tf2n, use the position of its N atom rather than its center-of-mass position
            res_com_all.append(i_res.atoms.select_atoms('name N').atoms.center_of_mass(unwrap=True))
        else:
            # for others, use its center-of-mass position
            res_com_all.append(i_res.atoms.center_of_mass(unwrap=True))
    res_com_all = np.array(res_com_all, dtype = np.float32)

    res_com_query = []
    for i_res in atomgroup.residues:
        if i_res.resname == 'tf2':
            res_com_query.append(i_res.atoms.select_atoms('name N').atoms.center_of_mass(unwrap=True))
        else:
            res_com_query.append(i_res.atoms.center_of_mass(unwrap=True).astype('float32'))
    res_com_query = np.asarray(res_com_query, dtype = np.float32)

    gridsearch = FastNS(rcutoff, res_com_all, box, pbc=True)
    ns = gridsearch.search(res_com_query)
    pairs = np.asarray(ns.get_pairs())
    distance = ns.get_pair_distances().reshape(-1,1)
    charge = u.residues[pairs[:,1]].charges.reshape(-1,1)
    resname = u.residues[pairs[:,1]].resnames.reshape(-1,1)
    query_resnames = atomgroup.residues[pairs[:,0]].resnames.reshape(-1,1)
    query_id = atomgroup.residues[pairs[:,0]].resids.reshape(-1,1)
    pair_id =  u.residues[pairs[:,1]].resids.reshape(-1,1)
    pairs = np.concatenate((query_id, pair_id, query_resnames, resname, charge, distance), axis = 1)
    df = pd.DataFrame(data=pairs, columns=["query", "pairs", "query_resnames", "resnames", "charges", "distances"])

    # if only accounting charged items, use: ns_result = df.loc[abs(df["charges"]) >= 1E-6]
    ns_result = df

    return ns_result

def get_neighbour_search(args):

    u = mda.Universe(args.tpr, args.trr)

    dt = u.trajectory.dt
    start_frame = int(args.begin/dt)

    select_group=u.select_atoms(args.selection,updating=args.update)
    resname = getUniqueItems(select_group.residues.resnames)[0]
    print('Search neighbors around '+args.selection)

    if args.end < 0 or args.end > u.trajectory.totaltime:
        args.end = u.trajectory.totaltime
        stop_frame = u.trajectory.n_frames
    else:
        stop_frame = int(np.ceil(args.end/dt))

    step = args.skip

    n_frame = len(range(start_frame,stop_frame,step))

    ns = AnalysisFromFunction(cal_neighbour_search, u.trajectory,
            u, select_group, args.cutoff)

    ns.run(start_frame, stop_frame, step)

    ns_timeseries =  ns.results.timeseries
    ns_tiems = ns.times
    ns_frames = ns.frames

    return ns_timeseries, ns_tiems, ns_frames, n_frame

def get_O_Ca_distance(ind, j, u, frame):
    u.trajectory[frame]
    ## note that here the index- ind and j start from 1 (as they come from MDanalysis), so we need to minus them by 1
    O_dis = u.residues[ind-1].atoms.select_atoms('name O*').atoms.positions
    Ca_dis = u.residues[j-1].atoms.positions

    dist_arr = distances.distance_array(Ca_dis, # reference
                                    O_dis, # configuration
                                    box=u.trajectory[frame].dimensions)
    return dist_arr

def subgraph(pointList, linkList, label_name):
    G = nx.Graph()
    # 转化为图结构
    for node in pointList:
        G.add_node(node)

    for link in linkList:
        G.add_edge(link[0], link[1], minlen = 1)

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    # # 得到不连通的子集
    # nodeSet = G.subgraph(c).nodes()
    # # 绘制子图
    # subgraph = G.subgraph(c)
    # plt.subplot(subplot[0])  # 第二整行
    # nx.draw_networkx(subgraph, with_labels=True)
    return S

def plotsubgraph(s, label_name):
    keys = list(s.nodes)
    dct = {key: label_name[key] for key in keys}
    plt.figure()
    pos=nx.drawing.nx_agraph.graphviz_layout(s, prog='dot')
    nx.draw(s,pos, labels=dct, with_labels=True)
    plt.show()

# check if two arrays have at least one same elements
def check_same_element(arr1, arr2): 
    for i in arr1:
        for j in arr2:
            if i == j:
                return True
    return False

def get_mode_indices_with_threshold(atom_data, threshold):
    """
    Given a 2D NumPy array where each row represents one atom and 
    columns represent their data at different time steps, returns 
    an array containing the mode value for each row, the occurrence 
    rate of the mode value for each row, and an array of indices for 
    each mode value that has an occurrence rate above the threshold.
    
    Parameters:
    -----------
    atom_data : numpy.ndarray
        A 2D numpy array containing the atom data. Each row represents
        one atom and each column represents the data at a different
        time step. The possible values are integers ranging from 1 to 6.
    threshold : float
        A float value between 0 and 1 indicating the minimum occurrence
        rate required for a mode value to be considered valid.
    
    Returns:
    --------
    numpy.ndarray, numpy.ndarray, list of numpy.ndarray
        A tuple of three objects. The first object is a 1D numpy array 
        containing the mode value for each row. The second object is a 
        1D numpy array containing the occurrence rate of the mode value 
        for each row. The third object is a list of NumPy arrays, where 
        each array contains the indices of the rows that have the same 
        mode value and whose occurrence rate is above the threshold.
    """
    
    # create empty arrays to store the mode value and occurrence rate for each row
    mode_values = np.zeros(atom_data.shape[0])
    mode_rates = np.zeros(atom_data.shape[0])

    # iterate over each row of the atom data array
    for i, row in enumerate(atom_data):
        # use numpy bincount function to count occurrence of each value in the row
        counts = np.bincount(row)

        # find the value with the highest count
        mode_value = np.argmax(counts)

        # calculate the occurrence rate of the mode value
        mode_rate = counts[mode_value] / len(row)

        # store the mode value and occurrence rate for the row
        mode_values[i] = mode_value
        mode_rates[i] = mode_rate

    # group the indices of rows by their mode value and filter by threshold
    mode_indices = [[] for _ in range(int(max(mode_values)) + 1)]
    for i, mode_value in enumerate(mode_values):
        if mode_rates[i] > threshold:
            mode_indices[int(mode_value)].append(i)

    return mode_values, mode_rates, mode_indices

def getdfmin(df, ion1_name, ion2_name):

        df_all = df.loc[(df["query_resnames"]==ion1_name)]
        df_all = df_all.astype({"distances": float})
        resids = df_all['query'].unique()
       
        min_row_index = []
        for ids in resids:
            df = df_all[df_all['query']==ids] 
            filter = (df['resnames']==ion2_name)
            df = df[filter]
            
            if len(df)>0:
                # record the most minimum distance of a ion1 and nearby ion2
                min_row_index.extend(df.distances.nsmallest(1).index.to_list())
            
        return df_all.loc[min_row_index]   
                     
def getdfminTFSI_Cation(df, ion1_name, ion2_name, args, u, frame):

        df_all = df.loc[(df["query_resnames"]==ion1_name)]
        df_all = df_all.astype({"distances": float})
        resids = df_all['query'].unique()
       
        min_row_index = []
        for ids in resids:
            df = df_all[df_all['query']==ids] 
            filter = (df['resnames']==ion2_name)
            df = df[filter]
            
            if len(df)>0:
                for index, row in df.iterrows():
                    cation_query = row['pairs']
                    anion_query = row['query']
                    dist_arr = get_O_Ca_distance(anion_query, cation_query, u, frame)
                    if dist_arr.min() >= args.rcutoff_cation_O_tfsi:
                        df = df.drop(index) 
                    else:
                        df.loc[index, 'cation_O_distance'] = dist_arr.min()
                # record the most two minimum distances of a ion1 and nearby ion2
                if len(df)>0:
                    min_row_index.extend(df.cation_O_distance.nsmallest(1).index.to_list())
                # print(df_all.loc[df.cation_O_distance.nsmallest(2).index.to_list()])
        return df_all.loc[min_row_index]                

def getdfminTFSI_Cation_simple(df, ion1_name, ion2_name, args, u, frame):

        df_all = df.loc[(df["query_resnames"]==ion1_name)]
        df_all = df_all.astype({"distances": float})
        resids = df_all['query'].unique()
       
        iso_node = []
        for ids in resids:
            df = df_all[df_all['query']==ids] 
            filter = (df['resnames']==ion2_name)
            df = df[filter]
            
            if len(df) == 0:
                iso_node.append(ids)
            elif len(df)>0:
                for index, row in df.iterrows():
                    cation_query = row['pairs']
                    anion_query = row['query']
                    dist_arr = get_O_Ca_distance(anion_query, cation_query, u, frame)
                    if dist_arr.min() >= args.rcutoff_cation_O_tfsi:
                        df = df.drop(index) 
                    else:
                        df.loc[index, 'cation_O_distance'] = dist_arr.min()
                        break
                # record the most two minimum distances of a ion1 and nearby ion2
                if len(df) == 0:
                    iso_node.append(ids)
                # print(df_all.loc[df.cation_O_distance.nsmallest(2).index.to_list()])
        return iso_node                


def getdfminTFSI_Cation(df, ion1_name, ion2_name, args, u, frame):

        df_all = df.loc[(df["query_resnames"]==ion1_name)]
        df_all = df_all.astype({"distances": float})
        resids = df_all['query'].unique()
       
        min_row_index = []
        for ids in resids:
            df = df_all[df_all['query']==ids] 
            filter = (df['resnames']==ion2_name)
            df = df[filter]
            
            if len(df)>0:
                for index, row in df.iterrows():
                    cation_query = row['pairs']
                    anion_query = row['query']
                    dist_arr = get_O_Ca_distance(anion_query, cation_query, u, frame)
                    if dist_arr.min() >= args.rcutoff_cation_O_tfsi:
                        df = df.drop(index) 
                    else:
                        df.loc[index, 'cation_O_distance'] = dist_arr.min()
                # record the most two minimum distances of a ion1 and nearby ion2
                if len(df)>0:
                    min_row_index.extend(df.cation_O_distance.nsmallest(1).index.to_list())
                # print(df_all.loc[df.cation_O_distance.nsmallest(2).index.to_list()])
        return df_all.loc[min_row_index]    

def getdfminCation_TFSI(df, ion1_name, ion2_name, args, u, frame):

        df_all = df.loc[(df["query_resnames"]==ion1_name)]
        df_all = df_all.astype({"distances": float})
        resids = df_all['query'].unique()
       
        min_row_index = []
        for ids in resids:
            df = df_all[df_all['query']==ids] 
            filter = (df['resnames']==ion2_name)
            df = df[filter]
            
            if len(df)>0:
                for index, row in df.iterrows():
                    cation_query = row['query']
                    anion_query = row['pairs']
                    dist_arr = get_O_Ca_distance(anion_query, cation_query, u, frame)
                    if dist_arr.min() >= args.rcutoff_cation_O_tfsi:
                        df = df.drop(index) 
                    else:
                        df.loc[index, 'cation_O_distance'] = dist_arr.min()
                # record the most two minimum distances of a ion1 and nearby ion2
                if len(df)>0:
                    min_row_index.extend(df.cation_O_distance.nsmallest(1).index.to_list())
                # print(df_all.loc[df.cation_O_distance.nsmallest(2).index.to_list()])
        return df_all.loc[min_row_index]   

def getdfminCation_TFSI_simple(df, ion1_name, ion2_name, args, u, frame):

        df_all = df.loc[(df["query_resnames"]==ion1_name)]
        df_all = df_all.astype({"distances": float})
        resids = df_all['query'].unique()
       
        iso_node = []
        for ids in resids:
            df = df_all[df_all['query']==ids] 
            filter = (df['resnames']==ion2_name)
            df = df[filter]
            
            if len(df) == 0:
                iso_node.append(ids)
                
            elif len(df)>0:
                for index, row in df.iterrows():
                    cation_query = row['query']
                    anion_query = row['pairs']
                    dist_arr = get_O_Ca_distance(anion_query, cation_query, u, frame)
                    if dist_arr.min() >= args.rcutoff_cation_O_tfsi:
                        df = df.drop(index) 
                    else:
                        df.loc[index, 'cation_O_distance'] = dist_arr.min()
                        break
                # record the most two minimum distances of a ion1 and nearby ion2
                if len(df)==0:
                    iso_node.append(ids)
                # print(df_all.loc[df.cation_O_distance.nsmallest(2).index.to_list()])
        return iso_node

def getdfminCa2_TFSI(df, ion1_name, ion2_name, args, u, frame):

        df_all = df.loc[(df["query_resnames"]==ion1_name)]
        df_all = df_all.astype({"distances": float})
        resids = df_all['query'].unique()
       
        min_row_index = []
        for ids in resids:
            df = df_all[df_all['query']==ids] 
            filter = (df['resnames']==ion2_name)
            df = df[filter]
            
            if len(df)>2:
                # print(df)
                
                for index, row in df.iterrows():
                    cation_query = row['query']
                    anion_query = row['pairs']
                    dist_arr = get_O_Ca_distance(anion_query, cation_query, u, frame)
                    if dist_arr.min() >= args.rcutoff_cation_O_tfsi:
                        df = df.drop(index) 
                    else:
                        df.loc[index, 'cation_O_distance'] = dist_arr.min()
                
                # print(df)
                
                # record the most two minimum distances of a ion1 and nearby ion2
                if len(df)>0:
                    min_row_index.extend(df.cation_O_distance.nsmallest(2).index.to_list())
                
                # print(df_all.loc[df.cation_O_distance.nsmallest(2).index.to_list()])
        
        return df_all.loc[min_row_index]   

def msd_c_windowed(positions):
    r""" Calculates the MSD via the simple "windowed" algorithm.

    """
    n_frames = len(positions)
    lagtimes = np.arange(1, n_frames)
    msd_collective = np.zeros(n_frames)
    # positions = self._position_array.astype(np.float64)
    for lag in lagtimes:
        disp = positions[:-lag] - positions[lag:]
        sqdist = np.square(disp).sum(axis=-1)
        msd_collective[lag] = np.mean(sqdist, axis=0)
    # self.results.timeseries = self.results.msds_by_particle.mean(axis=1)
    return msd_collective

def cal_cond_einstein(args, prefix, fit_percentage_begin, fit_percentage_end,\
                        start_time, end_time, Temp, plotOrNot=True):
    
    # Boltzmann constant
    kb = 1.38064852e-23 # J/K
    # elementary charge
    e = 1.60217662e-19 # C
    T = Temp # K
    # Avogadro's number
    Na = 6.022140857e23 # mol^-1
    

    msd_c = []
    box_volume_ave = []
    for idx, path in enumerate(glob.glob(os.path.join(args.dir,prefix))):
        center_of_charge = []
        time = []
        box_volume = []
        
        if not bool(glob.glob(os.path.join(path,'traj_nojump.xtc'))):
            gmx_trajconv = 'echo 0| gmx trjconv \
                            -f '+path+'/msd.xtc \
                            -s '+path+'/topol.tpr \
                            -o '+path+'/traj_nojump.xtc \
                            -pbc nojump '
            os.system(gmx_trajconv)
        
        u = mda.Universe(args.tpr,os.path.join(path,'traj_nojump.xtc'))
        # tot_abs_charge = sum(np.abs(u.atoms.charges))
        
        for ts in u.trajectory:
            if ts.time > start_time and ts.time < end_time:
                print('\r Time = {:.3f}'.
                    format(ts.time),end='')
                temp = (ts.positions*u.atoms.charges[:,None]).sum(axis=0)
                center_of_charge.append(temp)
                time.append(ts.time)
                box_volume.append(np.linalg.det(ts.triclinic_dimensions))
                
        center_of_charge = np.stack(center_of_charge, axis=0 )
        time = np.array(time)
        box_volume_ave.append(np.array(box_volume).mean()*1e-30)
        msd_c.append(msd_c_windowed(center_of_charge))

    msd_c = np.mean(msd_c, axis=0)
    start_t = int(fit_percentage_begin*len(time))
    end_t = int(fit_percentage_end*len(time))
    time_fit = time[start_t:end_t]
    msd_c_fit = msd_c[start_t:end_t]

    # use numpy to do linear fitting
    coef = np.polyfit(time_fit, msd_c_fit, 1)
    poly1d_fn = np.poly1d(coef) 
    
    if plotOrNot:
        plt.plot(time,msd_c,'ro-')
        plt.plot(time_fit,poly1d_fn(time_fit),'bs--')
        plt.show()

    # convert unit from A2/ps to m2/s and calculate diffution
    diff_c = coef[0]*1e-20/1e-12/6

    box_volume_ave = sum(box_volume_ave)/len(box_volume_ave) # A^3 to m^3
    sigma_e = (diff_c)*e*e/(kb*T*box_volume_ave)
    
    return diff_c, sigma_e

def cal_cond_nernst_einstein(args, fit_percentage_begin, fit_percentage_end,\
                             start_time, end_time, Temp, plotOrNot=True):
    
    # Boltzmann constant
    kb = 1.38064852e-23 # J/K
    # elementary charge
    e = 1.60217662e-19 # C
    T = Temp # K
    # Avogadro's number
    Na = 6.022140857e23 # mol^-1
    
    u = mda.Universe(args.tpr, args.trr)  
    dt = u.trajectory.dt
    cation_MSD = msd.EinsteinMSD(u, select='resname '+args.cation_name, msd_type='xyz', fft=True)
    anion_MSD = msd.EinsteinMSD(u, select='resname '+args.anion_name, msd_type='xyz', fft=True)
    cation_MSD.run(start=int(start_time/dt), stop=int(end_time/dt))
    anion_MSD.run(start=int(start_time/dt), stop=int(end_time/dt))
    
    msd_cation = cation_MSD.results.timeseries
    msd_anion = anion_MSD.results.timeseries
    
    box_volume = []    
    for ts in u.trajectory[-100:]:
        print('\r Time = {:.3f}'.
            format(ts.time),end='')
        box_volume.append(np.linalg.det(ts.triclinic_dimensions))
            
    time = cation_MSD.times
    box_volume = np.array(box_volume)

    start_t = int(fit_percentage_begin*len(time))
    end_t = int(fit_percentage_end*len(time))
    time_fit = time[start_t:end_t]
    msd_c_fit = msd_cation[start_t:end_t]
    msd_a_fit = msd_anion[start_t:end_t]

    # use numpy to do linear fitting
    coef_cation = np.polyfit(time_fit, msd_c_fit, 1)
    poly1d_fn_cation = np.poly1d(coef_cation) 
    coef_anion = np.polyfit(time_fit, msd_a_fit, 1)
    poly1d_fn_anion = np.poly1d(coef_anion) 
    
    if plotOrNot:
        plt.plot(time,msd_cation,'ro-')
        plt.plot(time_fit,poly1d_fn_cation(time_fit),'rs--')
        plt.plot(time,msd_anion,'bo-')
        plt.plot(time_fit,poly1d_fn_anion(time_fit),'bs--')
        plt.show()

    # convert unit from A2/ps to m2/s and calculate diffution
    diff_cation = coef_cation[0]*1e-20/1e-12/6
    diff_anion = coef_anion[0]*1e-20/1e-12/6

    box_volume_ave = box_volume.mean()*1e-30 # A^3 to m^3
    
    n_ions = len(u.select_atoms('resname '+args.cation_name+' '+args.anion_name).residues)
    sigma_ne = (diff_cation+diff_anion)*n_ions*e*e/(kb*T*box_volume_ave)
    
    return diff_cation, diff_anion, sigma_ne

def nextpow2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def getDiffusionFromVacf(filename_prefix_ions, plotOrNot=True, fitting_time=-1):
    # time_select = 0.5  # ps
    vacf_ions = {}
    vacf_anion = {}

    kb = 1.380649*1e-23 # J/K
    factor = 1e-18/1e-24 # (A/ps)^2 to (m/s)^2

    font_size = 14
    fig_size_x = 4.5
    fig_size_y = 4.5
    if plotOrNot:   
        fig, axs = plt.subplots(1, 1, figsize=(fig_size_x, fig_size_y))
 
    for ii, vacf_filename in \
        enumerate(glob.glob(filename_prefix_ions)):
        temp = np.loadtxt(open(vacf_filename).readlines()[:-1], skiprows=17)
        vacf_ions[ii] = temp[:,1]*factor
        
        if plotOrNot:
            axs.plot(temp[:,0],vacf_ions[ii])       
        
    time = temp[:,0]
    dt = (time[1] - time[0])*1e-12   # ps to s
    
    vacf_ions_ave = sum(vacf_ions.values()) / len(vacf_ions)  

    if plotOrNot:
        axs.plot(temp[:,0],vacf_ions_ave,'b-')
        
        axs.set_xlim([0,1])

    diff_ions = integrate.cumtrapz(vacf_ions_ave,dx=dt)/3
    diff_ions_smooth = uniform_filter1d(diff_ions,int(len(diff_ions)/10))

    if plotOrNot:
        plt.figure(figsize=(fig_size_x, fig_size_y))
        plt.plot(time[2000:-1],diff_ions[2000:])
        plt.plot(time[2000:-1],diff_ions_smooth[2000:])
        plt.show()

    if fitting_time == -1:
        idx = -1
    else:
        idx = np.searchsorted(time, fitting_time, side='left')

    # diff = diff_ions_smooth[-1]
    return vacf_ions_ave, diff_ions_smooth[idx] 

def getWeightedDiffusionFromVacf(filename_prefix_ions, plotOrNot=True, weight=[1], fitting_time=50):
    # time_select = 0.5  # ps
    vacf_ions = {}
    vacf_anion = {}

    kb = 1.380649*1e-23 # J/K
    factor = 1e-18/1e-24 # (A/ps)^2 to (m/s)^2

    font_size = 14
    fig_size_x = 4.5
    fig_size_y = 4.5
    if plotOrNot:   
        fig, axs = plt.subplots(1, 1, figsize=(fig_size_x, fig_size_y))
 
    for indice, (vacf_filename, w) in \
        enumerate(zip(glob.glob(filename_prefix_ions),weight)):
        ii = (indice,)
        temp = np.loadtxt(open(vacf_filename).readlines()[:-1], skiprows=17)
        vacf_ions[ii] = temp[:,1]*factor*w
        
        if plotOrNot:
            ind  = int(0.8*len(vacf_ions[ii]))
            axs.plot(temp[:ind,0],vacf_ions[ii][:ind])       
        
    time = temp[:,0]
    dt = (time[1] - time[0])*1e-12   # ps to s
    
    vacf_ions_ave = sum(vacf_ions.values()) / sum(weight)  

    if plotOrNot:
        axs.plot(temp[:ind,0],vacf_ions_ave[:ind],'b-')
        axs.set_xlim([0,1])

    diff_ions = integrate.cumtrapz(vacf_ions_ave,dx=dt)/3
    diff_ions_smooth = uniform_filter1d(diff_ions,int(len(diff_ions)/10))

    if plotOrNot:
        plt.figure(figsize=(fig_size_x, fig_size_y))
        plt.plot(time[2000:ind],diff_ions[2000:ind])
        plt.plot(time[2000:ind],diff_ions_smooth[2000:ind])
        plt.show()

    
    # diff = diff_ions_smooth[-1]
    if fitting_time == -1:
        idx = -1
    else:
        idx = np.searchsorted(time, fitting_time, side='left')
        
    return vacf_ions_ave, diff_ions_smooth[idx]

def getDiffusionFromMsd(filename, plotOrNot=True):

    # Open the file for reading
    with open(filename, 'r') as f:

        # Loop over the lines in the file
        for line in f:

            # Use regular expressions to extract the float after the equals sign
            match1 = re.search(r'Li\]\s*=\s*([\d.]+)', line)
            match1_1 = re.search(r'([\d.]+(?:e[+-]?\d+)?)\s+cm\^2/s', line)

            match2 = re.search(r'TFS\]\s*=\s*([\d.]+)', line)
            match2_2 = re.search(r'([\d.]+(?:e[+-]?\d+)?)\s+cm\^2/s', line)
            
            if match1 and match1_1:
                # Convert the matched string to a float
                diff_cation = float(match1.group(1))*float(match1_1.group(1)) /10000 # cm^2/s to m^2/s
            if match2 and match2_2:
                # Convert the matched string to a float
                diff_anion = float(match2.group(1))*float(match2_2.group(1)) /10000 # cm^2/s to m^2/s
    
    
    # Read the data from the XVG file
    if plotOrNot:
        data = np.genfromtxt(filename, delimiter=None, skip_header=25)

        # Extract the columns of data
        x = data[:,0]
        y1 = data[:,1]
        y2 = data[:,2]

        # Plot the data
        plt.plot(x, y1)
        plt.plot(x, y2)


        # Add labels and a title to the plot
        plt.xlabel('Time (ps)')
        plt.ylabel('MSD')
        plt.title('MSD of cation and anion')
        plt.legend(['cation', 'anion'])
        # Show the plot
        plt.show()
    print(diff_cation, diff_anion)
    return diff_cation, diff_anion

def getDiffusionFromMDAnalysisMSD(args, prefix, beginFit, endFit, plotOrNot=True):
        msd_cation_results = []
        msd_anion_results = []
        for idx, path in enumerate(glob.glob(os.path.join(args.dir,prefix))):
            if not bool(glob.glob(os.path.join(path,'traj_nojump.xtc'))):
                gmx_trajconv = 'echo 0| gmx trjconv \
                                -f '+path+'/msd.xtc \
                                -s '+path+'/topol.tpr \
                                -o '+path+'/traj_nojump.xtc \
                                -pbc nojump '
                os.system(gmx_trajconv)
            u=mda.Universe(args.tpr, os.path.join(path,'traj_nojump.xtc'))
            MSD_cation = msd.EinsteinMSD(u, select='resname Li', msd_type='xyz', fft=True)
            MSD_anion = msd.EinsteinMSD(u, select='resname TFS', msd_type='xyz', fft=True)
            MSD_cation.run()
            MSD_anion.run()
            msd_cation_results.append(MSD_cation.results.msds_by_particle)
            msd_anion_results.append(MSD_anion.results.msds_by_particle)
        
        combined_cation_msds = np.concatenate(msd_cation_results, axis=1)
        combined_anion_msds = np.concatenate(msd_anion_results, axis=1)
        average_cation_msd = np.mean(combined_cation_msds, axis=1)
        average_anion_msd = np.mean(combined_anion_msds, axis=1)

        nframes = MSD_cation.n_frames
        timestep = u.trajectory.dt # this needs to be the actual time between frames
        lagtimes = np.arange(nframes)*timestep # make the lag-time axis

        start_time = beginFit
        start_index = int(start_time/u.trajectory.dt)
        end_time = endFit
        end_index = int(end_time/u.trajectory.dt)
        linear_model = linregress(lagtimes[start_index:end_index],
                                                    average_cation_msd[start_index:end_index])
        slope = linear_model.slope
        error = linear_model.stderr

        D_cation = slope * 1/(6) * 1e-8 # convert A^2/ps to m^2/s
        
        
        linear_model = linregress(lagtimes[start_index:end_index],
                                                    average_anion_msd[start_index:end_index])
        slope = linear_model.slope
        error = linear_model.stderr
        
        D_anion = slope * 1/(6) * 1e-8 # convert A^2/ps to m^2/s
        
        if plotOrNot:
            middle_index = int((start_index+end_index)/2)
            residue_cation = lagtimes[middle_index]*D_cation*6*1e8 - average_cation_msd[middle_index]
            residue_anion = lagtimes[middle_index]*D_anion*6*1e8 - average_anion_msd[middle_index]
            plt.figure()
            plt.plot(lagtimes,average_cation_msd)
            plt.plot(lagtimes,average_anion_msd)
            plt.plot(lagtimes[start_index:end_index],lagtimes[start_index:end_index]*D_cation*6*1e8-residue_cation)
            plt.plot(lagtimes[start_index:end_index],lagtimes[start_index:end_index]*D_anion*6*1e8-residue_anion)
            plt.legend(['cation','anion','cation fit','anion fit'])
            plt.show()
        return D_cation, D_anion
    
def calCondFromECACF(ncases, gro_dir_prefix, xtc_dir_prefix, Temperature, plotOrNot=True):
    caf_data = {}
    caf_s = {}
    caf_cum_smooth_s = {}
    vol = {}
    T = Temperature # K
    kb = 1.380649*1e-23 # J/K
    factor = 1/(3*(kb*T)) * (1.602176634*1e-19)*(1.602176634*1e-19)*1e-18/1e-24 # 1/3KbT * e^2 * nm ^2 / ps^2

    font_size = 14
    fig_size_x = 4.5
    fig_size_y = 9
    if plotOrNot:
        fig, axs = plt.subplots(2, 1, figsize=(fig_size_x, fig_size_y))
        
    for ii in range(0, ncases):
        gro_filename =  gro_dir_prefix+str(ii)+'.gro'
        with open(gro_filename) as f:
            for line in f:
                pass
            last_line = line
            box = [float(x) for x in last_line.split()]
            vol[ii] = np.prod(box)*1e-27

    for ii in range(0, ncases):
        caf_filename = xtc_dir_prefix + str(ii) + '.xvg'
        temp = np.loadtxt(caf_filename, delimiter='\t', skiprows=18)
        caf_data[ii] = np.divide(temp[:,1],vol[ii])
        axs[0].plot(temp[:,0],caf_data[ii]*factor)
    
    time = temp[:,0]
    dt = (time[1] - time[0])*1e-12   
    
    caf = sum(caf_data.values()) / len(caf_data) * factor 
    axs[0].plot(temp[:,0],caf,'b-')
    axs[0].set_xlim([0,1])

    caf_cum = integrate.cumtrapz(caf,dx=dt)
    caf_cum_smooth = uniform_filter1d(caf_cum,int(len(caf_cum)/10))
    axs[1].plot(time[2000:-1],caf_cum[2000:])
    axs[1].plot(time[2000:-1],caf_cum_smooth[2000:])
    plt.show()
    # plt.ylim([1e-5,4e-5])
    
    cond = caf_cum_smooth[-1]
    return cond

def average_arrays(arrays):
    # find the maximum length of arrays in the list
    lengths = [len(array) for array in arrays]
    max_length = max(lengths)

    # pad the shorter arrays with zeros to make them the same length as the longest array
    padded_arrays = [np.pad(array, (0, max_length - len(array)), 'constant') for array in arrays]

    # compute the element-wise average while excluding padded zeros
    summed_array = np.zeros(max_length)
    count_array = np.zeros(max_length)
    for array in padded_arrays:
        summed_array += array
        count_array += (array != 0)
    
    count_array[0] = len(arrays)
    average_array = summed_array / count_array
    
    return average_array, lengths

def find_all_ones_segments(arr, min_length):
    segments = []
    seg_start = None

    for i in range(len(arr)):
        if arr[i] == 1 and seg_start is None:
            seg_start = i
        elif arr[i] != 1 and seg_start is not None:
            seg_end = i - 1
            if seg_end - seg_start + 1 >= min_length:
                segments.append((seg_start, seg_end))
            seg_start = None

    if seg_start is not None and len(arr) - seg_start >= min_length:
        segments.append((seg_start, len(arr) - 1))

    return segments

def get_msd_for_one_residue_based_on_segments(resid, start, end, skip, trajectory, residues):
    positions = np.zeros((len(trajectory[start:end:skip]), 3))
    for i, ts in enumerate(trajectory[start:end:skip]):
        positions[i] = residues[resid].atoms.center_of_mass()
    msd = msd_c_windowed(positions)
    return msd

def get_average_msd_based_on_segments(resids, smallCluster_or_not, ns_frames, life_time, trajectory, residues, dt, skip):
    msd = []
    for i, arr in enumerate(smallCluster_or_not):
        resid = resids[i]
        interval = int(life_time/((ns_frames[1]-ns_frames[0])*dt))
        if interval < 2:
            interval = 2
        segments = find_all_ones_segments(arr, min_length=interval)

        for segment in segments:
            start, end = ns_frames[segment[0]], ns_frames[segment[1]]
            msd.append(get_msd_for_one_residue_based_on_segments(resid, start, end, skip, trajectory, residues))
    msd_ave, lengths = average_arrays(msd)
    return msd_ave, lengths

def getDiffusionForFreeIonsBasedOnSegmentedMSD(args, prefix, expected_lifetime, skip, beginFit, endFit, plotOrNot=True):
    msd_free = []
    lengths = []
    for path in glob.glob(os.path.join(args.dir,prefix)):
        args.trr = os.path.join(path,'msd.xtc')
        smallCluster_or_not, ns_frames, query_resids, times = graph_theory_based_clustering_simple(args)

        u = mda.Universe(args.tpr, os.path.join(path,'traj_nojump.xtc'))
        # skip  = 100 
        msd_ave_temp, length_temp = get_average_msd_based_on_segments(query_resids,smallCluster_or_not, ns_frames, expected_lifetime, \
                                                            u.trajectory, u.residues, u.trajectory.dt, skip)
        msd_free.append(msd_ave_temp)
        length_temp = np.array(length_temp)*skip*u.trajectory.dt
        lengths.append(length_temp)
    
    lengths = np.concatenate(lengths)
    
    msd_free_ave, _ = average_arrays(msd_free)
    msd_time = np.arange(0, len(msd_free_ave)*skip*u.trajectory.dt, skip*u.trajectory.dt)

    
    start_time = beginFit
    start_index = min(range(len(msd_time)), key=lambda i: abs(msd_time[i]-start_time))
    end_time = endFit
    end_index = min(range(len(msd_time)), key=lambda i: abs(msd_time[i]-end_time))
    linear_model = linregress(msd_time[start_index:end_index],
                                                msd_free_ave[start_index:end_index])
    slope = linear_model.slope
    error = linear_model.stderr
    D_free = slope * 1/(6) * 1e-8
    
    if plotOrNot:
        middle_index = int((start_index+end_index)/2)
        residue = msd_time[middle_index]*D_free*6*1e8 - msd_free_ave[middle_index]
        plt.figure()
        plt.plot(msd_time,msd_free_ave)
        plt.plot(msd_time[start_index:end_index],msd_time[start_index:end_index]*D_free*6*1e8-residue)
        plt.legend(['free ion','free ion fit'])
        plt.show()
    return D_free, msd_free_ave, msd_time, lengths

def calculate_sigma_cne(cluster_population, diff_cluster, args, u, box_volume):
    # Boltzmann constant
    kb = 1.38064852e-23 # J/K
    # elementary charge
    e = 1.60217662e-19 # C
    # Avogadro's number
    Na = 6.022140857e23 # mol^-1

    sigma_cne = 0

    # Calculate the number of cations and anions
    n_cation = len(u.select_atoms('resname ' + args.cation_name).residues)
    n_anion = len(u.select_atoms('resname ' + args.anion_name).residues)

    # Calculate the charge per cation and anion
    charge_cation = u.select_atoms('resname ' + args.cation_name).total_charge() / n_cation
    charge_anion = u.select_atoms('resname ' + args.anion_name).total_charge() / n_anion

    sigma_contributions = np.zeros((args.cluster_limit + 1, args.cluster_limit + 1))

    for i in range(args.cluster_limit):
        for j in range(args.cluster_limit):
            a = cluster_population[i][j]
            cluster_charge = i * charge_cation + j * charge_anion
            
            D = diff_cluster[i][j]

            sigma_contributions[i, j] = D * a * (cluster_charge ** 2) * e ** 2 / (kb * args.temperature * box_volume)
            sigma_cne += sigma_contributions[i, j]

    return sigma_contributions, sigma_cne

def graph_theory_based_clustering_simple(args):

    # read traj and perform ns
    u = mda.Universe(args.tpr, args.trr)  
    dt = u.trajectory.dt
    ns, times, ns_frames, nframe = get_neighbour_search(args)
    times = times - times[0] + args.begin
    
    # get resids, resnames, charges
    query_resids = u.select_atoms(args.selection).residues.resids
    dict_ind = dict(zip(query_resids, range(len(query_resids))))

    
    # initial output variables
    smallCluster_or_not = np.zeros((len(query_resids), nframe), dtype='int')
    
    for i, frame in zip(range(nframe),ns_frames):
        # Cation list
        if args.anionIsTFSI:
            if args.cationIsCa2:
                df_cation_min = getdfminCa2_TFSI(ns[i], args.cation_name, args.anion_name, args, u, frame)
            else:
                df_cation_min = getdfminCation_TFSI_simple(ns[i], args.cation_name, args.anion_name, args, u, frame)
        else:
            df_cation_min = getdfmin(ns[i], args.cation_name, args.anion_name)
                
        # Anion list
        if args.anionIsTFSI:
            df_anion_min = getdfminTFSI_Cation_simple(ns[i], args.anion_name, args.cation_name, args, u, frame)
        else:
            df_anion_min = getdfmin(ns[i], args.anion_name, args.cation_name)
        
        iso_nodes = df_cation_min+df_anion_min
        for node in iso_nodes:
            smallCluster_or_not[dict_ind[node]][i] = 1
        
    return smallCluster_or_not, ns_frames, query_resids, times

def graph_theory_based_clustering(args):

    # read traj and perform ns
    u = mda.Universe(args.tpr, args.trr)  
    dt = u.trajectory.dt
    ns, times, ns_frames, nframe = get_neighbour_search(args)
    times = times - times[0] + args.begin
    
    # get resids, resnames, charges
    query_resids = u.select_atoms(args.selection).residues.resids
    query_resnames = u.select_atoms(args.selection).residues.resnames
    query_charges = [np.around(x) for x in u.select_atoms(args.selection).residues.charges]
    dict_names = dict(zip(query_resids, query_resnames))
    dict_charges = dict(zip(query_resids, query_charges))
    dict_ind = dict(zip(query_resids, range(len(query_resids))))
    dict_ind_reverse = dict(zip(range(len(query_resids)),query_resids))

    
    # initial output variables
    cluster_all = []
    cluster_trajectory = np.zeros((len(query_resids), nframe), dtype='U6')
    smallCluster_or_not = np.zeros((len(query_resids), nframe), dtype='int')
    num_edge_for_nodes = np.zeros((len(query_resids), nframe), dtype='int')
    free_or_not_for_nodes = np.zeros((len(query_resids), nframe), dtype='int')
    cluster_population = np.zeros((args.cluster_limit+1,args.cluster_limit+1), dtype='int')
    # initialize a list of lists
    linkofeachnode = [[] for i in range(len(query_resids))]
    
    for i, frame in zip(range(nframe),ns_frames):
        
        # Cation list
        if args.anionIsTFSI:
            if args.cationIsCa2:
                df_cation_min = getdfminCa2_TFSI(ns[i], args.cation_name, args.anion_name, args, u, frame)
            else:
                df_cation_min = getdfminCation_TFSI(ns[i], args.cation_name, args.anion_name, args, u, frame)
        else:
            df_cation_min = getdfmin(ns[i], args.cation_name, args.anion_name)
                
        # Anion list
        if args.anionIsTFSI:
            df_anion_min = getdfminTFSI_Cation(ns[i], args.anion_name, args.cation_name, args, u, frame)
        else:
            df_anion_min = getdfmin(ns[i], args.anion_name, args.cation_name)
        

        
        #  creat graph based on linkages
        linkList = list(zip(df_cation_min['query'],df_cation_min['pairs'])) + \
                    list(zip(df_anion_min['query'],df_anion_min['pairs'])) 
        clusters = subgraph(query_resids, linkList, dict_names)
       
        # loop over each subgraph
        for s in clusters:
            
            nodes = sorted(list(s.nodes))
            cluster_names = list(map(dict_names.get, nodes))
            cluster_charges =  list(map(dict_charges.get, nodes))

            cluster_n_cation = len([num for num in cluster_charges if num >= 1])
            cluster_n_anion = len([num for num in cluster_charges if num <= -1])
            
            # if cluster_n_cation > upper limit, set cluster_n_cation = 10:
            if cluster_n_cation > args.cluster_limit:
                cluster_n_cation = args.cluster_limit
            if cluster_n_anion > args.cluster_limit:
                cluster_n_anion = args.cluster_limit
                
            cluster_population[cluster_n_cation][cluster_n_anion] += 1
            
            cluster_ids_str = str(len(cluster_charges)) + '_' + \
                        str(cluster_n_cation)
            if cluster_ids_str not in cluster_all:
                if args.plotOrNot:
                    plotsubgraph(s, dict_names)
                    print(s.nodes)
            cluster_all.append(cluster_ids_str)                
            
            # loop over each node in the subgraph
            for node in nodes:   
                if abs(sum(cluster_charges)) > 1e-5: 
                    if len(nodes) < args.small_cluster:
                        smallCluster_or_not[dict_ind[node]][i] = len(nodes)
                    else:
                        smallCluster_or_not[dict_ind[node]][i] = args.small_cluster
                                     
                num_edge_for_nodes[dict_ind[node]][i] = len(s.edges(node))
                cluster_trajectory[dict_ind[node]][i] = cluster_ids_str
                
                if i == 0:
                    linkofeachnode[dict_ind[node]] = list(s.edges(node))
                else:                         
                    if not check_same_element(linkofeachnode[dict_ind[node]], list(s.edges(node))):
                        free_or_not_for_nodes[dict_ind[node]][i] = 1
                
                linkofeachnode[dict_ind[node]] = list(s.edges(node))

    counter = Counter(cluster_all).most_common()
    
    # get index of atoms in small clusters
    if args.write_small_cluster:
        
        mode_cluster_for_atoms, mode_rates, mode_indices = \
            get_mode_indices_with_threshold(smallCluster_or_not,threshold=args.threshold)

        with open(args.dir+args.small_cluster_ndx_filename, 'w') as f:
            for mode_val, indices in enumerate(mode_indices):
                # zero mode_val means the cluster has zero charge
                if bool(indices) and  mode_val != 0 :
                    transformed = np.concatenate([(u.residues[dict_ind_reverse[i]-1].atoms.ids+1).tolist() for i in indices])
                    f.write(f'[{mode_val}]\n')
                    max_length = 10  # maximum length of each line in the file
                    for i in range(0, len(transformed), max_length):
                        line = ' '.join(str(x) for x in transformed[i:i+max_length])
                        f.write(line + '\n')
                    f.write('\n')
    
    if args.plotOrNot:
        plt.plot(times,free_or_not_for_nodes.mean(axis=0))
        plt.show()
            
    return cluster_population/nframe, counter, ns_frames
    

# ## main function
# # input parameters
# my_arg = argparse.ArgumentParser('My argument parser')
# args = my_arg.parse_args('')

# args.tpr = '300K.tpr'
# args.xtc = '300K_nojump.xtc'
# temperature = 300

# cal_cond_einstein(args,0.05,0.3,20000,100000,temperature,True)