#!/usr/bin/env python3

__author__ = "Angel Ling"

import warnings
warnings.filterwarnings('ignore') 
# import obspy
# import numpy as np
import time
from os import path, listdir, makedirs
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from GMV_utils import readevent, read_data_inventory, Normalize, station_phases
from GMV_utils import GMV_plot, GMV_FK, GMV_FK_ray, GMV_Xsec

# ====================
# %% Parameter Box
prog_starttime = time.time() # Execution time of this program

## Event name (Folder name)
# event_name = "Nevada_20200515_M6.5"
# event_name = "Alaska_20180815_M6.5"
# event_name = "Mexico_20200623_M7.4"
# event_name = "NewZealand_20200618_M7.4"
# event_name = "Alaska_20200722_M7.8"
# event_name = "Mexico_20180216_M7.2"
# event_name = "Swiss_20201025_M4.5"
# event_name = "Alaska_20181130_M7.1"
# event_name = "Indonesia_20180928_M7.5"
event_name= "PNG_20170122_M7.9"

# Location of the data directoy
directory  = "/Users/angelling/Documents/obspyDMT/syngine_data/" 
#directory  = "/home/aling/obspyDMT/CorePhases/" # Tris

# Location of the figure directoy
fig_directory = '/Users/angelling/Documents/ETH/figures/' 
#fig_directory = '/home/aling/obspyDMT/CorePhases/' # Tris

# Path of AA logo
logo_loc = '/Users/angelling/Documents/ETH/figures/AA_logo.png'
#logo_loc = '/home/aling/AA_logo.png' # Tris

# Choose the starting and ending time in seconds after OT shown in the video (min: 0, max: 7200)
start = 500   # min: 0    # default: 500
end   = 7000  # max: 7200 # default: 7000

# Start and end of movie in seconds (min: 0, max: 7200)
start_movie = 1200  # min: 0, test: 1300
end_movie   = 3000  # max: 7200, test: 3150

# Choose movie interval (default: 1s)
select_interval = True # default: False 
interval        = 500  # in s, test: 500 (every 500s), default doesn't read this

# Choose bandpass freq
f1 = 1/200. # min freq, default 1/200
f2 = 1/50.  # max freq, default 1/50

# Select a reference station to plot by network, name, and location (e.g. CH.FUSIO.,CH.GRIMS.)
# if plot_Xsec is True, this will be ignored
station = "CH.GRIMS." 
# station = "Z3.A061A."
# station = "Z3.A025A.00"
# station = "Z3.A033A.00"

# Select phases to plot on seismograms
model     = "iasp91" # background model (e.g. iasp91/ak135)
# phases    = ["Pg","Sg"] [Local events]
# phases    = ['P','PcP','PP','PPP','S','SS','SSS','SKS','4kmps','4.4kmps'] # Phases to plot on seismograms [Teleseismic events]
# phases    = ['P','Pdiff','PKP','PKIKP','PP','PPP','S','SS','SSS','SKS','SKKS','4kmps','4.4kmps'] # Phases to plot on seismograms [100deg<dist<120deg]
phases    = ['Pdiff','PKP','PKiKP','PKIKP','PP','PPP','SS','SSS','SKS','SKKS','4kmps','4.4kmps','SKKKS','SKSP','PPPS','SSP']  # Phases to plot on seismograms [Core events]

# Plotting parameters for movies
plot_local  = False # Plot local earthquake instead of teleseismic (GMV only; enter event info in the utility code)
read_obs    = False # Read OBS stations (CH*)
plot_3c     = True  # Plot 3C motion 
plot_rotate = True  # Plot ZRT instead of ZNE in the simple map seismograms
plot_Xsec   = False # Plot Xsec instead of seismograms (includes rotation)
plot_FK     = False # Plot FK-analysis
plot_FK_ray = True  # Plot FK-analysis + ray path
plot_save   = True # Save plots in movie directory

# Parameters for GMV
vmin = -0.1   # colorbar min, default: -0.1
vmax =  0.1   # colorbar max, default: 0.1

## Parameters for FK analysis
sx       = 0.5     # slowness in x direction [s/km]
sy       = 0.5     # slowness in y direction [s/km]
sx_m     = 0.1     # slowness resolution splitting point [s/km] (must be smaller than sx/sy)
sl_s     = 0.02    # slowness step [s/km]
slow_unit    = "sdeg" # slowness unit in s/km:"skm" or s/deg:"sdeg"
plot_CH_only = False  # use only CH stations for FK analysis
# Parameters for 1st domain (P waves)
f11      = 1/30    # min freq for fk anlysis1
f22      = 1/5     # max freq for fk anlysis1
win_frac = 0.05    # window fraction [s]
interval_win = 60. # window length [s]
win_len  = interval_win/2  # sliding window length [s]
# Parameters for 2nd domain (Other waves)
f111          = 1/200.      # min freq for fk anlysis2
f222          = 1/50.       # max freq for fk anlysis2
interval_win2 = 200.       # window length2 [s]
win_len2 = interval_win2/2 # sliding window length2 [s] 
# Normalization for FK plotting
minval    = 0.1    # min value, default: 0.1
maxval    = 8.     # max value, default: 10.

# Parameters for dispersion curve
plot_DispC = False # Plot dispersion curve
start_dc   = 2000. # start of the trace
end_dc     = 4000. # end of the trace

# ====================
# %% Read catalog 

# Data directories
directory = directory + event_name + "/"

for f in listdir(directory):
    if f.endswith('.a'):
        folder = f
        
data_directory    = directory + folder +"/"
rawdata_directory = data_directory + "raw/"
prodata_directory = data_directory + "processed/"
resp_directory    = data_directory + "resp/"
syn_iasp91_directory = data_directory + "syngine_iasp91_2s/"
syn_ak135_directory  = data_directory + "syngine_ak135f_2s/"

# Set up figure directories
fig_directory = fig_directory + "figures_" + event_name + "/"
if not path.exists(fig_directory):  # If figure directory doesn't exist, it will create one
    makedirs(fig_directory)
if plot_CH_only:
    movie_directory = fig_directory + "movie_CH/" # movie directory for CH array
else:
    movie_directory = fig_directory + "movie/"
if not path.exists(movie_directory):  # If movie directory doesn't exist, it will create one
    makedirs(movie_directory)
    print("New movie directory "+movie_directory+" is created.")
else:
    print("Movie directory "+movie_directory+ " exists.")

## Read event catalog (Read QUAKEML or pkl)
# event_dic = readevent(event_name, data_directory)
event_dic = readevent(event_name, data_directory, local=plot_local)

# ==================== 
# %% Read data and inventory

## Read processed data and inventories    
data_dic = read_data_inventory(prodata_directory, resp_directory, event_dic, read_obs=read_obs, plot_3c=plot_3c)

# ==================== 
# %% Normalize displacement and store good data

## Filter and normalize one single event
GMV, stream_info = Normalize(data_dic, event_dic, f1, f2, start, end, plot_Xsec=plot_Xsec)

# ====================
# %% Plot dispersion curve of reference station

if plot_DispC:
    from GMV_utils import plot_dispersion_curve 
    plot_dispersion_curve(data_dic, event_dic, station, "Z", start_dc, end_dc, movie_directory)

# ====================
# %% Select reference station and streams for FK analysis

# Select stations for plotting cross-section
if plot_Xsec:
    from GMV_utils import select_Xsec
    Xsec_dict = select_Xsec(GMV)
    # Select ceneter station of the cross-section
    thechosenone_Xsec = station_phases(GMV, Xsec_dict["center_station"], event_dic, model, phases)
else:
    # Select a reference station for plotting seismogram
    thechosenone = station_phases(GMV, station, event_dic, model, phases)

# Only Vertical component for FK analysis
if plot_FK or plot_FK_ray:
    from GMV_utils import select_FK
    freq    = (f11, f22, f111, f222)
    FK_dict = (sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, interval_win2, win_len2, minval, maxval)
    subarray_dict = select_FK(event_dic, data_dic, start, end, model, freq, thechosenone, plot_CH_only=plot_CH_only)
    ## Plot array response function
    # from GMV_utils import plot_ARF
    # plot_ARF(subarray_dict, 5.)

# %% Prepare for plotting 

# AA logo
with get_sample_data(logo_loc) as file_img:
    arr_img = plt.imread(file_img, format='png')
    
# Choose movie interval and turn into index
if select_interval: # use the interval set in the parameter box
    interval = int(interval/stream_info["timestep"])
else: # default (1 Hz)
    interval = int(stream_info["sample_rate"])  

# %% Main GMV plotting
      
## ====================
## FK analysis + ray path plot
if plot_FK_ray:
            
    GMV_FK_ray(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
           start_movie, end_movie, interval,
           vmin, vmax, arr_img, 
           movie_directory, slow_unit=slow_unit,
           plot_save=plot_save, plot_3c=plot_3c, plot_rotate=plot_rotate)

## ====================
## FK analysis plot
elif plot_FK:
        
    GMV_FK(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
           start_movie, end_movie, interval,
           vmin, vmax, arr_img, 
           movie_directory, slow_unit=slow_unit,
           plot_save=plot_save, plot_3c=plot_3c, plot_rotate=plot_rotate)
    
## ====================
# # Cross-section plot
elif plot_Xsec:
    
    GMV_Xsec(GMV, event_dic, stream_info, Xsec_dict, thechosenone_Xsec, 
             start_movie, end_movie, interval,
             vmin, vmax, arr_img, 
             movie_directory, plot_save=plot_save, plot_3c=plot_3c)

## ====================
## Only GMV and reference seismograms
else:
    GMV_plot(GMV, event_dic, stream_info, thechosenone, 
             start_movie, end_movie, interval,
             vmin, vmax, arr_img, 
             movie_directory, plot_save=plot_save,
             plot_local=plot_local, plot_3c=plot_3c, plot_rotate=plot_rotate)
        
    
print("--- %.3f seconds ---" % (time.time() - prog_starttime))
print("---- %.3f mins ----" % ((time.time() - prog_starttime)/60))