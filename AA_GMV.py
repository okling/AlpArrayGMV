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
from operator import itemgetter
from GMV_utils import readevent, read_data_inventory, Normalize, station_phases, select_FK
from GMV_utils import GMV_plot, GMV_FK, GMV_FK_ray, GMV_Xsec, GMV_FK4, GMV_ZeroX, GMV_FK_NM

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
# event_name = "Mexico_20170908_M8.2"
event_name = "PNG_20170122_M7.9"

# Location of the data directoy
directory  = "/Users/angelling/Documents/obspyDMT/syngine_data/" 
#directory  = "/home/aling/obspyDMT/CorePhases/" # Tris

# Location of the figure directoy
fig_directory = '/Users/angelling/Documents/ETH/figures/' 
#fig_directory = '/home/aling/obspyDMT/CorePhases/' # Tris

# Path of AA logo
logo_loc = '/Users/angelling/Documents/ETH/figures/AA_logo.png'
#logo_loc = '/home/aling/AA_logo.png' # Tris

# Waveform setting
# Choose the starting and ending time in seconds after OT shown in reference seismograms (min: 0, max: 7200)
start       = 500  # min: 0    # default: 500
end         = 7000 # max: 7200 # default: 7000 
decimate_fc = 2   # Downsample data by an integer factor (Default: 2)

# Movie setting
# Choose movie interval (default: 1s)
select_interval = True # continuous: False 
interval        = 500  # in s, test: 500 (every 500s), default doesn't read this
timeframes      = [1700,2220,2500] # plot only certain time frame in list or array, e.g.[1772,2220,2527] (Default: None)
# Start and end of movie in seconds if continuous (min: 0, max: 7200)
start_movie     = 1200  # min: 0, test: 1300
end_movie       = 3000  # max: 7200, test: 3150
timelabel       = "s"   # "s"/"min"/"hr" for seismograms

# Choose bandpass freq
f1 = 1/200. # min freq, default 1/200
f2 = 1/50.  # max freq, default 1/50

# Select a reference station to plot by network, name, and location (e.g. CH.FUSIO.,CH.GRIMS.)
# if plot_Xsec is True, this will be ignored
station = "CH.GRIMS." 
# station = "CH.ROTHE."
# station = "CH.HASLI."
# station = "Z3.A061A."
# station = "Z3.A025A.00"
# station = "Z3.A033A.00"
# stations = ["BW.BDGS.", "OE.FETA.", "CH.GRIMS.", "FR.OGVG.00"]
stations = ["HU.CSKK.", "OE.FETA.", "CH.GRIMS.", "FR.OGVG.00"]

# Select station region and threshold
threshold = 0.25 # Default: None; 0.25

# Select phases to plot on seismograms
model  = "iasp91" # background model (e.g. iasp91/ak135)
# phases = ["Pg","Sg"] [Local events]
# phases = ['P','PcP','PP','PPP','S','SS','SSS','SKS','4kmps','4.4kmps'] # Phases to plot on seismograms [Teleseismic events]
# phases = ['P','Pdiff','PKP','PKIKP','PP','PPP','S','SS','SSS','SKS','SKKS','4kmps','4.4kmps'] # Phases to plot on seismograms [100deg<dist<120deg]
phases = ['Pdiff','PKP','PKiKP','PKIKP','PP','PPP','SS','SSS','SKS','SKKS','4kmps','4.4kmps','SKKKS','SKSP','PPPS','SSP']  # Phases to plot on seismograms [Core events]
# phases = ['4kmps','4.4kmps']

# Plotting parameters for movies
plot_local  = False # Plot local earthquake instead of teleseismic (GMV only; enter event info in the utility code)
plot_ZeroX  = False # Plot zero crossing for normal mode 
plot_Vonly  = False # Plot only vertical GMV and seismogram
read_obs    = False # Read OBS stations (CH*)
plot_3c     = True  # Plot 3C motion 
plot_rotate = True  # Plot ZRT instead of ZNE in the simple map seismograms
plot_Xsec   = False # Plot Xsec instead of seismograms (includes rotation)
plot_FK     = False # Plot FK-analysis
plot_FK_ray = True # Plot FK-analysis + ray path
plot_FK4    = False # Plot 4 FK-analysis
plot_FK_NM  = False # Plot FK-analysis for Normal Mode
plot_save   = False # Save plots in movie directory
save_option = "png" # Save format (Default: "png")
save_dpi    = 120   # Saved figure resolution (Default: 120)

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
f11      = 1/30    # min freq for fk anlysis1, e.g. 1/30
f22      = 1/5     # max freq for fk anlysis1, e.g. 1/5
win_frac = 0.05    # window fraction [s], Default: 0.05
interval_win = 60. # window length [s], e.g. 60
win_len  = interval_win/2  # sliding window length [s]
# Parameters for 2nd domain (Other waves)
f111          = 1/200.     # min freq for fk anlysis2, e.g. 1/200
f222          = 1/50.      # max freq for fk anlysis2, e.g. 1/50
interval_win2 = 200.       # window length2 [s], e.g. 200
win_len2 = interval_win2/2 # sliding window length2 [s] 
# Normalization for FK plotting
minval    = 0.1    # min value, default: 0.1
maxval    = 8.     # max value, default: 10., e.g. 8.
# Station list for FK analysis
station_list = ["CH.ROTHE.","CH.ROTHE.","CH.BALST.","CH.BERNI.","CH.BNALP.","CH.BRANT.","CH.DAGMA.","CH.DAVOX.","CH.DIX.","CH.FUORN.","CH.FUSIO.","CH.GRIMS.","CH.HASLI.","CH.ILLEZ.","CH.LIENZ.","CH.LLS.","CH.METMA.","CH.MMK.","CH.MTI02.","CH.MUGIO.","CH.PANIX.","CH.PLONS.","CH.SALAN.","CH.SALEV.","CH.SENIN.","CH.SIMPL.","CH.SLE.","CH.SULZ.","CH.TORNY.","CH.TRULL.","CH.VDL.","CH.VDR.","CH.VINZL.","CH.VMV.","CH.WILA.","CH.WIMIS.","CH.ZUR.","FR.BOUC.00","FR.CHMF.00","FR.OGVA.00","FR.RONF.00","GU.CIRO.","GU.TRAV.","IV.MDI.","OE.DAVA."]

# Parameters for dispersion curve
plot_DispC = False # Plot dispersion curve
start_dc   = 2000. # start of the trace
end_dc     = 4000. # end of the trace

# ====================
# %% Read event catalog 

# Data directories
directory = directory + event_name + "/"

for f in listdir(directory):
    if f.endswith('.a'):
        folder = f
    elif f in ["continuous1"]:
        folder = f
        
data_directory    = directory + folder +"/"
rawdata_directory = data_directory + "raw/"
prodata_directory = data_directory + "processed/"
resp_directory    = data_directory + "resp/"
syn_iasp91_directory = data_directory + "syngine_iasp91_2s/"
syn_ak135_directory  = data_directory + "syngine_ak135f_2s/"

# Set up figure directory
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
GMV, stream_info = Normalize(data_dic, event_dic, f1, f2, start, end, decimate_fc=decimate_fc, threshold=threshold, plot_Xsec=plot_Xsec, plot_ZeroX=plot_ZeroX)

# ====================
# %% Plot dispersion curve of reference station

if plot_DispC:
    from GMV_utils import plot_dispersion_curve 
    plot_dispersion_curve(data_dic, event_dic, station, "Z", start_dc, end_dc, decimate_fc, movie_directory)

# ====================
# %% Select reference station and streams for FK analysis

# Select stations for plotting cross-section
if plot_Xsec:
    from GMV_utils import select_Xsec
    Xsec_dict = select_Xsec(GMV)
    # Select ceneter station of the cross-section
    thechosenone_Xsec = station_phases(GMV, Xsec_dict["center_station"], event_dic, model, phases)
elif plot_FK4:
    thechosenones = []
    for sta in stations:
        thechosensta = station_phases(GMV, sta, event_dic, model, phases)
        thechosenones.append(thechosensta)
else:
    # Select a reference station for plotting seismogram
    thechosenone = station_phases(GMV, station, event_dic, model, phases)

# Only Vertical component for FK analysis
if plot_FK or plot_FK_ray:
    freq    = (f11, f22, f111, f222)
    FK_dict = (sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, interval_win2, win_len2, minval, maxval)
    subarray_dict = select_FK(event_dic, data_dic, start, end, model, freq, thechosenone, decimate_fc=decimate_fc, plot_CH_only=plot_CH_only)
    ## Plot array response function
    # from GMV_utils import plot_ARF
    # plot_ARF(subarray_dict, 5.)
elif plot_FK4:
    subarrays = []
    freq    = (f11, f22, f111, f222)
    FK_dict = (sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, interval_win2, win_len2, minval, maxval)
    for thechosensta in thechosenones:
        subarray = select_FK(event_dic, data_dic, start, end, model, freq, thechosensta, decimate_fc=decimate_fc)
        subarrays.append(subarray)
        subarrays.sort(key=itemgetter('center_dist'))
elif plot_FK_NM:
    freq    = (f11, f22)
    FK_dict = (sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, minval, maxval)
    # subarray_dict = select_FK(event_dic, data_dic, start, end, model, freq, thechosenone, decimate_fc=decimate_fc, radius=150., station_list=None, plot_CH_only=plot_CH_only)
    subarray_dict = select_FK(event_dic, data_dic, start, end, model, freq, thechosenone, decimate_fc=decimate_fc, radius=150., station_list=station_list, plot_CH_only=plot_CH_only)

# %% Prepare for plotting 

# AA logo
with get_sample_data(logo_loc) as file_img:
    arr_img = plt.imread(file_img, format='png')
    
# Choose movie interval and turn into index
if select_interval: # use the interval set in the parameter box
    interval = int(interval/stream_info["timestep"])
else: # default (for 1Hz interval)
    interval = int(stream_info["sample_rate"])  

# %% Main GMV plotting
      
## ====================
## FK analysis + ray path plot
if plot_FK_ray:
            
    GMV_FK_ray(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
           start_movie, end_movie, interval,
           vmin, vmax, arr_img, 
           movie_directory, timeframes=timeframes, timelabel=timelabel,
           slow_unit=slow_unit,
           plot_save=plot_save, save_option=save_option, save_dpi=save_dpi,
           plot_3c=plot_3c, plot_rotate=plot_rotate)

## ====================
## single FK analysis plot
elif plot_FK:
        
    GMV_FK(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
           start_movie, end_movie, interval,
           vmin, vmax, arr_img, 
           movie_directory, timeframes=timeframes, timelabel=timelabel,
           slow_unit=slow_unit,
           plot_save=plot_save, save_option=save_option, save_dpi=save_dpi,
           plot_3c=plot_3c, plot_rotate=plot_rotate)
    
## ====================
## single FK analysis plot
elif plot_FK4:
    
    GMV_FK4(GMV, event_dic, stream_info, thechosenones, subarrays, FK_dict,
            start_movie, end_movie, interval,
            vmin, vmax, arr_img, 
            movie_directory, timeframes=timeframes, timelabel=timelabel,
            slow_unit="sdeg",
            plot_save=plot_save, save_option=save_option, save_dpi=save_dpi,
            plot_3c=plot_3c, plot_rotate=plot_rotate)
    
## ====================
# # Cross-section plot
elif plot_Xsec:
    
    GMV_Xsec(GMV, event_dic, stream_info, Xsec_dict, thechosenone_Xsec, 
             start_movie, end_movie, interval,
             vmin, vmax, arr_img, 
             movie_directory, timeframes=timeframes, timelabel=timelabel,
             plot_save=plot_save, save_option=save_option, save_dpi=save_dpi,
             plot_3c=plot_3c)

## ====================
# # Normal mode zero crossing plot and vertical seismogram only
elif plot_Vonly or (plot_ZeroX and not plot_FK_NM):
    
    GMV_ZeroX(GMV, event_dic, stream_info, thechosenone, 
              start_movie, end_movie, interval,
              vmin, vmax, arr_img, 
              movie_directory, timeframes=timeframes, timelabel=timelabel,
              plot_save=plot_save, save_option=save_option, save_dpi=save_dpi, 
              plot_local=False, plot_Vonly=plot_Vonly)

## ====================
# # FK analysis plot for normal mode
elif plot_FK_NM:
    
     GMV_FK_NM(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
               start_movie, end_movie, interval,
               vmin, vmax, arr_img, 
               movie_directory, timeframes=timeframes, 
               slow_unit=slow_unit, timelabel=timelabel,
               plot_save=plot_save, save_option=save_option, save_dpi=save_dpi,
               plot_3c=plot_3c, plot_rotate=plot_rotate)

## ====================
## Only GMV and reference seismograms
else:
    
    # GMV_plot(GMV, event_dic, stream_info, thechosenone, 
    #          start_movie, end_movie, interval,
    #          vmin, vmax, arr_img, 
    #          movie_directory, timeframes=timeframes, timelabel=timelabel,
    #          plot_save=plot_save, save_option=save_option, save_dpi=save_dpi, 
    #          plot_local=plot_local, plot_3c=plot_3c, plot_rotate=plot_rotate)    
    GMV_plot(GMV, event_dic, stream_info, thechosenone, 
             start_movie, end_movie, interval,
             vmin, vmax, arr_img, 
             movie_directory, timeframes=timeframes, timelabel=timelabel,
             plot_save=plot_save, save_option=save_option, save_dpi=save_dpi, 
             plot_local=plot_local, plot_3c=plot_3c, plot_rotate=plot_rotate)
        
    
print("--- %.3f seconds ---" % (time.time() - prog_starttime))
print("---- %.3f mins ----" % ((time.time() - prog_starttime)/60))
