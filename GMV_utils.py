#!/usr/bin/env python3

__author__ = "Angel Ling"

import obspy
import string
import pickle
import numpy as np
from os import listdir
from obspy.geodetics.base import gps2dist_azimuth  # distance in m, azi and bazi in degree
from obspy.geodetics import kilometers2degrees     # distance in degree
from obspy.geodetics.flinnengdahl import FlinnEngdahl
from obspy.signal.rotate import rotate_ne_rt
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import simple_linear_interpolation
from matplotlib.gridspec import GridSpec
from operator import itemgetter
from obspy.signal.array_analysis import array_processing
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import MultipleLocator

# ==================================
# %% General variables and functions

KM_PER_DEG = 111.1949
cmap_fk    = plt.cm.get_cmap('hot_r') # color map for GMV

# Set up AA domain 
global AA_lat1, AA_lat2, AA_lon1, AA_lon2, box, dlat, dlon
AA_lat1 = 40. # starting latitude  40N
AA_lat2 = 52. # ending latitude    52N
AA_lon1 = 0.  # starting longitude 0E
AA_lon2 = 22. # ending longitude   22E
# Create grids on map
box = 4 # number of boxes on each side
dlat = (AA_lat2-AA_lat1)/box
dlon = (AA_lon2-AA_lon1)/box

def maxabs(x):
    return max(abs(x))

# Function to calculate the antipode
def antipode(lat, lon):
    lat *= -1
    if lon > 0:
        lon -= 180
    elif lon <= 0:
        lon += 180
    return lat, lon

# Degrees to cardinal directions
def degrees_to_cardinal(d):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]

# ==================================
# %% Event catalog 

def readevent(event_name, data_directory, local=False):
    """
    Read QUAKEML or pkl downloaded via obspyDMT
    
    :param event_name: 
    :param data_directory:
    :param local: plot local event
    :return: event dictionary
    """
    print("Reading event "+event_name+"...")
    print("Reading event catalog "+event_name+"...")
    event_dic = {}
    if not local:
        try: # QUAKEML
            cat = obspy.read_events(data_directory+"../EVENTS-INFO/catalog.ml", format="QUAKEML")
            ev  = cat[0]
            origin    = ev.preferred_origin()
            lat_event = origin.latitude
            lon_event = origin.longitude
            dep_event = origin.depth/1000.
            ev_otime  = origin.time
            time_event_sec = ev_otime.timestamp
            year   = ev_otime.year
            month  = ev_otime.month
            day    = ev_otime.day
            hour   = ev_otime.hour
            minute = ev_otime.minute
            second = float(ev_otime.second) + ev_otime.microsecond / 1E6
            
            if ev.preferred_magnitude().magnitude_type in ["mw", "mww"]:
                    mag_type  = ev.preferred_magnitude().magnitude_type
                    mag_event = ev.preferred_magnitude().mag
            else:
                for _i in ev.magnitudes:
                    if _i.magnitude_type in ["Mw", "Mww"]:
                        mag_type  = _i.magnitude_type
                        mag_event = _i.mag
                    else:
                        mag_type  = _i.magnitude_type
                        mag_event = _i.mag
        except FileNotFoundError: # Event pkl
            ev_pkl = pickle.load(open(data_directory+"info/event.pkl", "rb"))
            lat_event = ev_pkl.get('latitude')
            lon_event = ev_pkl.get('longitude')
            dep_event = ev_pkl.get('depth')
            ev_otime  = ev_pkl.get("datetime")
            time_event_sec = ev_otime.timestamp
            year   = ev_otime.year
            month  = ev_otime.month
            day    = ev_otime.day
            hour   = ev_otime.hour
            minute = ev_otime.minute
            second = float(ev_otime.second) + ev_otime.microsecond / 1E6
            
            mag_type  = ev_pkl.get("magnitude_type")
            mag_event = ev_pkl.get("magnitude")
    
    # Enter local event info by hand
    elif local:
        lat_event = -20.546#46.91
        lon_event = -175.390#9.13
        dep_event = 0.0#0.3
        ev_otime  = obspy.UTCDateTime("2022-01-15 04:14:45")#obspy.UTCDateTime("2020-10-25 19:35:43")
        time_event_sec = ev_otime.timestamp
        year   = ev_otime.year
        month  = ev_otime.month
        day    = ev_otime.day
        hour   = ev_otime.hour
        minute = ev_otime.minute
        second = float(ev_otime.second) + ev_otime.microsecond / 1E6
            
        mag_type  = "M" #"ML"
        mag_event = 5.8 #4.4
    
    fe = FlinnEngdahl()
    event_dic['event_name'] = event_name
    event_dic['region']     = fe.get_region(lon_event,lat_event)
    event_dic['lat']        = lat_event
    event_dic['lon']        = lon_event
    event_dic['depth']      = dep_event
    event_dic["ev_otime"]   = ev_otime
    event_dic['time_sec']   = time_event_sec
    event_dic['year']       = year
    event_dic['month']      = month
    event_dic['day']        = day
    event_dic['hour']       = hour
    event_dic['minute']     = minute
    event_dic['second']     = second
    event_dic['mag_type']   = mag_type
    event_dic['mag']        = mag_event
    
    return event_dic

# ==================================
# %% Read and process streams and inventories

def keep_trace(st, keep):
    newst  = obspy.Stream()
    for i, tr in enumerate(st):
        if i in keep:
            newst += tr
    return newst

def filter_streams(st, f1, f2, f=None, ftype="bandpass"):
    """
    Filter streams
    
    :param st: stream
    :param f1: lower frequency
    :param f2: higher frequency
    :param f: frequency for lowpass/highpass
    :param ftype: filter type, default: "bandpass"
    :return: filtered stream
    """
    st_filtered = st.copy()
    st_filtered.detrend("demean")
    st_filtered.detrend('linear')
    st_filtered.taper(0.05, 'cosine')
    if ftype == "bandpass":
        st_filtered.filter(ftype, freqmin=f1, freqmax=f2, corners=4, zerophase=True)
    else:
        st.filter(ftype, freq=f, corners=4, zerophase=True)
    st_filtered.detrend('linear') # detrend after filtering
    # st_filtered.decimate(2) # downsample the stream from 20 Hz to 10 Hz
    
    return st_filtered

def read_data_inventory(prodata_directory, resp_directory, event_dic, tstart=0, tend=7200, read_obs=False, plot_3c=True):
    """
    Read raw data and station inventory
    
    :param prodata_directory: data directory
    :param resp_directory: inventory directory
    :param event_dic: event dictionary
    :param read_obs: option to read OBS data
    :param plot_3c: option to plot in 3 component
    :return: dictionary with station data and inventory
    """
    print ('Reading data and inventory...')
    
    st1 = obspy.Stream() # Z channels
    st2 = obspy.Stream() # N channels
    st3 = obspy.Stream() # E channels 
    lat_sta  = np.array([])
    lon_sta  = np.array([])
    elv_sta  = np.array([])
    dist_sta = np.array([])
    azi_sta  = np.array([])
    bazi_sta = np.array([])
    name_sta = np.array([])
    net_sta  = np.array([])
    
    lat_event  = event_dic["lat"]
    lon_event  = event_dic["lon"]
    region     = event_dic["region"]
    event_time = event_dic["time_sec"] 
    stime      = obspy.UTCDateTime(event_time + tstart) 
    etime      = obspy.UTCDateTime(event_time + tend)
    data_dict_list = []
    
    obs = 0 # obs counter
    Z12_count  = 0 # Z12 station counter 
    
    print ('Processing data of '+event_dic['mag_type'].capitalize()+"%.1f "%event_dic['mag']+region+' earthquake...')
    print ('Reading trace from '+str(tstart)+' to '+str(tend)+'s from the origin time...')
    
    for file in sorted(listdir(prodata_directory)):
        if file.endswith('HHZ'):
            fsp = file.split(".")
            try:
                # HHZ
                tr11    = obspy.read(prodata_directory+file, starttime=stime, endtime=etime) # read the trace
                inv_sta = obspy.read_inventory(resp_directory+'STXML.'+file) # read inventory
                # Stations with data missing more than 10% of their samples are discarded.
                data_count = (tend-tstart)*tr11[0].stats.sampling_rate - ((tend-tstart)*tr11[0].stats.sampling_rate*0.1) 
                if tr11[0].stats.npts < data_count:
                    continue              
            except Exception as e: # If file not present, skip this station
                # print(e)
                continue
            
            # Read station location
            lat = inv_sta[0][0].latitude
            lon = inv_sta[0][0].longitude
            elv = inv_sta[0][0].elevation /1000. # in km
            dist, azi, bazi = gps2dist_azimuth(lat_event, lon_event, lat, lon)
            dist_deg        = kilometers2degrees(dist/1000.)
                
            # Ignore stations outside the plotting region
            if lat < AA_lat1-0.1 or lat > AA_lat2+0.1:
                continue
            if lon < AA_lon1-0.1 or lon > AA_lon2+0.1:
                continue
            
            tr11[0].stats.distance       = dist_deg
            tr11[0].stats.backazimuth    = bazi
            tr11[0].stats["coordinates"] = {}
            tr11[0].stats["coordinates"]["latitude"]  = lat
            tr11[0].stats["coordinates"]["longitude"] = lon
            tr11[0].stats["coordinates"]["elevation"] = elv
            
            try: # Read HHN
                tr22 = obspy.read(prodata_directory+fsp[0]+"."+fsp[1]+"."+fsp[2]+".HHN", starttime=stime, endtime=etime)              
                if tr22[0].stats.npts < data_count:
                    continue 
                tr22[0].stats.distance       = dist_deg
                tr22[0].stats.backazimuth    = bazi
                tr22[0].stats["coordinates"] = {}
                tr22[0].stats["coordinates"]["latitude"]  = lat
                tr22[0].stats["coordinates"]["longitude"] = lon
                tr22[0].stats["coordinates"]["elevation"] = elv               
            except Exception as e: # If file not present, skip this station
                # print(e)
                try: # Read HH1
                    tr22 = obspy.read(prodata_directory+fsp[0]+"."+fsp[1]+"."+fsp[2]+".HH1", starttime=stime, endtime=etime)
                    if tr22[0].stats.npts < data_count:
                        continue 
                except Exception as e:
                    if plot_3c:
                        continue
                    else:
                        # If there's no HHN channel
                        tr22 = obspy.Trace(np.zeros(len(tr11[0].data)))
                        tr22.stats.network  = tr11[0].stats.network
                        tr22.stats.station  = tr11[0].stats.station
                        tr22.stats.location = tr11[0].stats.location 
                        tr22.stats.delta    = tr11[0].stats.delta
                        tr22.stats.starttime= tr11[0].stats.starttime
                        tr22.stats.channel  = "HHN"
                        tr22.stats.distance    = dist_deg
                        tr22.stats.backazimuth = bazi
                        tr22.stats["coordinates"] = {}
                        tr22.stats["coordinates"]["latitude"]  = lat
                        tr22.stats["coordinates"]["longitude"] = lon
                        tr22.stats["coordinates"]["elevation"] = elv
                        tr22 = obspy.Stream(traces=[tr22])
                        
            try: # Read HHE
                tr33 = obspy.read(prodata_directory+fsp[0]+"."+fsp[1]+"."+fsp[2]+".HHE", starttime=stime, endtime=etime)
                if tr33[0].stats.npts < data_count:
                    continue                 
                tr33[0].stats.distance    = dist_deg
                tr33[0].stats.backazimuth = bazi
                tr33[0].stats["coordinates"] = {}
                tr33[0].stats["coordinates"]["latitude"]  = lat
                tr33[0].stats["coordinates"]["longitude"] = lon
                tr33[0].stats["coordinates"]["elevation"] = elv            
            except Exception as e:
                try: # Read HH2
                    tr33 = obspy.read(prodata_directory+fsp[0]+"."+fsp[1]+"."+fsp[2]+".HH2", starttime=stime, endtime=etime)
                    if tr33[0].stats.npts < data_count:
                        continue 
                except Exception as e: # If file not present, skip this station
                    # print(e)
                    if plot_3c:
                        continue
                    else:
                        # If there's no HHE channel
                        tr33 = obspy.Trace(np.zeros(len(tr11[0].data)))
                        tr33.stats.network  = tr11[0].stats.network
                        tr33.stats.station  = tr11[0].stats.station
                        tr33.stats.location = tr11[0].stats.location 
                        tr33.stats.delta    = tr11[0].stats.delta
                        tr33.stats.starttime= tr11[0].stats.starttime
                        tr33.stats.channel     = "HHE"
                        tr33.stats.distance    = dist_deg
                        tr33.stats.backazimuth = bazi
                        tr33.stats["coordinates"] = {}
                        tr33.stats["coordinates"]["latitude"]  = lat
                        tr33.stats["coordinates"]["longitude"] = lon
                        tr33.stats["coordinates"]["elevation"] = elv
                        tr33 = obspy.Stream(traces=[tr33])
                    
            if tr22[0].stats.channel != "HHN" or tr33[0].stats.channel != "HHE":
                trZ12 = tr11+tr22+tr33
                try:
                    invZNE = obspy.read_inventory(resp_directory+'STXML.'+fsp[0]+"."+fsp[1]+"."+fsp[2]+"*")
                    trZNE  = trZ12.copy()._rotate_to_zne(invZNE, components=('Z12'))   # rotate to ZNE
                except Exception:
                    continue
                for tr in trZNE:
                    if tr.stats.channel.endswith('Z'):
                        tr1 = tr.copy()
                        tr1.stats.distance       = dist_deg
                        tr1.stats.backazimuth    = bazi
                        tr1.stats["coordinates"] = {}
                        tr1.stats["coordinates"]["latitude"]  = lat
                        tr1.stats["coordinates"]["longitude"] = lon
                        tr1.stats["coordinates"]["elevation"] = elv
                    elif tr.stats.channel.endswith('N'):
                        tr2 = tr.copy()
                        tr2.stats.distance       = dist_deg
                        tr2.stats.backazimuth    = bazi
                        tr2.stats["coordinates"] = {}
                        tr2.stats["coordinates"]["latitude"]  = lat
                        tr2.stats["coordinates"]["longitude"] = lon
                        tr2.stats["coordinates"]["elevation"] = elv
                    elif tr.stats.channel.endswith('E'):
                        tr3 = tr.copy()
                        tr3.stats.distance       = dist_deg
                        tr3.stats.backazimuth    = bazi
                        tr3.stats["coordinates"] = {}
                        tr3.stats["coordinates"]["latitude"]  = lat
                        tr3.stats["coordinates"]["longitude"] = lon
                        tr3.stats["coordinates"]["elevation"] = elv
                Z12_count += 1
            else:
                tr1 = tr11
                tr2 = tr22
                tr3 = tr33
                
        # Read all OBS channels
        elif file.endswith('CHZ'):
            if not read_obs: # If False, don't read OBS stations
                continue 
            
            fsp = file.split(".")
            try:
                file_ch = fsp[0]+"."+fsp[1]+"."+fsp[2]+".CH*"
                ch      = obspy.read(prodata_directory+file_ch, starttime=stime, endtime=etime) # read as a stream
                inv_sta = obspy.read_inventory(resp_directory+'STXML.'+file_ch)   # read all channel inventory
                ch_zne  = ch.copy()._rotate_to_zne(inv_sta, components=('Z12'))   # rotate to ZNE
                
                # Read station location
                lat = inv_sta[0][0].latitude
                lon = inv_sta[0][0].longitude
                elv = inv_sta[0][0].elevation /1000. # in km
                dist, azi, bazi = gps2dist_azimuth(lat_event, lon_event, lat, lon)
                dist_deg        = kilometers2degrees(dist/1000.) # in km
                
                # Assign rotated channels back to separate trace
                for tr in ch_zne:
                    if tr.stats.channel.endswith('Z'):
                        tr1 = tr.copy()
                        tr1.stats.channel  = "CHZ"
                        tr1.stats["coordinates"] = {}
                        tr1.stats["coordinates"]["latitude"]  = lat
                        tr1.stats["coordinates"]["longitude"] = lon
                        tr1.stats["coordinates"]["elevation"] = elv
                    elif tr.stats.channel.endswith('N'):
                        tr2 = tr.copy()
                        tr2.stats.channel  = "CHN"
                        tr2.stats["coordinates"] = {}
                        tr2.stats["coordinates"]["latitude"]  = lat
                        tr2.stats["coordinates"]["longitude"] = lon
                        tr2.stats["coordinates"]["elevation"] = elv
                    elif tr.stats.channel.endswith('E'):
                        tr3 = tr.copy()
                        tr3.stats.channel  = "CHE"
                        tr3.stats["coordinates"] = {}
                        tr3.stats["coordinates"]["latitude"]  = lat
                        tr3.stats["coordinates"]["longitude"] = lon
                        tr3.stats["coordinates"]["elevation"] = elv
                        
                obs += 1 # Count obs stations
                    
            except Exception: # If file not present, skip this station
                continue
        
        else:
            continue

        # Remove stations below 41N and beyond 52N
        if lat <= 41. or lat > 52.:
            continue
            
        # Array list
        arraydic = {}
        arraydic["net_sta"]  = fsp[0]
        arraydic["name_sta"] = fsp[0]+"."+fsp[1]+"."+fsp[2]
        arraydic["lat_sta"]  = lat
        arraydic["lon_sta"]  = lon
        arraydic["elv_sta"]  = elv
        arraydic["dist_sta"] = dist_deg
        arraydic["bazi_sta"] = bazi
        arraydic["azi_sta"]  = azi
        arraydic["tr"]       = tr1
        arraydic["tr_N"]     = tr2
        arraydic["tr_E"]     = tr3
        data_dict_list.append(arraydic)
            
    # Show number of OBS
    if obs != 0:
        print ('Read ' + str(obs)+ ' OBS stations.')
    if Z12_count != 0:
        print ('Read ' + str(Z12_count)+ ' Z12 stations.')
    # Sort the array list by distance
    print ('Sorting the list of dictionaries according to distance in degree...')
    data_dict_list.sort(key=itemgetter('dist_sta'))
    
    for i in range(len(data_dict_list)):
        # Store data in arrays
        st1 += data_dict_list[i]["tr"].copy()
        st2 += data_dict_list[i]["tr_N"].copy()
        st3 += data_dict_list[i]["tr_E"].copy()
        lat_sta  = np.append(lat_sta,  data_dict_list[i]["lat_sta"])
        lon_sta  = np.append(lon_sta,  data_dict_list[i]["lon_sta"])
        elv_sta  = np.append(elv_sta,  data_dict_list[i]["elv_sta"])
        dist_sta = np.append(dist_sta, data_dict_list[i]["dist_sta"])
        bazi_sta = np.append(bazi_sta, data_dict_list[i]["bazi_sta"])
        azi_sta  = np.append(azi_sta,  data_dict_list[i]["azi_sta"])
        net_sta  = np.append(net_sta,  data_dict_list[i]["net_sta"])
        name_sta = np.append(name_sta, data_dict_list[i]["name_sta"])

    # Storing raw data
    print ('Storing raw data...')
    data_inv_dic = {}
    # inv
    data_inv_dic["net_sta"]  = net_sta
    data_inv_dic["name_sta"] = name_sta
    data_inv_dic["lat_sta"]  = lat_sta
    data_inv_dic["lon_sta"]  = lon_sta
    data_inv_dic["elv_sta"]  = elv_sta
    data_inv_dic["dist_sta"] = dist_sta
    data_inv_dic["bazi_sta"] = bazi_sta
    data_inv_dic["azi_sta"]  = azi_sta
    # streams
    data_inv_dic["st"]   = st1
    data_inv_dic["st_N"] = st2
    data_inv_dic["st_E"] = st3
    
    return data_inv_dic

def Normalize(data_inv_dic, event_dic, f1, f2, start, end, decimate_fc=2, threshold=None, plot_Xsec=False, plot_ZeroX=False): 
    """
    Filter and normalize streams for one single event
    
    :param data_inv_dic: data and inventory dictionary
    :param event_dic: event dictionary
    :param f1: lower frequency
    :param f2: higher frequency
    :param start: start time of the stream
    :param end: end time of the stream
    :param plot_Xsec: option for plotting cross-sections
    :return: dictionary of processed data and dictionary of stream info
    """
    gmv1 = np.array([]) # Z
    gmv2 = np.array([]) # N
    gmv3 = np.array([]) # E
    gmv4 = np.array([]) # R
    gmv5 = np.array([]) # T
    tr1_std = np.array([])
    tr2_std = np.array([])
    tr3_std = np.array([])
    
    st1 = data_inv_dic["st"]
    st2 = data_inv_dic["st_N"]
    st3 = data_inv_dic["st_E"]
    lat_sta  = data_inv_dic["lat_sta"]
    lon_sta  = data_inv_dic["lon_sta"]
    name_sta = data_inv_dic["name_sta"]
    dist_sta = data_inv_dic["dist_sta"]
    bazi_sta = data_inv_dic["bazi_sta"]
    azi_sta  = data_inv_dic["azi_sta"]
    region   = event_dic['region']
    time_event_sec   = event_dic['time_sec']
    mag = event_dic['mag']
    
    # Starting and ending time after OT
    if  type(start) is obspy.core.utcdatetime.UTCDateTime and type(end) is obspy.core.utcdatetime.UTCDateTime:
        starttime = start
        endtime   = end
    else:
        starttime = obspy.UTCDateTime(time_event_sec + start) 
        endtime   = obspy.UTCDateTime(time_event_sec + end)
    
    # Filter and downsample streams
    print ('Filtering between %.2f s and %.2f s...' % (1/f2, 1/f1))
    st_all_raw = st1+st2+st3
    st_all_f   = filter_streams(st_all_raw, f1, f2)
    print(st_all_f)
    
    print ('Trimming traces from '+str(start)+'s to '+str(end)+'s...')
    # Trim the filtered traces
    st_all = st_all_f.trim(starttime, endtime, pad=True, fill_value=0)
    # Downsample data by an integer factor
    if decimate_fc:
        print("Traces will be downsampled from "+str(st_all[0].stats.sampling_rate)+"Hz to "+str(st_all[0].stats.sampling_rate/decimate_fc)+"Hz")
        st_all.decimate(decimate_fc) 
    else:
        st_all = st_all
    
    st   = st_all[:len(st1)]
    st_N = st_all[len(st1):len(st1)+len(st2)]
    st_E = st_all[len(st1)+len(st2):len(st1)+len(st2)+len(st3)]
    
    timestep    = st[0].stats.delta # Sample distance in seconds (timestep)
    nt          = st[0].stats.npts  # Total number of samples
    sample_rate = st[0].stats.sampling_rate # Sampling rate in Hz
    time_st     = st[0].times()     # stream time
    
    # Normalize each trace by max, abs value and store in matrix
    print ('Normalizing traces by the max, abs value...')
    for i, (tr1, tr2, tr3, bazi) in enumerate(zip(st, st_N, st_E, bazi_sta)):
        # Rotate HHN/HHE to Radial/Transverse before normalizing
        tr4,tr5 = rotate_ne_rt(tr2.copy().data, tr3.copy().data, bazi) # rotate N and E component to R and T
        if plot_Xsec:
            tr1_new = tr1.data/max(maxabs(tr1.data),maxabs(tr2.data),maxabs(tr3.data))
            tr2_new = tr2.data/max(maxabs(tr2.data),maxabs(tr3.data))
            tr3_new = tr3.data/max(maxabs(tr2.data),maxabs(tr3.data))        
            tr4_new = tr4/max(maxabs(tr4),maxabs(tr5))
            tr5_new = tr5/max(maxabs(tr4),maxabs(tr5))
        else:
            tr1_new = tr1.data/maxabs(tr1.data)
            tr2_new = tr2.data/max(maxabs(tr2.data),maxabs(tr3.data))
            tr3_new = tr3.data/max(maxabs(tr2.data),maxabs(tr3.data))
            tr4_new = tr4/maxabs(tr4)
            tr5_new = tr5/maxabs(tr5)
        
        tr1_std = np.append(tr1_std, tr1_new.std())
        tr2_std = np.append(tr2_std, (tr2.data/maxabs(tr2.data)).std())
        tr3_std = np.append(tr3_std, (tr3.data/maxabs(tr3.data)).std())
        
        gmv1 = np.append(gmv1, tr1_new)
        gmv2 = np.append(gmv2, tr2_new)
        gmv3 = np.append(gmv3, tr3_new)
        gmv4 = np.append(gmv4, tr4_new)
        gmv5 = np.append(gmv5, tr5_new)
    
    # Replace nans with zeros
    gmv1 = np.reshape(gmv1, (len(st),len(tr1_new.data)))
    gmv1[np.isnan(gmv1)] = 0
    gmv2 = np.reshape(gmv2, (len(st),len(tr1_new.data)))
    gmv2[np.isnan(gmv2)] = 0
    gmv3 = np.reshape(gmv3, (len(st),len(tr1_new.data)))
    gmv3[np.isnan(gmv3)] = 0
    gmv4 = np.reshape(gmv4, (len(st),len(tr1_new.data)))
    gmv4[np.isnan(gmv4)] = 0
    gmv5 = np.reshape(gmv5, (len(st),len(tr1_new.data)))
    gmv5[np.isnan(gmv5)] = 0
    
    # Remove noisy stations
    print ('Removing noisy stations...')
    
    # Remove traces with STD above a threshold # default: 0.3
    if threshold:
        threshold = threshold # set by hand
    else:
        if mag >= 7.0:
            threshold = 0.25 # default: 0.25
        else:
            threshold = 0.3  # default: 0.3
    goodstations = ((tr1_std <= threshold) & (tr2_std <= threshold) & (tr3_std <= threshold)) # Store good stations
    
    # Store good stations
    normalized_dic = {}
    normalized_dic["name_sta"] = name_sta[goodstations]
    normalized_dic["lat_sta"]  = lat_sta[goodstations]
    normalized_dic["lon_sta"]  = lon_sta[goodstations]
    normalized_dic["dist_sta"] = dist_sta[goodstations]
    normalized_dic["bazi_sta"] = bazi_sta[goodstations]
    normalized_dic["azi_sta"]  = azi_sta[goodstations]       
    # Store good streams
    normalized_dic["GMV_Z"]   = gmv1[goodstations]
    normalized_dic["GMV_N"]   = gmv2[goodstations]
    normalized_dic["GMV_E"]   = gmv3[goodstations]
    normalized_dic["GMV_R"]   = gmv4[goodstations]
    normalized_dic["GMV_T"]   = gmv5[goodstations]
    if plot_ZeroX: # Zero crossings
        print("Zero crossing data is saved.")
        gmv_ZeroX = gmv1[goodstations]
        gmv_ZeroX[abs(gmv_ZeroX)<=0.01] = 1 # only +/- 1%
        gmv_ZeroX[gmv_ZeroX!=1]         = 0 # turn off the rest
        normalized_dic["GMV_ZeroX"] = gmv_ZeroX
    
    print ('Data processing for one single event DONE!')
    print ('The total number of stations of '+region+' earthquake: '+str(len(name_sta[goodstations])))
    
    return normalized_dic, dict(start=start, end=end, timestep=timestep, nt=nt, sample_rate=sample_rate, region=region, time_st=time_st)

def station_phases(GMV, station, event_dic, model, phases):
    """
    Get single station info and arrivals
    
    :param GMV: processed data dictionary
    :param station: station name
    :param event_dic: event dictionary
    :param model: earth model
    :param phases: phases to plot on seismograms and ray path plot
    :return: dictionary of station info and arrivals
    """
    # Select a station by name
    name_sta  = GMV["name_sta"].tolist()
    dist_sta  = GMV["dist_sta"]
    lat_sta   = GMV["lat_sta"]
    lon_sta   = GMV["lon_sta"]
    bazi_sta  = GMV["bazi_sta"]
    dep_event = event_dic['depth']
    print("Reading station "+station+"...")
    try:
        sta_index = name_sta.index(station) # station index
    except ValueError:
        print("Station not found. Selecting a random station...")
        station = name_sta[int(len(name_sta)/2)] # Select a station by name
        sta_index = name_sta.index(station) # station index
    print("Getting "+station+" arrival time...")
    
    # Phases and their travel times
    model_ev  = TauPyModel(model=model) 
    arr = model_ev.get_ray_paths(source_depth_in_km=dep_event, distance_in_degree=dist_sta[sta_index],phase_list=phases)
    
    print("Station "+station+" read!")
    return dict(arr=arr, 
                sta_index=sta_index, 
                sta_name=station, 
                sta_dist=dist_sta[sta_index],
                sta_lat=lat_sta[sta_index],
                sta_lon=lon_sta[sta_index],
                sta_bazi=bazi_sta[sta_index])

def get_slowness(model, dist_deg_array, event_dic, phase):
    """
    Get slownesses of a particular phase across an array
    
    :param model: earth model
    :param dist_deg_array: station distances of an array
    :param event_dic: event dictionary
    :param phase: selected phase
    :return: slownesses of a particular phase
    """
    rayp_deg_array = np.array([])
    model_ev  = TauPyModel(model=model)
    dep_event = event_dic['depth']
    
    for dist_deg in dist_deg_array:
        arr_path = model_ev.get_ray_paths(source_depth_in_km=dep_event, distance_in_degree=dist_deg, phase_list=phase)
            
        for i in range(len(arr_path)):
            rayp     = arr_path[i].ray_param   # ray parameter in [s/rad]
            rayp_deg = rayp * np.pi / 180      # ray parameter in [s/deg]
            rayp_deg_array = np.append(rayp_deg_array, rayp_deg)

    return rayp_deg_array

def select_Xsec(GMV, mode="azi"):
    """
    Select stations for plotting cross-section
    
    :param GMV: Processed data dictionary
    :param mode: selecting stations based on latitude, backazimuth or azimuth, default: "azi"
    :return: cross-section dictionary
    """
    if mode == "lat":   
        lon_sta_new  = GMV["lon_sta"]
        cross_sec = (lon_sta_new >= 10.5) & (lon_sta_new <= 11.5) # stations within this longitude range
    elif mode == "bazi":
        bazi_sta_new = GMV["bazi_sta"]
        cross_sec = (bazi_sta_new >= np.median(bazi_sta_new)-0.5) & (bazi_sta_new <= np.median(bazi_sta_new)+0.5) # stations within this backazimuth range
    else:
        azi_sta_new  = GMV["azi_sta"]
        cross_sec = (azi_sta_new >= np.median(azi_sta_new)-0.5) & (azi_sta_new <= np.median(azi_sta_new)+0.5) # stations within this azimuth range
    
    if GMV["lat_sta"][cross_sec][0] > 46.:
        facenorth = True
        print("The cross-section is facing North-ish.")
    else:
        facenorth = False
        print("The cross-section is facing South-ish.")
    
    bazi_Xsec = np.median(GMV["bazi_sta"][cross_sec])
    if bazi_Xsec >= 0. and bazi_Xsec <= 180. :
        bazi_Xsec_r = bazi_Xsec + 180.
    elif bazi_Xsec > 180.:
        bazi_Xsec_r = bazi_Xsec - 180.
    
    leftDir  = degrees_to_cardinal(bazi_Xsec)
    rightDir = degrees_to_cardinal(bazi_Xsec_r)
    
    # Find the center station among the selected stations
    center_station = GMV["name_sta"][cross_sec][int(len(GMV["name_sta"][cross_sec])/2)+1]
        
    # Distance label for cross-section 
    dist_sta_Xsec   = GMV["dist_sta"][cross_sec] # Distance of selected stations
    dist_range      = np.arange(int(min(dist_sta_Xsec)-2), int(max(dist_sta_Xsec)+2), 2)
    dist_label_Xsec = []
    for i in dist_range:
        dist_label_Xsec.append(str(i)+'\N{DEGREE SIGN}')

    return dict(cross_sec=cross_sec, 
                facenorth=facenorth, 
                leftDir=leftDir, 
                rightDir=rightDir, 
                center_station=center_station,
                dist_sta_Xsec=dist_sta_Xsec,
                dist_label_Xsec=dist_label_Xsec)

def select_FK(event_dic, data_dic, start, end, model, freq, thechosenone, decimate_fc=2, radius=100., threshold=0.25, station_list=None, plot_CH_only=False):
    """
    Select stations for FK analysis (vertical channels only)
    
    :param event_dic: event dictionary
    :param data_dic: raw data and inventory dictionary
    :param start: start of stream
    :param end: end of stream
    :param model: earth model
    :param freq: f11 (min1), f22 (max1), f111 (min2), f222 (max)
    :param thechosenone: the center station of the subarray for FK-analysis
    :param plot_CH_only: use Swiss network as subarray
    :return: FK dictionary with subarray streams and info
    """
    st_new = obspy.Stream()
    baz_rec  = []
    lat_rec  = []
    lon_rec  = []
    elv_rec  = []
    dist_rec = []
    
    if  type(start) is obspy.core.utcdatetime.UTCDateTime and type(end) is obspy.core.utcdatetime.UTCDateTime:
        starttime_fk = start
        endtime_fk   = end
    else:
        starttime_fk = obspy.UTCDateTime(event_dic["time_sec"] + start) 
        endtime_fk   = obspy.UTCDateTime(event_dic["time_sec"] + end)
    
    for i in range(len(data_dic["net_sta"])):
        if plot_CH_only:
            if data_dic["net_sta"][i] == "CH":
                if (data_dic["st"][i].data/maxabs(data_dic["st"][i].data)).std() > threshold:
                    continue
                else:
                    st_new += data_dic["st"][i] # HHZ
        
        elif station_list:
            if data_dic["name_sta"][i] in station_list:
                if (data_dic["st"][i].data/maxabs(data_dic["st"][i].data)).std() > threshold:
                    continue
                else:
                    st_new += data_dic["st"][i] # HHZ
                    baz_rec.append(data_dic["bazi_sta"][i])
                    lat_rec.append(data_dic["lat_sta"][i])
                    lon_rec.append(data_dic["lon_sta"][i])
                    elv_rec.append(data_dic["elv_sta"][i])
                    dist_rec.append(data_dic["dist_sta"][i])
                
        else:
            dist_fk, azi_fk, bazi_fk = gps2dist_azimuth(thechosenone["sta_lat"], thechosenone["sta_lon"], data_dic["lat_sta"][i], data_dic["lon_sta"][i])
            dist_fk_km               = dist_fk/1000.
            if (dist_fk_km > 20. and dist_fk_km < radius) or dist_fk_km==0.: # distance from the center station
                if (data_dic["st"][i].data/maxabs(data_dic["st"][i].data)).std() > threshold:
                    continue
                else:
                    st_new += data_dic["st"][i] # HHZ
                    baz_rec.append(data_dic["bazi_sta"][i])
                    lat_rec.append(data_dic["lat_sta"][i])
                    lon_rec.append(data_dic["lon_sta"][i])
                    elv_rec.append(data_dic["elv_sta"][i])
                    dist_rec.append(data_dic["dist_sta"][i])
            else:
                continue
        
    # Get network location/coord for FK analysis
    if plot_CH_only:
        net = "CH"
        baz_rec  = data_dic["bazi_sta"][data_dic["net_sta"] == net]
        lat_rec  = data_dic["lat_sta"][data_dic["net_sta"]  == net]
        lon_rec  = data_dic["lon_sta"][data_dic["net_sta"]  == net]
        elv_rec  = data_dic["elv_sta"][data_dic["net_sta"]  == net]
        dist_rec = data_dic["dist_sta"][data_dic["net_sta"] == net]
        lats_rec = [min(lat_rec), max(lat_rec), max(lat_rec), min(lat_rec)] # lower left, upper left, upper right, lower right
        lons_rec = [min(lon_rec), min(lon_rec), max(lon_rec), max(lon_rec)]
    else:
        lats_rec = [min(lat_rec), max(lat_rec), max(lat_rec), min(lat_rec)] # lower left, upper left, upper right, lower right
        lons_rec = [min(lon_rec), min(lon_rec), max(lon_rec), max(lon_rec)]
    
    # See array slowness
    phase_test      = ["4kmps"]
    array_sl        = get_slowness(model, dist_rec, event_dic, phase_test)
    array_sl_center = get_slowness(model, np.array([thechosenone["sta_dist"]]), event_dic, phase_test)
    # print(array_sl)
    if len(array_sl) > 0:
        for ph in phase_test:
            # print("Median slowness of "+ph+" at the sub-array: %.2f" % np.median(array_sl))
            # print("Mean slowness of "+ph+" at the sub-array: %.2f" % np.mean (array_sl))
            print("The slowness of "+ph+" at the center of the sub-array: %.2f s/deg" % np.mean(array_sl_center))
    else:
        print("Selected phases are not present in this event.")
    
    if len(freq) == 2:
        f11, f22 = freq
        st_fk1 = filter_streams(st_new.copy(), f11, f22) # filter for FK analysis
        st_fk1.trim(starttime_fk, endtime_fk, pad=True, fill_value=0)
        st_fk1.decimate(decimate_fc) # Downsample data by an integer factor
        st_fk2 = obspy.Stream() # empty stream to pass in the dictionary
    else:
        f11, f22, f111, f222 = freq
        st_fk1 = filter_streams(st_new.copy(), f11, f22) # filter for FK analysis1
        st_fk1.trim(starttime_fk, endtime_fk, pad=True, fill_value=0)
        st_fk1.decimate(decimate_fc) # Downsample data by an integer factor
        st_fk2 = filter_streams(st_new.copy(), f111, f222) # filter for FK analysis2
        st_fk2.trim(starttime_fk, endtime_fk, pad=True, fill_value=0)
        st_fk2.decimate(decimate_fc) # Downsample data by an integer factor
    
        
    print('The total number of stations used for FK-analysis: ' + str(len(st_fk1)))
    
    return dict(baz_rec=baz_rec,
                elv_rec=elv_rec,
                dist_rec=dist_rec,
                lat_rec=lat_rec,
                lon_rec=lon_rec,
                lats_rec=lats_rec,
                lons_rec=lons_rec,
                center_dist=thechosenone["sta_dist"],
                st_fk1=st_fk1,
                st_fk2=st_fk2,
                starttime_fk=starttime_fk,
                endtime_fk=endtime_fk)

# ==================================
# %% Plotting functions

def plot_ARF(subarray_dict, klim):
    """
    Plot array response function
    
    :param subarray_dict: event dictionary
    :param klim: wavenumber
    """
    #set limits for wavenumber differences to analyze
    kxmin = -klim
    kxmax = klim
    kymin = -klim
    kymax = klim
    kstep = klim / 100.
    
    # compute transfer function as a function of wavenumber difference
    from obspy.signal.array_analysis import array_transff_wavenumber
    coords = []
    for (x, y, e) in zip(subarray_dict["lon_rec"], subarray_dict["lat_rec"], subarray_dict["elv_rec"]):
        coords.append([x, y, e])
    transff = array_transff_wavenumber(np.array(coords), klim, kstep, coordsys='lonlat')
    
    # plot array response function
    plt.pcolor(np.arange(kxmin, kxmax + kstep * 1.1, kstep) - kstep / 2.,
                np.arange(kymin, kymax + kstep * 1.1, kstep) - kstep / 2.,
                transff.T, cmap=plt.cm.get_cmap('hot_r'))
    
    plt.colorbar()
    plt.clim(vmin=0., vmax=0.1)
    plt.xlim(kxmin, kxmax)
    plt.xlabel("wavenumber")
    plt.ylim(kymin, kymax)
    plt.ylabel("wavenumber")
    plt.title("Array Response Function")
    plt.show()

def plot_dispersion_curve(data_dic, event_dic, station, chan, start_dc, end_dc, decimate_fc=2, save=None):
    """
    Plot dispersion curve of a single station
    
    :param data_dic: raw data and inventory dictionary
    :param event_dic: event dictionary
    :param station: station name
    :param chan: channel, "Z", "R", "T"
    :param start_dc: starting time
    :param end_dc: ending time
    :param save: figure directory, default: None, then it's not saved
    """
    from scipy.signal import hilbert
    
    # Select a station by name
    name_sta  = data_dic["name_sta"].tolist()
    print("Plotting dispersion curve for station "+station+"...")
    try:
        thechosenone = name_sta.index(station) # station index
    except ValueError:
        print("Station not found. Selecting a random station...")
        station = name_sta[int(len(name_sta)/2)] # Select a station by name
        thechosenone = name_sta.index(station) # station index
    
    fig, ax = plt.subplots(1,1, figsize=(7, 9))
    
    iplot = 0
    
    if chan == "R":
        st_select = "st_R"
    elif chan == "T":
        st_select = "st_T"
    else:
        st_select = "st"
    
    time_event_sec = event_dic['time_sec']   
    if  type(start_dc) is obspy.core.utcdatetime.UTCDateTime and type(end_dc) is obspy.core.utcdatetime.UTCDateTime:
        starttime_dc = start_dc
        endtime_dc   = end_dc
    else:
        starttime_dc   = obspy.UTCDateTime(time_event_sec + start_dc) 
        endtime_dc     = obspy.UTCDateTime(time_event_sec + end_dc)
    
    st = data_dic[st_select][thechosenone]
    st_work = st.copy()
    st_work.detrend()
    st_work.trim(starttime_dc, endtime_dc, pad=True, fill_value=0)
    st_work.decimate(decimate_fc) # Downsample data by an integer factor
    data = (st_work.data / maxabs(st_work.data)) * 0.6
    ax.plot(st_work.times()+ start_dc, data, 'k', lw=0.65, label=name_sta[thechosenone]+chan)
    ax.text(start_dc+50, iplot + 0.25, 'Broadband', verticalalignment='center', horizontalalignment='left')
        
    for pcenter in 2**np.arange(3, 8, 0.25):
        iplot += 1
        st_work_filter = st.copy()
        st_work_filter.detrend()
        freqmin=1./(pcenter) / np.sqrt(2) 
        freqmax=1./(pcenter) * np.sqrt(2)
        st_work_filter.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True)
        st_work_filter.trim(starttime_dc, endtime_dc, pad=True, fill_value=0)
        st_work_filter.decimate(decimate_fc) # Downsample data by an integer factor
        data_filter = (st_work_filter.data/maxabs(st_work_filter.data)) * 0.5
        ax.plot(st_work_filter.times()+ start_dc, data_filter + iplot, 'darkgrey')
        ax.plot(st_work_filter.times()+ start_dc, abs(hilbert(data_filter)) + iplot, 'red')
        ax.text(start_dc+50, iplot + 0.25, '%d s ' % pcenter, verticalalignment='center', horizontalalignment='left')
            
    ax.text(start_dc+40, iplot + 0.75, 'Period', verticalalignment='center', horizontalalignment='left')
    ax.set_xlim(start_dc, end_dc)
    ax.set_ylim(-1, iplot+1.2)
    ax.set_xticks(np.arange(start_dc, end_dc+1, 500), minor=False)
    ax.set_xticks(np.arange(start_dc, end_dc+1, 100), minor=True)
    ax.set_yticks(())
    ax.set_xlabel('Time after origin')    
    ax.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                 "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                 ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                 '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                 ', Distance '+ "%.1f"% data_dic["dist_sta"][thechosenone]+'\N{DEGREE SIGN}')
    
    ax.legend(loc="lower left", fontsize=8)
    plt.show()
    
    if save:
        plt.savefig(save + event_dic["event_name"] + "_" + station + "_DispersionCurve.pdf", dpi=150)

def phase_marker(arr, ax, channel, start, end, timelabel="s", move=False, plot_local=False, move_label=["Sg","PcP","PKP","PKiKP","SKS"], ignore=['PPP','SKKKS','SKSP','PPPS','SSP','PKiKP']):
    """
    Plot phase marker of the selected station on seismograms
    
    :param arr: list of arrivals
    :param ax: axes to plot in. 
    :param channel: channel, "Z", "N", "E", "R", "T"
    :param start: starting time of stream
    :param end: ending time of stream
    :param move: label adjustment
    :param plot_local: label adjustment if plot local
    :param ignore: these phases will not be plotted on seismograms but on ray path plot
    """
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    j = 1 # Rayleigh wave
    k = 1 # Love wave
    if move:
        n = 0.5
    else:
        n = 1
    repeat = []
    for i in range(len(arr)):
        phase_label = arr[i].name
        if phase_label in ignore: # Do not plot these phases
           continue
        if channel in ["Z","R","E","N"]:
            if phase_label == "4kmps": # Rayleigh wave
                phase_label = "R"+str(j)
                j+=1
            elif phase_label in ["4.4kmps","4.5kmps"]: # Skip Love wave
                continue
        elif channel in ["T"]:
            if phase_label in ["4.4kmps","4.5kmps"]: # Love wave
                 phase_label = "G"+str(k)
                 k+=1
            elif phase_label == "4kmps" or phase_label.startswith("P"): # Skip all P phases and Rayleigh wave
                continue
            # elif phase_label == "SKKS":
            #     continue
        else:
            print("Please specify the channel: Z, E, N, R or T")
            break
        phase_time = arr[i].time
        
        if timelabel == "hr":
            phase_time = phase_time/(60*60)
        elif timelabel == "min":
            phase_time = phase_time/60
        else:
            phase_time = phase_time
        if phase_time >= end or phase_time <= start:
            continue
        
        ax.axvline(x=phase_time, color="b", linewidth=0.8)
        
        # phase text label adjustment

        if timelabel == "hr":
            x_adjust = 0.005
        elif timelabel == "min":
            x_adjust = 0.2
        else:
            if plot_local:
                x_adjust = 3
            else:
                x_adjust = 12
            
        if phase_label in repeat or phase_label in move_label:
            ax.text(x=phase_time+x_adjust, y=-0.9*n, s=phase_label, color="b", fontsize=8, bbox=props) # plot label on the bottom
        else:
            ax.text(x=phase_time+x_adjust, y= 0.75*n, s=phase_label, color="b", fontsize=8, bbox=props)
        repeat.append(phase_label)
    
import matplotlib.text
class _SmartPolarText(matplotlib.text.Text):
    """
    Automatically align text on polar plots to be away from axes.

    This class automatically sets the horizontal and vertical alignments
    based on which sides of the spherical axes the text is located.
    """
    def draw(self, renderer, *args, **kwargs):
        fig = self.get_figure()
        midx = fig.get_figwidth() * fig.dpi / 2
        midy = fig.get_figheight() * fig.dpi / 2

        extent = self.get_window_extent(renderer, dpi=fig.dpi)
        points = extent.get_points()

        is_left = points[0, 0] < midx
        is_top = points[0, 1] > midy
        updated = False

        ha = 'right' if is_left else 'left'
        if self.get_horizontalalignment() != ha:
            self.set_horizontalalignment(ha)
            updated = True
        va = 'bottom' if is_top else 'top'
        if self.get_verticalalignment() != va:
            self.set_verticalalignment(va)
            updated = True

        if updated:
            self.update_bbox_position_size(renderer)

        matplotlib.text.Text.draw(self, renderer, *args, **kwargs)

def plot_arr_rays(arr, start, end, it, legend=False, ax=None, fig=None):
    """
    Plot ray paths of selected phases
    
    :param arr: list of arrivals
    :param start: starting time of stream
    :param end: ending time of stream
    :param it: time index
    :param legend: if boolean, specify whether or not to show the legend (at the default location.)
    :param ax: axes to plot in. 
    :param fig: fgiures to plot in. 
    """
    import copy
    if not ax:
        ax = fig.add_subplot(1, 1, 1, polar=True)
    elif not fig and not ax:
        fig, ax = plt.subplots()

    arrivals = []
    requested_phase_names = []
    for arrival in arr:
        if arrival.path is None:
            continue
        dist = arrival.purist_distance % 360.0
        distance = arrival.distance
        if distance < 0:
            distance = (distance % 360)
        if abs(dist - distance) / dist > 1E-5:
            # Mirror on axis.
            arrival = copy.deepcopy(arrival)
            arrival.path["dist"] *= -1.0
        arrivals.append(arrival)
        requested_phase_names.append(arrival.name)

        if not arrivals:
            raise ValueError("Can only plot arrivals with calculated ray paths.")

    # get the velocity discontinuities in your model for plotting:
    discons = arr.model.s_mod.v_mod.get_discontinuity_depths()
    desire  = (discons == 0) | (discons>600)
    discons = discons[desire] # Plot only desire discontinuities (the surface and below 600km)
    
    # Set polar projection
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])

    intp   = simple_linear_interpolation
    radius = arr.model.radius_of_planet
    
    # Set ray color
    # colors = plt.cm.get_cmap('jet_r')(np.linspace(0, 1.0, len(arr)))
    colormap = plt.cm.get_cmap('Paired', lut=12)
    COLORS = ['#%02x%02x%02x' % tuple(int(col * 255) for col in colormap(i)[:3]) for i in range(12)]
    COLORS = COLORS[1:][::2][:-1] + COLORS[::2][:-1]
    requested_phase_name_map = {}
    i = 0
    for phase_name in requested_phase_names:
        if phase_name in requested_phase_name_map:
            continue
        requested_phase_name_map[phase_name] = i
        i += 1
    phase_names_encountered = {ray.name for ray in arr}
    colors = {name: COLORS[i % len(COLORS)] for name, i in requested_phase_name_map.items()}
    ii = len(colors)
    for name in sorted(phase_names_encountered):
        if name in colors:
            continue
        colors[name] = COLORS[ii % len(COLORS)]
        ii += 1
        
    # Plot ray paths
    station_radius = radius - arrivals[0].receiver_depth # station location
    jj = 1 # Rayleigh wave
    kk = 1 # Love wave
    repeat = []
    for ray in arrivals:
        phase_label = ray.name
        color = colors.get(phase_label, 'k')
        phase_time = ray.time
        if phase_time >= end or phase_time <= start:
            continue
        # Rename R and L
        if phase_label == "4kmps": # Rayleigh wave
            phase_label = "R"+str(jj)
            jj+=1
        elif phase_label in ["4.4kmps","4.5kmps"]: # Love wave
            phase_label = "G"+str(kk)
            kk+=1
        
        # Requires interpolation; otherwise, diffracted phases look funny.
        if it+start >= int(phase_time-90) and it+start <= int(phase_time+90): # plot phase on ray path plot within +/-90s around the arrival time       
            ax.plot(intp(ray.path["dist"], 100),
                    radius - intp(ray.path["depth"], 100),
                    color=color, label=phase_label, lw=2.2)
            repeat.append(phase_label) 
        else:
            ax.plot(intp(ray.path["dist"], 100),
                radius - intp(ray.path["depth"], 100),
                color=color, label=phase_label, lw=1., alpha=0.2)
        
        ax.set_yticks(radius - discons)
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        
    if len(repeat)>0:
        phase_labels = ','.join(sorted(set(repeat)))
        tx = _SmartPolarText(np.deg2rad(distance),
                    station_radius + radius * 0.15,
                    phase_labels, clip_on=False)
                
        ax.add_artist(tx)

    # Pretty earthquake marker.
    ax.plot([0], [radius - arrivals[0].source_depth],
            marker="*", color="#FEF215", markersize=16, zorder=10,
            markeredgewidth=1.5, markeredgecolor="0.3",
            clip_on=False)

    # Pretty station marker.
    arrowprops = dict(arrowstyle='-|>,head_length=0.7,'
                      'head_width=0.3',
                      color='#C95241', lw=1.5)
    ax.annotate('',
                xy=(np.deg2rad(distance), station_radius),
                xycoords='data',
                xytext=(np.deg2rad(distance),
                        station_radius + radius * 0.02),
                textcoords='data',
                arrowprops=arrowprops,
                clip_on=False)
    arrowprops = dict(arrowstyle='-|>,head_length=0.8,'
                      'head_width=0.4',
                      color='0.3', lw=1.5, fill=False)
    ax.annotate('',
                xy=(np.deg2rad(distance), station_radius),
                xycoords='data',
                xytext=(np.deg2rad(distance),
                        station_radius + radius * 0.01),
                textcoords='data',
                arrowprops=arrowprops,
                clip_on=False)
    
    # Arrow to show wave direction 
    ax.annotate("", xy=(0.5, station_radius+225), 
                xytext=(0.1, station_radius+190), 
                arrowprops=dict(arrowstyle="->,head_length=0.6, "
                                "head_width=0.3",
                                color="green",
                                connectionstyle="arc3,rad=-0.2",lw=2.5
                                ), annotation_clip=False
                )
    
    ax.annotate("", xy=(-0.5, station_radius+225), 
                xytext=(-0.1, station_radius+190), 
                arrowprops=dict(arrowstyle="->,head_length=0.6, "
                                "head_width=0.3",
                                color="orange",
                                connectionstyle="arc3,rad=0.2",lw=2.5
                                ), annotation_clip=False
                )
    
    ax.set_rmax(radius)
    ax.set_rmin(0.0)

    if legend:
        if isinstance(legend, bool):
            if 0 <= distance <= 180.0:
                loc = "upper left"
            else:
                loc = "upper right"
        else:
            loc = legend
        ax.legend(loc=loc, prop=dict(size="small"))
    
    plt.tight_layout()

    plt.show()

def draw_screen_poly(lats, lons, m, ax, c='magenta'):
    """
    Draw the subarray box on map
    
    :param lats: list latitudes of the subarray
    :param lons: lons latitudes of the subarray
    :param m: basemap
    :param ax: axes to plot in. 
    :param c: color of the box
    """
    from matplotlib.patches import Polygon
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon(list(xy), edgecolor=c, facecolor="None", linewidth=4)
    # plt.gca().add_patch(poly)
    ax.add_patch(poly)
    
def mcolorbar(mappable, vmin, vmax, title=None):
    """
    Fit the color bar to the map
    
    :param mappable: basemap
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param title: colorbar title
    :return: colorbar
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax  = divider.append_axes("right", size="4%", pad=0.05)
    ticks = [vmin*0.95, 0, vmax*0.95]
    cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cbar.ax.set_yticklabels(["Down","0","Up"], fontsize=12)
    cbar.ax.tick_params(size=0)
    
    if title: # colorbar title
        cbar.ax.set_ylabel(title, fontsize=11, rotation=270, labelpad=14)

    plt.sca(last_axes)
    return cbar

# %% GMV functions

# GMV and reference seismograms only
def GMV_plot(GMV, event_dic, stream_info, thechosenone, 
             start_movie, end_movie, interval,
             vmin, vmax, arr_img, 
             movie_directory, timeframes=None, timelabel="s",
             plot_save=False, save_option="png", save_dpi=120,
             plot_local=False, plot_3c=True, plot_rotate=True):
    
    """
    Plot single timestep of GMV and reference seismograms 
    
    :param GMV: processed data dictionary
    :param event_dic: event dictionary
    :param stream_info: stream info dictionary
    :param thechosenone: the reference station info dictionary
    :param start_movie: starting time of the movie (in s)
    :param end_movie: ending time of the movie (in s)
    :param interval: movie interval (index)
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param arr_img: AlpArray logo
    :param movie_directory: movie directory path
    :param plot_save: If True, save figure
    :param plot_local: If True, plot local event
    :param plot_3c: If True, plot in 3 component on GMV
    :param plot_rotate: If True, seismograms plotted in RT instead of NE
    """
    
    # Variables
    start    = stream_info["start"]
    end      = stream_info["end"]
    timestep = stream_info["timestep"]
    time_st  = stream_info["time_st"]
    lat_sta_new  = GMV["lat_sta"]
    lon_sta_new  = GMV["lon_sta"]
    name_sta_new = GMV["name_sta"]
    
    start_movie_index = int((start_movie-start)/timestep)
    end_movie_index   = int((end_movie-start)/timestep)
    
    if timelabel == "hr":
        d_time        = 60*60
        seismo_labels = np.arange(round(start/d_time *2 +0.5)/2, round(end/d_time *2 +0.5)/2, 0.5)
        print("Seismograms will be plotted in HOURS.")
    elif timelabel == "min":
        d_time        = 60
        seismo_labels = np.arange(round(round(start/d_time*2 +1)/2, -1), round(round(end/d_time*2 +1)/2,-1), 15)
        print("Seismograms will be plotted in MINUTES.")
    else:
        d_time = 1
        if plot_local:
            seismo_labels = np.arange(int(start), int(end)+1, 100)
        else:
            seismo_labels = np.arange(round(start+1, -3), round(end, -3)+1, 1000)
        print("Seismograms will be plotted in SECONDS.")
    
    if timeframes is not None:
        print("The following time frames in seconds will be plotted:")
        print(*timeframes)
        timeframes = ((np.array(timeframes)-start)/timestep).astype(int)
    else:
        timeframes = range(start_movie_index, end_movie_index+1, interval)
        # Start and end of movies in seconds
        if timelabel == "hr":
            print("Movie will start at %02.1f"%(start_movie/d_time)+" hr ("+str(start_movie)+"s) after OT and end at %02.1f"%(end_movie/d_time)+" hr ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        elif timelabel == "min":
            print("Movie will start at %04.1f"%(start_movie/d_time)+" min ("+str(start_movie)+"s) after OT and end at %04.1f"%(end_movie/d_time)+" min ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        else:
            print("Movie will start at "+str(start_movie)+"s after OT and end at "+str(end_movie)+"s after OT with interval "+str(interval*timestep)+"s.")
        
    if plot_3c:
        print("Ready to plot 3C figures!")
    else:
        print("Ready to plot (vertical component only)!")  
        
    if plot_save:
        print("Plots will be saved in " + movie_directory)
    else:
        print("Plots will not be saved.")
        
    step = 0 # used only when plot_local is True
    for it in timeframes: # from start time to end time 
        if it == start_movie_index or (start+it*timestep) % 500 == 0:
            print("Plotting %07.1f s..."%(start+it*timestep))
        # Just the simple map and seismograms
        # Setting up the plot
        fig = plt.figure(figsize=(10, 8.5)) 
        gs=GridSpec(5,1, height_ratios=[4,0.05,0.55,0.55,0.55])
        gs.update(hspace=0.1)
        ax1 = plt.subplot(gs[0])
        axs = plt.subplot(gs[1])
        axs.set_visible(False)
        ax2 = plt.subplot(gs[2])
        ax3 = plt.subplot(gs[3])
        ax4 = plt.subplot(gs[4])
        
        # Setting up the map
        m = Basemap(projection='mill',llcrnrlat=AA_lat1,urcrnrlat=AA_lat2,llcrnrlon=AA_lon1,urcrnrlon=AA_lon2,resolution="i",ax=ax1)
        # m.shadedrelief()
        # m.etopo()
        m.drawcoastlines()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightyellow',lake_color='lightblue')
        m.drawcountries(color="lightgrey")
        
        # Draw parallels and meridians.
        parallels = np.arange(AA_lat1,AA_lat2+dlat,dlat)
        # Label the meridians and parallels
        m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        # Draw Meridians and Labels
        meridians = np.arange(AA_lon1,AA_lon2+dlon,dlon)
        m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        
        # Plot the stations
        # Plot 3C motion
        if plot_3c:
            if plot_local:
                scale = 0.65
            else:
                scale = 1
            x_sta, y_sta = m(lon_sta_new+(GMV["GMV_E"][:,it]*scale), lat_sta_new+(GMV["GMV_N"][:,it]*scale))
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot vertical motion only
        else:
            x_sta, y_sta = m(lon_sta_new, lat_sta_new)
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot reference station (Plotting it again to avoid being covered by other dots)
        alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_Z"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
        alpmap_n = m.plot(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], fillstyle='none', markeredgecolor="lime", markeredgewidth=3, marker="o", markersize=8, zorder=5, ax=ax1)
        
        if plot_local: # Plot local epicenter
            x_eq, y_eq = m(event_dic["lon"], event_dic["lat"])
            alpmap_eq = m.plot(x_eq, y_eq, c="yellow", markeredgecolor="k", markeredgewidth=1.5, marker="*", markersize=15, zorder=4, ax=ax1)
        
        mcolorbar(alpmap, vmin, vmax) # Plot map colorbar
        
        ax1.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                      "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                      ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                      '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                      ', Distance '+ "%.1f"% np.median(GMV["dist_sta"])+'\N{DEGREE SIGN}, '+str(len(name_sta_new))+' STA', fontsize=14)
    
        # Add AA logo on plot
        imagebox = OffsetImage(arr_img, zoom=0.45)
        imagebox.image.axes = ax1
        ab = AnnotationBbox(imagebox, (1, 1),
                            xybox=(40., 280.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5, frameon=False)
        
        ax1.add_artist(ab)
        
        # Plot reference seismograms
        # HHZ
        ax2.plot( (time_st+start)/d_time, GMV["GMV_Z"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".Z")
        ax2.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        # Plot phase marker for channel Z
        phase_marker(thechosenone["arr"], ax2, "Z", start/d_time, end/d_time, timelabel=timelabel, plot_local=plot_local)
        
        ax2.grid(b=True, which='major')
        ax2.set_xlim([start/d_time,end/d_time])
        ax2.set_ylim([-1.1,1.1]) 
        ax2.set_xticks(seismo_labels)
        ax2.set_xticklabels([])
        ax2.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.07))
        
        # HHN/R
        if plot_rotate:
            ax3.plot((time_st+start)/d_time, GMV["GMV_R"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".R")
            phase_marker(thechosenone["arr"], ax3, "R", start/d_time, end/d_time, timelabel=timelabel, plot_local=plot_local)
        else:    
            ax3.plot((time_st+start)/d_time, GMV["GMV_N"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".N")
            phase_marker(thechosenone["arr"], ax3, "N", start/d_time, end/d_time, timelabel=timelabel, plot_local=plot_local)
        ax3.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        
        ax3.grid(b=True, which='major')    
        ax3.set_xlim([start/d_time,end/d_time])
        ax3.set_ylim([-1.1,1.1]) 
        ax3.set_xticks(seismo_labels)
        ax3.set_xticklabels([])
        ax3.set_ylabel('Normalized Displacement', fontsize=11)
        ax3.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.07))
        
        # HHE/T
        if plot_rotate:
            ax4.plot((time_st+start)/d_time, GMV["GMV_T"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".T")
            phase_marker(thechosenone["arr"], ax4, "T", start/d_time, end/d_time, timelabel=timelabel, plot_local=plot_local)
        else:
            ax4.plot((time_st+start)/d_time, GMV["GMV_E"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".E")
            phase_marker(thechosenone["arr"], ax4, "E", start/d_time, end/d_time, timelabel=timelabel, plot_local=plot_local)
        ax4.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        
        ax4.grid(b=True, which='major')
        ax4.tick_params(axis="x",labelsize=12)
        if timelabel == "hr":
            ax4.set_xlabel('Time after origin [hr]', fontsize=14)
            ax4.xaxis.set_minor_locator(MultipleLocator(0.1))
        elif timelabel == "min":
            ax4.set_xlabel('Time after origin [min]', fontsize=14)
        else:
            ax4.set_xlabel('Time after origin [s]', fontsize=14)
        ax4.set_xlim([start/d_time,end/d_time])
        ax4.set_ylim([-1.1,1.1]) 
        ax4.set_xticks(seismo_labels)
        ax4.set_xticklabels(seismo_labels) 
        ax4.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.07)) 
        
        plt.tight_layout(h_pad=1)
        
        if plot_save:
            if plot_local:
                step += 1
            else:
                step = start+it*timestep 
            if plot_3c:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_3C_"+ "%07.1f"%(step)+"s."+save_option, dpi=save_dpi)
            else:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_"+ "%07.1f"%(step)+"s."+save_option, dpi=save_dpi)
                
            plt.clf()
            plt.close()
            
        else:
            plt.show()
            
    if plot_save:
        print("Plots are saved. End of plotting.")
    else:
        print("Plots are not saved. End of plotting.")

# ====================
# FK analysis plot
def GMV_FK(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
           start_movie, end_movie, interval,
           vmin, vmax, arr_img, 
           movie_directory, timeframes=None,
           slow_unit="sdeg", timelabel="s",
           plot_save=False, save_option="png", save_dpi=120,
           plot_3c=True, plot_rotate=True):
    
    """
    Plot single timestep of GMV, reference seismograms, and FK diagram
    
    :param GMV: processed data dictionary
    :param event_dic: event dictionary
    :param stream_info: stream info dictionary
    :param thechosenone: the reference station info dictionary
    :param subarray_dict: subarray info dictionary
    :param FK_dict: FK analysis streams and info dictionary
    :param start_movie: starting time of the movie (in s)
    :param end_movie: ending time of the movie (in s)
    :param interval: movie interval (index)
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param arr_img: AlpArray logo
    :param movie_directory: movie directory path
    :param slow_unit: slowness unit "sdeg" or "skm", default: "sdeg"
    :param plot_save: If True, save figure
    :param plot_3c: If True, plot in 3 component on GMV
    :param plot_rotate: If True, seismograms plotted in RT instead of NE
    """
    
    # Variables
    start    = stream_info["start"]
    end      = stream_info["end"]
    timestep = stream_info["timestep"]
    time_st  = stream_info["time_st"]
    lat_sta_new  = GMV["lat_sta"]
    lon_sta_new  = GMV["lon_sta"]
    name_sta_new = GMV["name_sta"]
    st_fk1   = subarray_dict["st_fk1"]
    st_fk2   = subarray_dict["st_fk2"]
    starttime_fk = subarray_dict["starttime_fk"]
    sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, interval_win2, win_len2, minval, maxval = FK_dict
    f11, f22, f111, f222 = freq
    
    start_movie_index = int((start_movie-start)/timestep)
    end_movie_index   = int((end_movie-start)/timestep)

    if timelabel == "hr":
        d_time        = 60*60
        seismo_labels = np.arange(round(start/d_time *2 +0.5)/2, round(end/d_time *2 +0.5)/2, 0.5)
        print("Seismograms will be plotted in HOURS.")
    elif timelabel == "min":
        d_time        = 60
        seismo_labels = np.arange(round(round(start/d_time*2 +1)/2, -1), round(round(end/d_time*2 +1)/2,-1), 15)
        print("Seismograms will be plotted in MINUTES.")
    else:
        d_time = 1
        seismo_labels = np.arange(round(start+1, -3), round(end, -3)+1, 1000)
        print("Seismograms will be plotted in SECONDS.")
    
    if timeframes is not None:
        print("The following time frames in seconds will be plotted:")
        print(*timeframes)
        timeframes = ((np.array(timeframes)-start)/timestep).astype(int)
    else:
        timeframes = range(start_movie_index, end_movie_index+1, interval)
        # Start and end of movies in seconds
        if timelabel == "hr":
            print("Movie will start at %02.1f"%(start_movie/d_time)+" hr ("+str(start_movie)+"s) after OT and end at %02.1f"%(end_movie/d_time)+" hr ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        elif timelabel == "min":
            print("Movie will start at %04.1f"%(start_movie/d_time)+" min ("+str(start_movie)+"s) after OT and end at %04.1f"%(end_movie/d_time)+" min ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        else:
            print("Movie will start at "+str(start_movie)+"s after OT and end at "+str(end_movie)+"s after OT with interval "+str(interval*timestep)+"s.")
        
    if plot_3c:
        print("Ready to plot 3C figures!")
    else:
        print("Ready to plot (vertical component only)!")  
        
    if plot_save:
        print("Plots will be saved in " + movie_directory)
    else:
        print("Plots will not be saved.")
        
    print("Plotting seismograms and FK analysis...")
        
    for it in timeframes: # from start time to end time 
        if it == start_movie_index or (start+it*timestep) % 500 == 0:
            print("Plotting %06.1f s..."%(start+it*timestep))
        # Setting up the plot
        fig = plt.figure(figsize=(10, 9)) 
        gs=GridSpec(5,30, height_ratios=[3.4,0.07,0.59,0.59,0.59])
        gs.update(hspace=0.1)
        ax1 = plt.subplot(gs[0,:])
        axs = plt.subplot(gs[1,:])
        axs.set_visible(False)
        ax2 = plt.subplot(gs[2,:18])
        ax3 = plt.subplot(gs[3,:18])
        ax4 = plt.subplot(gs[4,:18])
        ax5 = plt.subplot(gs[2:,19:-2], polar=True)
        ax6 = plt.subplot(gs[2:,-1])
        
        # Array processing interval
        t_interval = it*timestep
        end_fk     = end # 3600 # ending of seismogram
        
        # Setting up the map
        m = Basemap(projection='mill',llcrnrlat=AA_lat1,urcrnrlat=AA_lat2,llcrnrlon=AA_lon1,urcrnrlon=AA_lon2,resolution="i",ax=ax1)
        m.drawcoastlines()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightyellow',lake_color='lightblue')
        m.drawcountries(color="lightgrey")
        
        # Draw parallels and meridians.
        parallels = np.arange(AA_lat1,AA_lat2+dlat,dlat)
        # Label the meridians and parallels
        m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        # Draw Meridians and Labels
        meridians = np.arange(AA_lon1,AA_lon2+dlon,dlon)
        m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        
        # Draw subarray box
        draw_screen_poly(subarray_dict["lats_rec"], subarray_dict["lons_rec"], m, ax1)
        
        # Plot the stations
        # Plot 3C motion
        if plot_3c:
            x_sta, y_sta = m(lon_sta_new+GMV["GMV_E"][:,it], lat_sta_new+GMV["GMV_N"][:,it])
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot vertical motion only
        else:
            x_sta, y_sta = m(lon_sta_new, lat_sta_new)
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot reference station (Plotting it again to avoid being covered by other dots)
        alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_Z"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
        alpmap_n = m.plot(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], fillstyle='none', markeredgecolor="lime", markeredgewidth=3, marker="o", markersize=8, zorder=5, ax=ax1)
        mcolorbar(alpmap, vmin, vmax) # Plot map colorbar
        
        ax1.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                      "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                      ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                      '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                      ', Distance '+ "%.1f"% np.median(GMV["dist_sta"])+'\N{DEGREE SIGN}, '+str(len(name_sta_new))+' STA', fontsize=14)
    
        # Add AA logo on plot
        imagebox = OffsetImage(arr_img, zoom=0.45)
        imagebox.image.axes = ax1
        ab = AnnotationBbox(imagebox, (1, 1),
                            xybox=(40., 273.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5, frameon=False)
        
        ax1.add_artist(ab)
        
        # Plot reference seismograms
        # HHZ
        ax2.plot((time_st+start)/d_time, GMV["GMV_Z"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".Z")
        ax2.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        # Plot phase marker for channel Z
        phase_marker(thechosenone["arr"], ax2, "Z", start/d_time, end_fk/d_time, timelabel=timelabel)
        # Box showing the window for array processing
        ax2.add_patch(plt.Rectangle(( (t_interval+start-(interval_win2/2) )/d_time, -1.1), interval_win2/d_time, 2.2, fc="c", alpha =0.6))
        
        ax2.grid(b=True, which='major')
        ax2.set_xticks(seismo_labels)
        ax2.set_xticklabels([])
        ax2.set_xlim([start/d_time,end_fk/d_time])
        ax2.set_ylim([-1.1,1.1]) 
        ax2.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.05))
        
        # HHN/R
        if plot_rotate:
            ax3.plot((time_st+start)/d_time, GMV["GMV_R"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".R")
            phase_marker(thechosenone["arr"], ax3, "R", start/d_time, end_fk/d_time, timelabel=timelabel)
        else:    
            ax3.plot((time_st+start)/d_time, GMV["GMV_N"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".N")
            phase_marker(thechosenone["arr"], ax3, "N", start/d_time, end_fk/d_time, timelabel=timelabel)
        ax3.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        
        ax3.grid(b=True, which='major')    
        ax3.set_xticks(seismo_labels)
        ax3.set_xticklabels([])
        ax3.set_xlim([start/d_time,end_fk/d_time])
        ax3.set_ylim([-1.1,1.1]) 
        ax3.set_ylabel('Normalized Displacement', fontsize=11)
        ax3.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.05))
        
        # HHE/T
        if plot_rotate:
            ax4.plot((time_st+start)/d_time, GMV["GMV_T"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".T")
            phase_marker(thechosenone["arr"], ax4, "T", start/d_time, end_fk/d_time, timelabel=timelabel)
        else:
            ax4.plot((time_st+start)/d_time, GMV["GMV_E"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".E")
            phase_marker(thechosenone["arr"], ax4, "E", start/d_time, end_fk/d_time, timelabel=timelabel)
        ax4.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
               
        ax4.grid(b=True, which='major')
        ax4.tick_params(axis="x",labelsize=12)
        if timelabel == "hr":
            ax4.set_xlabel('Time after origin [hr]', fontsize=14)
            ax4.xaxis.set_minor_locator(MultipleLocator(0.1))
        elif timelabel == "min":
            ax4.set_xlabel('Time after origin [min]', fontsize=14)
        else:
            ax4.set_xlabel('Time after origin [s]', fontsize=14)
        ax4.set_xlim([start/d_time,end_fk/d_time])
        ax4.set_ylim([-1.1,1.1]) 
        ax4.set_xticks(seismo_labels)
        ax4.set_xticklabels(seismo_labels) 
        ax4.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.05)) 
        
        ## Array Processing

        # Plotting             
        ax5.set_theta_direction(-1)
        ax5.set_theta_zero_location("N")
        
        # First resolution
        kwargs1 = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-sx_m, slm_x=sx_m, sll_y=-sx_m, slm_y=sx_m, sl_s=sl_s,
            # sliding window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=f11, frqhigh=f22, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9,
            stime=starttime_fk+t_interval-(interval_win/2),
            etime=starttime_fk+t_interval+(interval_win/2)
        )
        
        out1 = array_processing(st_fk1.copy(), **kwargs1) # output of array_processing
        
        # Make output human readable, adjust backazimuth to values between 0 and 360
        t1, rel_power1, abs_power1, baz1, slow1 = out1.T
        baz1[baz1 < 0.0] += 360
        
        N1 = 36
        N2 = int(sx_m/sl_s)
        abins1 = np.arange(N1 + 1) * 360. / N1   # angle resolution
        if slow_unit == "skm":
            sbins1 = np.linspace(0, sx_m, N2 + 1)      # slowness resolution s/km
        else:
            sbins1 = np.linspace(0, sx_m*100, N2 + 1)  # slowness resolution s/deg
        
        ## Sum rel power in bins given by abins and sbins
        # s/km
        if slow_unit == "skm":
            hist1, baz_edges1, sl_edges1 = \
                np.histogram2d(baz1, slow1, bins=[abins1, sbins1], weights=rel_power1)
        # s/deg
        else: 
            hist1, baz_edges1, sl_edges1 = \
                np.histogram2d(baz1, slow1*KM_PER_DEG, bins=[abins1, sbins1], weights=rel_power1)
            
        # Transform to radian
        baz_edges1 = np.radians(baz_edges1)
        dh1 = abs(sl_edges1[1] - sl_edges1[0])
        dw1 = abs(baz_edges1[1] - baz_edges1[0])
        
        # circle through backazimuth
        for i, row in enumerate(hist1):
            row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
            row_norm_log = np.log10(row_norm)
            row_norm_log /= np.log10(maxval) - np.log10(minval)
            # s/km or s/deg
            bars1 = ax5.bar(x=(i * dw1) * np.ones(N2),                
                            height=dh1 * np.ones(N2),
                            width=dw1, bottom=dh1 * np.arange(N2),
                            color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
        
        # Second resolution
        kwargs2 = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-sx, slm_x=sx, sll_y=-sy, slm_y=sy, sl_s=sl_s,
            # sliding window properties
            win_len=win_len2, win_frac=win_frac,
            # frequency properties
            frqlow=f111, frqhigh=f222, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9,
            stime=starttime_fk+t_interval-(interval_win2/2),
            etime=starttime_fk+t_interval+(interval_win2/2)
        )
                 
        out2 = array_processing(st_fk2.copy(), **kwargs2) # output of array_processing
        
        # Make output human readable, adjust backazimuth to values between 0 and 360
        t2, rel_power2, abs_power2, baz2, slow2 = out2.T
        baz2[baz2 < 0.0] += 360
        
        N1_m = 72
        N2_m = int((sx-sx_m)/sl_s)
        abins2 = np.arange(N1_m + 1) * 360. / N1_m   # angle resolution
        if slow_unit == "skm":
            sbins2 = np.linspace(sx_m, sx, N2_m + 1)         # slowness resolution s/km
        else:
            sbins2 = np.linspace(sx_m*100, sx*100, N2_m + 1) # slowness resolution s/deg
        
        # Sum rel power in bins given by abins and sbins
        # s/km
        if slow_unit == "skm":
            hist2, baz_edges2, sl_edges2 = \
                np.histogram2d(baz2, slow2, bins=[abins2, sbins2], weights=rel_power2)
        # s/deg
        else:
            hist2, baz_edges2, sl_edges2 = \
                np.histogram2d(baz2, slow2*KM_PER_DEG, bins=[abins2, sbins2], weights=rel_power2)
        
        # Transform to radian
        baz_edges2 = np.radians(baz_edges2)
        dh2 = abs(sl_edges2[1] - sl_edges2[0])
        dw2 = abs(baz_edges2[1] - baz_edges2[0])
                   
        for i, row in enumerate(hist2):
            row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
            row_norm_log = np.log10(row_norm)
            row_norm_log /= np.log10(maxval) - np.log10(minval)
            # s/km
            if slow_unit == "skm":
                bars2 = ax5.bar(x=(i * dw2) * np.ones(N2_m),                
                                height=dh2 * np.ones(N2_m) + sx_m,
                                width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m,
                                color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
            # s/deg
            else:
                bars2 = ax5.bar(x=(i * dw2) * np.ones(N2_m),                
                                height=dh2 * np.ones(N2_m) + sx_m*100,
                                width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m*100,
                                color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
            
        ax5.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax5.set_xticklabels(['N', 'E', 'S', 'W'])
                    
        # Set slowness limits
        if slow_unit == "skm":
            ax5.set_ylim(0, sx) # s/km
        else:
            ax5.set_ylim(0, sx*100) # s/deg
        [i.set_color('grey') for i in ax5.get_yticklabels()]
        
        # add a line to split two resolutions
        rads = np.arange(0, (2*np.pi), 0.01)
        if slow_unit == "skm":
            ax5.plot(rads,[sx_m]*len(rads), color="k",zorder=5, lw=0.9)      # s/km
        else:
            ax5.plot(rads,[sx_m*100]*len(rads), color="k",zorder=5, lw=0.8)  # s/deg
        
        # Plot backzimuth on FK plot
        # s/km
        if slow_unit == "skm":
            arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
            arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
            ax5.annotate('', xy=(arrow_baz, 0.5), xytext=(arrow_baz, 0.62),
                              arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
            ax5.annotate('', xy=(arrow_baz_r, 0.5), xytext=(arrow_baz_r, 0.62),
                              arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
        # s/deg
        else:
            arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
            arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
            ax5.annotate('', xy=(arrow_baz, 50), xytext=(arrow_baz, 62),
                              arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
            ax5.annotate('', xy=(arrow_baz_r, 50), xytext=(arrow_baz_r, 62),
                              arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)

        # set colorbar
        cbar = ColorbarBase(ax6, cmap=cmap_fk) 
        cbar.set_ticks([])
        cbar.set_label("Relative Power", rotation=270, labelpad=14, fontsize=11)
        
        plt.tight_layout(h_pad=1)
        
        if plot_save:
            if plot_3c:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_3C_FKanalysis_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
            else:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_FKanalysis_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
                
            plt.clf()
            plt.close()
            
        else:
            plt.show()
            
    if plot_save:
        print("Plots are saved. End of plotting.")
    else:
        print("Plots are not saved. End of plotting.")
                
# ====================
# FK analysis + ray path plot
        
def GMV_FK_ray(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
               start_movie, end_movie, interval, 
               vmin, vmax, arr_img, 
               movie_directory, timeframes=None, 
               slow_unit="sdeg", timelabel="s",
               plot_save=False, save_option="png", save_dpi=120,
               plot_3c=True, plot_rotate=True):
    
    """
    Plot single timestep of GMV, reference seismograms, ray path plot, and FK diagram
    
    :param GMV: processed data dictionary
    :param event_dic: event dictionary
    :param stream_info: stream info dictionary
    :param thechosenone: the reference station info dictionary
    :param subarray_dict: subarray info dictionary
    :param FK_dict: FK analysis streams and info dictionary
    :param start_movie: starting time of the movie (in s)
    :param end_movie: ending time of the movie (in s)
    :param interval: movie interval (index)
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param arr_img: AlpArray logo
    :param movie_directory: movie directory path
    :param slow_unit: slowness unit "sdeg" or "skm", default: "sdeg"
    :param plot_save: If True, save figure
    :param plot_3c: If True, plot in 3 component on GMV
    :param plot_rotate: If True, seismograms plotted in RT instead of NE
    """
    
    # Variables
    start    = stream_info["start"]
    end      = stream_info["end"]
    timestep = stream_info["timestep"]
    time_st  = stream_info["time_st"]
    lat_sta_new  = GMV["lat_sta"]
    lon_sta_new  = GMV["lon_sta"]
    name_sta_new = GMV["name_sta"]
    st_fk1   = subarray_dict["st_fk1"]
    st_fk2   = subarray_dict["st_fk2"]
    starttime_fk = subarray_dict["starttime_fk"]
    sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, interval_win2, win_len2, minval, maxval = FK_dict
    f11, f22, f111, f222 = freq
    lat_ant, lon_ant     = antipode(event_dic["lat"], event_dic["lon"])
    
    start_movie_index = int((start_movie-start)/timestep)
    end_movie_index   = int((end_movie-start)/timestep)

    if timelabel == "hr":
        d_time        = 60*60
        seismo_labels = np.arange(round(start/d_time *2 +0.5)/2, round(end/d_time *2 +0.5)/2, 0.5)
        print("Seismograms will be plotted in HOURS.")
    elif timelabel == "min":
        d_time        = 60
        seismo_labels = np.arange(round(round(start/d_time*2 +1)/2, -1), round(round(end/d_time*2 +1)/2,-1), 15)
        print("Seismograms will be plotted in MINUTES.")
    else:
        d_time = 1
        seismo_labels = np.arange(round(start+1, -3), round(end, -3)+1, 1000)
        print("Seismograms will be plotted in SECONDS.")
    
    if timeframes is not None:
        print("The following time frames in seconds will be plotted:")
        print(*timeframes)
        timeframes = ((np.array(timeframes)-start)/timestep).astype(int)
    else:
        timeframes = range(start_movie_index, end_movie_index+1, interval)
        # Start and end of movies in seconds
        if timelabel == "hr":
            print("Movie will start at %02.1f"%(start_movie/d_time)+" hr ("+str(start_movie)+"s) after OT and end at %02.1f"%(end_movie/d_time)+" hr ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        elif timelabel == "min":
            print("Movie will start at %04.1f"%(start_movie/d_time)+" min ("+str(start_movie)+"s) after OT and end at %04.1f"%(end_movie/d_time)+" min ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        else:
            print("Movie will start at "+str(start_movie)+"s after OT and end at "+str(end_movie)+"s after OT with interval "+str(interval*timestep)+"s.")
        
    if plot_3c:
        print("Ready to plot 3C figures!")
    else:
        print("Ready to plot (vertical component only)!")  
        
    if plot_save:
        print("Plots will be saved in " + movie_directory)
    else:
        print("Plots will not be saved.")
        
    print("Plotting seismograms, ray path, and FK analysis...")
    
    for it in timeframes: # from start time to end time 
        if it == start_movie_index or (start+it*timestep) % 500 == 0:
            print("Plotting %06.1f s..."%(start+it*timestep))
        # Setting up the plot
        fig = plt.figure(figsize=(16, 9)) 
        gs=GridSpec(7,37, height_ratios=[2.9,0.15,0.5,0.07,0.6,0.6,0.6])
        gs.update(hspace=0.1)
        ax1 = plt.subplot(gs[:3,:23])
        axs = plt.subplot(gs[3,:])
        axs.set_visible(False)
        ax2 = plt.subplot(gs[4,1:22])
        ax3 = plt.subplot(gs[5,1:22])
        ax4 = plt.subplot(gs[6,1:22])
        # FK-analysis
        ax5 = plt.subplot(gs[0,25:-2], polar=True)
        ax6 = plt.subplot(gs[0,-1]) # FK colorbar
        # Ray path plot
        ax7 = plt.subplot(gs[2:,23:], polar=True)
        
        # Array processing interval
        t_interval = it*timestep
        end_fk     = end # 3600 # ending of seismogram
        
        # Setting up the map
        m = Basemap(projection='mill',llcrnrlat=AA_lat1,urcrnrlat=AA_lat2,llcrnrlon=AA_lon1,urcrnrlon=AA_lon2,resolution="i",ax=ax1)
        m.drawcoastlines()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightyellow',lake_color='lightblue')
        m.drawcountries(color="lightgrey")
        
        # Draw parallels and meridians.
        parallels = np.arange(AA_lat1,AA_lat2+dlat,dlat)
        # Label the meridians and parallels
        m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        # Draw Meridians and Labels
        meridians = np.arange(AA_lon1,AA_lon2+dlon,dlon)
        m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        
        # Draw subarray box
        draw_screen_poly(subarray_dict["lats_rec"], subarray_dict["lons_rec"], m, ax1)
        
        # Plot the stations
        # Plot 3C motion
        if plot_3c:
            x_sta, y_sta = m(lon_sta_new+GMV["GMV_E"][:,it], lat_sta_new+GMV["GMV_N"][:,it])
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot vertical motion only
        else:
            x_sta, y_sta = m(lon_sta_new, lat_sta_new)
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot reference station (Plotting it again to avoid being covered by other dots)
        alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_Z"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
        alpmap_n = m.plot(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], fillstyle='none', markeredgecolor="lime", markeredgewidth=3, marker="o", markersize=8, zorder=5, ax=ax1)
        mcolorbar(alpmap, vmin, vmax) # Plot map colorbar
        
        # Plot great circle path
        m.drawgreatcircle(thechosenone["sta_lon"],thechosenone["sta_lat"],event_dic["lon"],event_dic["lat"],linewidth=2,color='g', zorder=5)
        m.drawgreatcircle(thechosenone["sta_lon"],thechosenone["sta_lat"],lon_ant,lat_ant,linewidth=2,color='orange', zorder=5)
                    
        ax1.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                      "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                      ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                      '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                      ', Distance '+ "%.1f"% np.median(GMV["dist_sta"])+'\N{DEGREE SIGN}, '+str(len(name_sta_new))+' STA', fontsize=14)
       
        # Add AA logo on plot
        imagebox = OffsetImage(arr_img, zoom=0.45)
        imagebox.image.axes = ax1
        ab = AnnotationBbox(imagebox, (1, 1),
                            xybox=(40., 288.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5, frameon=False)
        
        ax1.add_artist(ab)
        
        # Plot seismograms
        # HHZ
        ax2.plot( (time_st+start)/d_time, GMV["GMV_Z"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".Z")
        ax2.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        # Plot phase marker for channel Z
        phase_marker(thechosenone["arr"], ax2, "Z", start/d_time, end_fk/d_time, timelabel=timelabel)
        # Box showing the window for array processing
        ax2.add_patch(plt.Rectangle( ((t_interval+start-(interval_win2/2))/d_time, -1.1), interval_win2/d_time, 2.2, fc="c", alpha =0.6))
        
        ax2.grid(b=True, which='major')
        ax2.set_xticks(seismo_labels)
        ax2.set_xticklabels([])
        ax2.set_xlim([start/d_time,end_fk/d_time])
        ax2.set_ylim([-1.1,1.1]) 
        ax2.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.05))
        
        # HHN/R
        if plot_rotate:
            ax3.plot((time_st+start)/d_time, GMV["GMV_R"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".R")
            phase_marker(thechosenone["arr"], ax3, "R", start/d_time, end_fk/d_time, timelabel=timelabel)
        else:    
            ax3.plot((time_st+start)/d_time, GMV["GMV_N"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".N")
            phase_marker(thechosenone["arr"], ax3, "N", start/d_time, end_fk/d_time, timelabel=timelabel)
        ax3.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        
        ax3.grid(b=True, which='major')    
        ax3.set_xticks(seismo_labels)
        ax3.set_xticklabels([])
        ax3.set_xlim([start/d_time,end_fk/d_time])
        ax3.set_ylim([-1.1,1.1]) 
        ax3.set_ylabel('Normalized Displacement', fontsize=11)
        ax3.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.05))
        
        # HHE/T
        if plot_rotate:
            ax4.plot((time_st+start)/d_time, GMV["GMV_T"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".T")
            phase_marker(thechosenone["arr"], ax4, "T", start/d_time, end_fk/d_time, timelabel=timelabel)
        else:
            ax4.plot((time_st+start)/d_time, GMV["GMV_E"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".E")
            phase_marker(thechosenone["arr"], ax4, "E", start/d_time, end_fk/d_time, timelabel=timelabel)
        ax4.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        
        ax4.grid(b=True, which='major')
        ax4.tick_params(axis="x",labelsize=12)
        if timelabel == "hr":
            ax4.set_xlabel('Time after origin [hr] \n'+str(int(t_interval+start))+'s', fontsize=14)
            ax4.xaxis.set_minor_locator(MultipleLocator(0.1))
        elif timelabel == "min":
            ax4.set_xlabel('Time after origin [min] \n'+str(int(t_interval+start))+'s', fontsize=14)
        else:
            ax4.set_xlabel('Time after origin [s] \n'+str(int(t_interval+start))+'s', fontsize=14)
        ax4.set_xlim([start/d_time,end_fk/d_time])
        ax4.set_ylim([-1.1,1.1]) 
        ax4.set_xticks(seismo_labels)
        ax4.set_xticklabels(seismo_labels) 
        ax4.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.05)) 
        
        ## Array Processing

        # Plotting             
        ax5.set_theta_direction(-1)
        ax5.set_theta_zero_location("N")
        
        # First resolution
        kwargs1 = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-sx_m, slm_x=sx_m, sll_y=-sx_m, slm_y=sx_m, sl_s=sl_s,
            # sliding window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=f11, frqhigh=f22, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9,
            stime=starttime_fk+t_interval-(interval_win/2),
            etime=starttime_fk+t_interval+(interval_win/2)
        )
        
        out1 = array_processing(st_fk1.copy(), **kwargs1) # output of array_processing
        
        # Make output human readable, adjust backazimuth to values between 0 and 360
        t1, rel_power1, abs_power1, baz1, slow1 = out1.T
        baz1[baz1 < 0.0] += 360
        
        N1 = 36
        N2 = int(sx_m/sl_s)
        abins1 = np.arange(N1 + 1) * 360. / N1   # angle resolution
        if slow_unit == "skm":
            sbins1 = np.linspace(0, sx_m, N2 + 1)      # slowness resolution s/km
        else:
            sbins1 = np.linspace(0, sx_m*100, N2 + 1)  # slowness resolution s/deg
        
        # Sum rel power in bins given by abins and sbins
        # s/km
        if slow_unit == "skm":
            hist1, baz_edges1, sl_edges1 = \
                np.histogram2d(baz1, slow1, bins=[abins1, sbins1], weights=rel_power1)
        # s/deg
        else: 
            hist1, baz_edges1, sl_edges1 = \
                np.histogram2d(baz1, slow1*KM_PER_DEG, bins=[abins1, sbins1], weights=rel_power1)
            
        # Transform to radian
        baz_edges1 = np.radians(baz_edges1)
        dh1 = abs(sl_edges1[1] - sl_edges1[0])
        dw1 = abs(baz_edges1[1] - baz_edges1[0])
        
        # circle through backazimuth
        for i, row in enumerate(hist1):
            row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
            row_norm_log = np.log10(row_norm)
            row_norm_log /= np.log10(maxval) - np.log10(minval)
            # s/km or s/deg
            bars1 = ax5.bar(x=(i * dw1) * np.ones(N2),                
                            height=dh1 * np.ones(N2),
                            width=dw1, bottom=dh1 * np.arange(N2),
                            color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
        
        # Second resolution
        kwargs2 = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-sx, slm_x=sx, sll_y=-sy, slm_y=sy, sl_s=sl_s,
            # sliding window properties
            win_len=win_len2, win_frac=win_frac,
            # frequency properties
            frqlow=f111, frqhigh=f222, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9,
            stime=starttime_fk+t_interval-(interval_win2/2),
            etime=starttime_fk+t_interval+(interval_win2/2)
        )
                 
        out2 = array_processing(st_fk2.copy(), **kwargs2) # output of array_processing
        
        # Make output human readable, adjust backazimuth to values between 0 and 360
        t2, rel_power2, abs_power2, baz2, slow2 = out2.T
        baz2[baz2 < 0.0] += 360
        
        N1_m = 72
        N2_m = int((sx-sx_m)/sl_s)
        abins2 = np.arange(N1_m + 1) * 360. / N1_m   # angle resolution
        if slow_unit == "skm":
            sbins2 = np.linspace(sx_m, sx, N2_m + 1)   # slowness resolution s/km
        else:
            sbins2 = np.linspace(sx_m*100, sx*100, N2_m + 1)   # slowness resolution s/deg
        
        # Sum rel power in bins given by abins and sbins
        # s/km
        if slow_unit == "skm":
            hist2, baz_edges2, sl_edges2 = \
                np.histogram2d(baz2, slow2, bins=[abins2, sbins2], weights=rel_power2)
        # s/deg
        else:
            hist2, baz_edges2, sl_edges2 = \
                np.histogram2d(baz2, slow2*KM_PER_DEG, bins=[abins2, sbins2], weights=rel_power2)
        
        # Transform to radian
        baz_edges2 = np.radians(baz_edges2)
        dh2 = abs(sl_edges2[1] - sl_edges2[0])
        dw2 = abs(baz_edges2[1] - baz_edges2[0])
                   
        for i, row in enumerate(hist2):
            row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
            row_norm_log = np.log10(row_norm)
            row_norm_log /= np.log10(maxval) - np.log10(minval)
            # s/km
            if slow_unit == "skm":
                bars2 = ax5.bar(x=(i * dw2) * np.ones(N2_m),                
                                height=dh2 * np.ones(N2_m) + sx_m,
                                width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m,
                                color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
            # s/deg
            else:
                bars2 = ax5.bar(x=(i * dw2) * np.ones(N2_m),                
                                height=dh2 * np.ones(N2_m) + sx_m*100,
                                width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m*100,
                                color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
            
        ax5.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax5.set_xticklabels(['N', 'E', 'S', 'W'])
        # ax5.set_title('t = '+str(int(t_interval+start))+"s\n" +str(len(st_fk2))+" STA, baz = %0.1f" % thechosenone["sta_bazi"], y=1.1,fontsize=11)
        ax5.set_title(thechosenone["sta_name"]+", "+str(len(st_fk2))+" STA \n"+" dist = %0.1f" % thechosenone["sta_dist"]+"\N{DEGREE SIGN}, baz = %0.1f" % thechosenone["sta_bazi"]+"\N{DEGREE SIGN}", y=1.1,fontsize=11)
        
        # Set slowness limits
        if slow_unit == "skm":
            ax5.set_ylim(0, sx) # s/km
        else:
            ax5.set_ylim(0, sx*100) # s/deg
        [i.set_color('grey') for i in ax5.get_yticklabels()]
        
        # add a line to split two resolutions
        rads = np.arange(0, (2*np.pi), 0.01)
        if slow_unit == "skm":
            ax5.plot(rads,[sx_m]*len(rads), color="k",zorder=5, lw=0.9)      # s/km
        else:
            ax5.plot(rads,[sx_m*100]*len(rads), color="k",zorder=5, lw=0.8)  # s/deg
        
        # Plot backzimuth on FK plot
        # s/km
        if slow_unit == "skm":
            arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
            arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
            ax5.annotate('', xy=(arrow_baz, 0.5), xytext=(arrow_baz, 0.62),
                              arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
            ax5.annotate('', xy=(arrow_baz_r, 0.5), xytext=(arrow_baz_r, 0.62),
                              arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
        # s/deg
        else:
            arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
            arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
            ax5.annotate('', xy=(arrow_baz, 50), xytext=(arrow_baz, 62),
                              arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
            ax5.annotate('', xy=(arrow_baz_r, 50), xytext=(arrow_baz_r, 62),
                              arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)

        # set colorbar
        cbar = ColorbarBase(ax6, cmap=cmap_fk) 
        cbar.set_ticks([])
        cbar.set_label("Relative Power", rotation=270, labelpad=14, fontsize=11)
        
        # Ray path plot
        # rayplot = arr.plot_rays(ax=ax7)
        plot_arr_rays(thechosenone["arr"], start, end, t_interval, ax=ax7)
        
        plt.tight_layout(h_pad=1)
        
        if plot_save:
            if plot_3c:
                if save_option == "pdf":
                    plt.savefig(movie_directory+ event_dic['event_name'] +"_3C_FKanalysis_raypath_"+ "%07.1f"%(start+it*timestep)+"s."+save_option)
                else:
                    plt.savefig(movie_directory+ event_dic['event_name'] +"_3C_FKanalysis_raypath_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
            else:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_FKanalysis_raypath_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
                
            plt.clf()
            plt.close()
            
        else:
            plt.show()
            
    if plot_save:
        print("Plots are saved. End of plotting.")
    else:
        print("Plots are not saved. End of plotting.")


# ====================
# 4 FK analysis plots
        
def GMV_FK4(GMV, event_dic, stream_info, thechosenones, subarray_dicts, FK_dict,
            start_movie, end_movie, interval,
            vmin, vmax, arr_img, 
            movie_directory, timeframes=None, slow_unit="sdeg", timelabel="s",
            plot_save=False, save_option="png", save_dpi=120,
            plot_3c=True, plot_rotate=True):
    
    """
    Plot single timestep of GMV, reference seismograms, ray path plot, and FK diagram
    
    :param GMV: processed data dictionary
    :param event_dic: event dictionary
    :param stream_info: stream info dictionary
    :param thechosenone: the reference station info dictionary
    :param subarray_dict: subarray info dictionary
    :param FK_dict: FK analysis streams and info dictionary
    :param start_movie: starting time of the movie (in s)
    :param end_movie: ending time of the movie (in s)
    :param interval: movie interval (index)
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param arr_img: AlpArray logo
    :param movie_directory: movie directory path
    :param slow_unit: slowness unit "sdeg" or "skm", default: "sdeg"
    :param plot_save: If True, save figure
    :param plot_3c: If True, plot in 3 component on GMV
    :param plot_rotate: If True, seismograms plotted in RT instead of NE
    """
    
    # Variables
    start    = stream_info["start"]
    end      = stream_info["end"]
    timestep = stream_info["timestep"]
    time_st  = stream_info["time_st"]
    lat_sta_new  = GMV["lat_sta"]
    lon_sta_new  = GMV["lon_sta"]
    name_sta_new = GMV["name_sta"]

    sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, interval_win2, win_len2, minval, maxval = FK_dict
    f11, f22, f111, f222 = freq
    lat_ant, lon_ant     = antipode(event_dic["lat"], event_dic["lon"])
    
    start_movie_index = int((start_movie-start)/timestep)
    end_movie_index   = int((end_movie-start)/timestep)

    if timelabel == "hr":
        d_time        = 60*60
        seismo_labels = np.arange(round(start/d_time *2 +0.5)/2, round(end/d_time *2 +0.5)/2, 0.5)
        print("Seismograms will be plotted in HOURS.")
    elif timelabel == "min":
        d_time        = 60
        seismo_labels = np.arange(round(round(start/d_time*2 +1)/2, -1), round(round(end/d_time*2 +1)/2,-1), 15)
        print("Seismograms will be plotted in MINUTES.")
    else:
        d_time = 1
        seismo_labels = np.arange(round(start+1, -3), round(end, -3)+1, 1000)
        print("Seismograms will be plotted in SECONDS.")
    
    if timeframes is not None:
        print("The following time frames in seconds will be plotted:")
        print(*timeframes)
        timeframes = ((np.array(timeframes)-start)/timestep).astype(int)
    else:
        timeframes = range(start_movie_index, end_movie_index+1, interval)
        # Start and end of movies in seconds
        if timelabel == "hr":
            print("Movie will start at %02.1f"%(start_movie/d_time)+" hr ("+str(start_movie)+"s) after OT and end at %02.1f"%(end_movie/d_time)+" hr ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        elif timelabel == "min":
            print("Movie will start at %04.1f"%(start_movie/d_time)+" min ("+str(start_movie)+"s) after OT and end at %04.1f"%(end_movie/d_time)+" min ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        else:
            print("Movie will start at "+str(start_movie)+"s after OT and end at "+str(end_movie)+"s after OT with interval "+str(interval*timestep)+"s.")

    if plot_3c:
        print("Ready to plot 3C figures!")
    else:
        print("Ready to plot (vertical component only)!")  
        
    if plot_save:
        print("Plots will be saved in " + movie_directory)
    else:
        print("Plots will not be saved.")
        
    print("Plotting GMV and 4 FK analysis plots...")
    
    for it in timeframes: # from start time to end time 
        if it == start_movie_index or (start+it*timestep) % 500 == 0:
            print("Plotting %06.1f s..."%(start+it*timestep))
        # Setting up the plot
        fig = plt.figure(figsize=(16, 9)) 
        gs=GridSpec(7,6, left=0.05, right=0.97, height_ratios=[1.5,0.8,0.06,0.3,0.3,0.3,0.3], width_ratios=[4.,0.05,1.5,0.08,1.5,0.09])
        gs.update(hspace=0.1)
        ax1 = plt.subplot(gs[:2,0])
        axs1 = plt.subplot(gs[2,0])
        axs1.set_visible(False)
        ax2 = plt.subplot(gs[3,0])
        ax3 = plt.subplot(gs[4,0])
        ax4 = plt.subplot(gs[5,0])
        ax5 = plt.subplot(gs[6,0])
        # FK-analysis
        ax6 = plt.subplot(gs[0, 2], polar=True)
        ax7 = plt.subplot(gs[0, 4], polar=True)
        ax8 = plt.subplot(gs[2:,2], polar=True)
        ax9 = plt.subplot(gs[2:,4], polar=True)
        # Colorbar
        ax_c = plt.subplot(gs[2: ,-1])
        
        # Array processing interval
        t_interval = it*timestep
        end_fk     = end # 3600 # ending of seismogram
        
        # Setting up the map
        m = Basemap(projection='mill',llcrnrlat=AA_lat1,urcrnrlat=AA_lat2,llcrnrlon=AA_lon1,urcrnrlon=AA_lon2,resolution="i",ax=ax1)
        m.drawcoastlines()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightyellow',lake_color='lightblue')
        m.drawcountries(color="lightgrey")
        
        # Draw parallels and meridians.
        parallels = np.arange(AA_lat1,AA_lat2+dlat,dlat)
        # Label the meridians and parallels
        m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        # Draw Meridians and Labels
        meridians = np.arange(AA_lon1,AA_lon2+dlon,dlon)
        m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        
        # Draw subarray box
        array_colors = ["orange","k", "m", "lime"]
        for subarray_dict, clr in zip(subarray_dicts, array_colors):
            draw_screen_poly(subarray_dict["lats_rec"], subarray_dict["lons_rec"], m, ax1, c=clr)
        
        # Plot the stations
        # Plot 3C motion
        if plot_3c:
            x_sta, y_sta = m(lon_sta_new+GMV["GMV_E"][:,it], lat_sta_new+GMV["GMV_N"][:,it])
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot vertical motion only
        else:
            x_sta, y_sta = m(lon_sta_new, lat_sta_new)
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot reference stations of subarrays
        for thechosenone, clr in zip(thechosenones, array_colors):
            alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_Z"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
            alpmap_n = m.plot(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], fillstyle='none', markeredgecolor="lime", markeredgewidth=3, marker="o", markersize=8, zorder=5, ax=ax1)
        mcolorbar(alpmap, vmin, vmax) # Plot map colorbar
        
        # Plot great circle path
        m.drawgreatcircle(thechosenones[1]["sta_lon"],thechosenones[1]["sta_lat"],event_dic["lon"],event_dic["lat"],linewidth=2,color='g', zorder=5)
        m.drawgreatcircle(thechosenones[1]["sta_lon"],thechosenones[1]["sta_lat"],lon_ant,lat_ant,linewidth=2,color='orange', zorder=5)
                    
        ax1.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                      "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                      ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                      '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                      ', Distance '+ "%.1f"% np.median(GMV["dist_sta"])+'\N{DEGREE SIGN}, '+str(len(name_sta_new))+' STA', fontsize=14)
       
        # Add AA logo on plot
        imagebox = OffsetImage(arr_img, zoom=0.45)
        imagebox.image.axes = ax1
        ab = AnnotationBbox(imagebox, (1, 1),
                            xybox=(40., 277.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5, frameon=False)
        
        ax1.add_artist(ab)
        
        # Plot seismograms
        # HHZ
        for ax, thechosenone in zip((ax2,ax3,ax4,ax5), thechosenones):
            ax.plot((time_st+start)/d_time, GMV["GMV_Z"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".Z")
            ax.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
            # Plot phase marker for channel Z
            phase_marker(thechosenone["arr"], ax , "Z", start/d_time, end_fk/d_time, timelabel=timelabel)
            # Box showing the window for array processing
            ax.add_patch(plt.Rectangle(( (t_interval+start-(interval_win2/2))/d_time, -1.1), interval_win2/d_time, 2.2, fc="c", alpha =0.6))
            
            ax.grid(b=True, which='major')
            ax.set_xticks(seismo_labels)
            ax.set_xlim([start/d_time,end_fk/d_time])
            ax.set_ylim([-1.1,1.1]) 
            ax.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.05))
        
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        ax4.set_xticklabels([])

        ax4.set_ylabel('             Normalized Displacement', fontsize=11)
        
        ax5.tick_params(axis="x",labelsize=12)
        if timelabel == "hr":
            ax5.set_xlabel('Time after origin [hr] \n'+str(int(t_interval+start))+'s', fontsize=14)
            ax5.xaxis.set_minor_locator(MultipleLocator(0.1))
        elif timelabel == "min":
            ax5.set_xlabel('Time after origin [min] \n'+str(int(t_interval+start))+'s', fontsize=14)
        else:
            ax5.set_xlabel('Time after origin [s] \n'+str(int(t_interval+start))+'s', fontsize=14)
        ax4.set_xticks(seismo_labels)
        ax4.set_xticklabels(seismo_labels) 
        
        ## Array Processing

        # Plotting
        for ax, thechosenone, subarray_dict in zip((ax7,ax9,ax6,ax8), thechosenones, subarray_dicts):
            st_fk1   = subarray_dict["st_fk1"]
            st_fk2   = subarray_dict["st_fk2"]
            starttime_fk = subarray_dict["starttime_fk"]

            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")
            
            # First resolution
            kwargs1 = dict(
                # slowness grid: X min, X max, Y min, Y max, Slow Step
                sll_x=-sx_m, slm_x=sx_m, sll_y=-sx_m, slm_y=sx_m, sl_s=sl_s,
                # sliding window properties
                win_len=win_len, win_frac=win_frac,
                # frequency properties
                frqlow=f11, frqhigh=f22, prewhiten=0,
                # restrict output
                semb_thres=-1e9, vel_thres=-1e9,
                stime=starttime_fk+t_interval-(interval_win/2),
                etime=starttime_fk+t_interval+(interval_win/2)
            )
            
            out1 = array_processing(st_fk1.copy(), **kwargs1) # output of array_processing
            
            # Make output human readable, adjust backazimuth to values between 0 and 360
            t1, rel_power1, abs_power1, baz1, slow1 = out1.T
            baz1[baz1 < 0.0] += 360
            
            N1 = 36
            N2 = int(sx_m/sl_s)
            abins1 = np.arange(N1 + 1) * 360. / N1   # angle resolution
            if slow_unit == "skm":
                sbins1 = np.linspace(0, sx_m, N2 + 1)      # slowness resolution s/km
            else:
                sbins1 = np.linspace(0, sx_m*100, N2 + 1)  # slowness resolution s/deg
            
            # Sum rel power in bins given by abins and sbins
            # s/km
            if slow_unit == "skm":
                hist1, baz_edges1, sl_edges1 = \
                    np.histogram2d(baz1, slow1, bins=[abins1, sbins1], weights=rel_power1)
            # s/deg
            else: 
                hist1, baz_edges1, sl_edges1 = \
                    np.histogram2d(baz1, slow1*KM_PER_DEG, bins=[abins1, sbins1], weights=rel_power1)
                
            # Transform to radian
            baz_edges1 = np.radians(baz_edges1)
            dh1 = abs(sl_edges1[1] - sl_edges1[0])
            dw1 = abs(baz_edges1[1] - baz_edges1[0])
            
            # circle through backazimuth
            for i, row in enumerate(hist1):
                row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
                row_norm_log = np.log10(row_norm)
                row_norm_log /= np.log10(maxval) - np.log10(minval)
                # s/km or s/deg
                bars1 = ax.bar(x=(i * dw1) * np.ones(N2),                
                               height=dh1 * np.ones(N2),
                               width=dw1, bottom=dh1 * np.arange(N2),
                               color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
            
            # Second resolution
            kwargs2 = dict(
                # slowness grid: X min, X max, Y min, Y max, Slow Step
                sll_x=-sx, slm_x=sx, sll_y=-sy, slm_y=sy, sl_s=sl_s,
                # sliding window properties
                win_len=win_len2, win_frac=win_frac,
                # frequency properties
                frqlow=f111, frqhigh=f222, prewhiten=0,
                # restrict output
                semb_thres=-1e9, vel_thres=-1e9,
                stime=starttime_fk+t_interval-(interval_win2/2),
                etime=starttime_fk+t_interval+(interval_win2/2)
            )
                     
            out2 = array_processing(st_fk2.copy(), **kwargs2) # output of array_processing
            
            # Make output human readable, adjust backazimuth to values between 0 and 360
            t2, rel_power2, abs_power2, baz2, slow2 = out2.T
            baz2[baz2 < 0.0] += 360
            
            N1_m = 72
            N2_m = int((sx-sx_m)/sl_s)
            abins2 = np.arange(N1_m + 1) * 360. / N1_m   # angle resolution
            if slow_unit == "skm":
                sbins2 = np.linspace(sx_m, sx, N2_m + 1)   # slowness resolution s/km
            else:
                sbins2 = np.linspace(sx_m*100, sx*100, N2_m + 1)   # slowness resolution s/deg
            
            # Sum rel power in bins given by abins and sbins
            # s/km
            if slow_unit == "skm":
                hist2, baz_edges2, sl_edges2 = \
                    np.histogram2d(baz2, slow2, bins=[abins2, sbins2], weights=rel_power2)
            # s/deg
            else:
                hist2, baz_edges2, sl_edges2 = \
                    np.histogram2d(baz2, slow2*KM_PER_DEG, bins=[abins2, sbins2], weights=rel_power2)
            
            # Transform to radian
            baz_edges2 = np.radians(baz_edges2)
            dh2 = abs(sl_edges2[1] - sl_edges2[0])
            dw2 = abs(baz_edges2[1] - baz_edges2[0])
                       
            for i, row in enumerate(hist2):
                row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
                row_norm_log = np.log10(row_norm)
                row_norm_log /= np.log10(maxval) - np.log10(minval)
                # s/km
                if slow_unit == "skm":
                    bars2 = ax.bar(x=(i * dw2) * np.ones(N2_m),                
                                    height=dh2 * np.ones(N2_m) + sx_m,
                                    width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m,
                                    color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
                # s/deg
                else:
                    bars2 = ax.bar(x=(i * dw2) * np.ones(N2_m),                
                                   height=dh2 * np.ones(N2_m) + sx_m*100,
                                   width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m*100,
                                   color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
                
            
            # Set slowness limits
            if slow_unit == "skm":
                ax.set_ylim(0, sx) # s/km
            else:
                ax.set_ylim(0, sx*100) # s/deg
            [i.set_color('grey') for i in ax.get_yticklabels()]
            
            # add a line to split two resolutions
            rads = np.arange(0, (2*np.pi), 0.01)
            if slow_unit == "skm":
                ax.plot(rads,[sx_m]*len(rads), color="k",zorder=5, lw=0.9)      # s/km
            else:
                ax.plot(rads,[sx_m*100]*len(rads), color="k",zorder=5, lw=0.8)  # s/deg
            
            # Plot backzimuth on FK plot
            # s/km
            if slow_unit == "skm":
                arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
                arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
                ax.annotate('', xy=(arrow_baz, 0.5), xytext=(arrow_baz, 0.62),
                                  arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
                ax.annotate('', xy=(arrow_baz_r, 0.5), xytext=(arrow_baz_r, 0.62),
                                  arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
            # s/deg
            else:
                arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
                arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
                ax.annotate('', xy=(arrow_baz, 50), xytext=(arrow_baz, 62),
                                  arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
                ax.annotate('', xy=(arrow_baz_r, 50), xytext=(arrow_baz_r, 62),
                                  arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)

        
            ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
            ax.set_xticklabels(['N', 'E', 'S', 'W'])
        
            # Set title
            ax.set_title(thechosenone["sta_name"]+", "+str(len(st_fk2))+" STA \n"+" dist = %0.1f" % thechosenone["sta_dist"]+"\N{DEGREE SIGN}, baz = %0.1f" % thechosenone["sta_bazi"]+"\N{DEGREE SIGN}", y=1.1,fontsize=11)
        
        # set colorbar
        
        cbar = ColorbarBase(ax_c, cmap=cmap_fk) 
        cbar.set_ticks([])
        cbar.set_label("Relative Power", rotation=270, labelpad=14, fontsize=11)
        
        plt.tight_layout(h_pad=1)
        
        if plot_save:
            if plot_3c:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_3C_4FKanalysis_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
            else:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_4 FKanalysis_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
                
            plt.clf()
            plt.close()
            
        else:
            plt.show()
            
    if plot_save:
        print("Plots are saved. End of plotting.")
    else:
        print("Plots are not saved. End of plotting.")

# ====================
def GMV_ZeroX(GMV, event_dic, stream_info, thechosenone, 
              start_movie, end_movie, interval,
              vmin, vmax, arr_img, 
              movie_directory, timeframes=None, timelabel="s",
              plot_save=False, save_option="png", save_dpi=120,
              plot_local=False, plot_Vonly=False):
    
    """
    Plot zero-crossing in GMV and reference seismogram
    
    :param GMV: processed data dictionary
    :param event_dic: event dictionary
    :param stream_info: stream info dictionary
    :param thechosenone: the reference station info dictionary
    :param start_movie: starting time of the movie (in s)
    :param end_movie: ending time of the movie (in s)
    :param interval: movie interval (index)
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param arr_img: AlpArray logo
    :param movie_directory: movie directory path
    :param plot_save: If True, save figure
    :param plot_local: If True, plot local event
    :param plot_3c: If True, plot in 3 component on GMV
    :param plot_rotate: If True, seismograms plotted in RT instead of NE
    """
    
    # Variables
    start    = stream_info["start"]
    end      = stream_info["end"]
    timestep = stream_info["timestep"]
    time_st  = stream_info["time_st"]
    lat_sta_new  = GMV["lat_sta"]
    lon_sta_new  = GMV["lon_sta"]
    name_sta_new = GMV["name_sta"]
    
    start_movie_index = int((start_movie-start)/timestep)
    end_movie_index   = int((end_movie-start)/timestep)
    
    if timelabel == "hr":
        d_time        = 60*60
        seismo_labels = np.arange(round(start/d_time *2 +0.5)/2, round(end/d_time *2 +0.5)/2, 0.5)
        print("Seismograms will be plotted in HOURS.")
    elif timelabel == "min":
        d_time        = 60
        seismo_labels = np.arange(round(round(start/d_time*2 +1)/2, -1), round(round(end/d_time*2 +1)/2,-1), 15)
        print("Seismograms will be plotted in MINUTES.")
    else:
        d_time = 1
        if plot_local:
            seismo_labels = np.arange(int(start), int(end)+1, 100)
        else:
            seismo_labels = np.arange(round(start+1, -3), round(end, -3)+1, 1000)
        print("Seismograms will be plotted in SECONDS.")
    
    if timeframes is not None:
        print("The following time frames in seconds will be plotted:")
        print(*timeframes)
        timeframes = ((np.array(timeframes)-start)/timestep).astype(int)
    else:
        timeframes = range(start_movie_index, end_movie_index+1, interval)
        # Start and end of movies in seconds
        if timelabel == "hr":
            print("Movie will start at %02.1f"%(start_movie/d_time)+" hr ("+str(start_movie)+"s) after OT and end at %02.1f"%(end_movie/d_time)+" hr ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        elif timelabel == "min":
            print("Movie will start at %04.1f"%(start_movie/d_time)+" min ("+str(start_movie)+"s) after OT and end at %04.1f"%(end_movie/d_time)+" min ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        else:
            print("Movie will start at "+str(start_movie)+"s after OT and end at "+str(end_movie)+"s after OT with interval "+str(interval*timestep)+"s.")
    
    if plot_Vonly:
        print("Ready to plot (vertical component only)!")  
    else:
        print("Ready to plot zero crossings (vertical component only)!")  
        
    if plot_save:
        print("Plots will be saved in " + movie_directory)
    else:
        print("Plots will not be saved.")
        
    step = 0 # used only when plot_local is True
    for it in timeframes: # from start time to end time 
        if it == start_movie_index or (start+it*timestep) % 500 == 0:
            print("Plotting %07.1f s..."%(start+it*timestep))
        # Just the simple map and seismograms
        # Setting up the plot
        fig = plt.figure(figsize=(10, 8.5)) 
        gs=GridSpec(3,1, height_ratios=[4.8,0.03,0.8])
        gs.update(hspace=0.1)
        ax1 = plt.subplot(gs[0])
        axs = plt.subplot(gs[1])
        axs.set_visible(False)
        ax2 = plt.subplot(gs[2])
        
        # Setting up the map
        m = Basemap(projection='mill',llcrnrlat=AA_lat1,urcrnrlat=AA_lat2,llcrnrlon=AA_lon1,urcrnrlon=AA_lon2,resolution="i",ax=ax1)
        # m.shadedrelief()
        # m.etopo()
        m.drawcoastlines()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightyellow',lake_color='lightblue')
        m.drawcountries(color="lightgrey")
        
        # Draw parallels and meridians.
        parallels = np.arange(AA_lat1,AA_lat2+dlat,dlat)
        # Label the meridians and parallels
        m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        # Draw Meridians and Labels
        meridians = np.arange(AA_lon1,AA_lon2+dlon,dlon)
        m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        
        # Plot the stations
        # Plot vertical motion only
        x_sta, y_sta = m(lon_sta_new, lat_sta_new)
        if plot_Vonly:
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        else:
            alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_ZeroX"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot reference station (Plotting it again to avoid being covered by other dots)
        if plot_Vonly:
            alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_Z"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
        else:
            alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_ZeroX"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
        alpmap_n = m.plot(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], fillstyle='none', markeredgecolor="lime", markeredgewidth=3, marker="o", markersize=8, zorder=5, ax=ax1)
        
        if plot_local: # Plot local epicenter
            x_eq, y_eq = m(event_dic["lon"], event_dic["lat"])
            alpmap_eq = m.plot(x_eq, y_eq, c="yellow", markeredgecolor="k", markeredgewidth=1.5, marker="*", markersize=15, zorder=4, ax=ax1)
        
        mcolorbar(alpmap, vmin, vmax) # Plot map colorbar
        
        ax1.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                      "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                      ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                      '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                      ', Distance '+ "%.1f"% np.median(GMV["dist_sta"])+'\N{DEGREE SIGN}, '+str(len(name_sta_new))+' STA', fontsize=14)
    
        # Add AA logo on plot
        imagebox = OffsetImage(arr_img, zoom=0.45)
        imagebox.image.axes = ax1
        ab = AnnotationBbox(imagebox, (1, 1),
                            xybox=(40., 337.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5, frameon=False)
        
        ax1.add_artist(ab)
        
        # Plot reference vertical seismogram
        # HHZ
        ax2.plot( (time_st+start)/d_time, GMV["GMV_Z"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".Z")
        ax2.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        # Plot phase marker for channel Z
        phase_marker(thechosenone["arr"], ax2, "Z", start/d_time, end/d_time, timelabel=timelabel, plot_local=plot_local)
        
        ax2.grid(b=True, which='major')
        ax2.set_ylabel('Normalized'+'\n'+'Displacement', fontsize=11)
        if timelabel == "hr":
            ax2.set_xlabel('Time after origin [hr]', fontsize=14)
            ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
        elif timelabel == "min":
            ax2.set_xlabel('Time after origin [min]', fontsize=14)
        else:
            ax2.set_xlabel('Time after origin [s]', fontsize=14)
        ax2.set_xlim([start/d_time,end/d_time])
        ax2.set_ylim([-1.1,1.1]) 
        ax2.set_xticks(seismo_labels)
        ax2.set_xticklabels(seismo_labels) 
        ax2.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.07))
        
        plt.tight_layout(h_pad=1)
        
        if plot_save:
            if plot_local:
                step += 1
            else:
                step = start+it*timestep 
            
            if plot_Vonly:
                plt.savefig(movie_directory+ event_dic['event_name'] + "%07.1f"%(step)+"s."+save_option, dpi=save_dpi)               
            else:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_ZeroX_"+ "%07.1f"%(step)+"s."+save_option, dpi=save_dpi)               
            plt.clf()
            plt.close()
            
        else:
            plt.show()
            
    if plot_save:
        print("Plots are saved. End of plotting.")
    else:
        print("Plots are not saved. End of plotting.")
        
# ====================
# FK analysis plot for normal mode
def GMV_FK_NM(GMV, event_dic, stream_info, thechosenone, subarray_dict, FK_dict,
              start_movie, end_movie, interval,
              vmin, vmax, arr_img, 
              movie_directory, timeframes=None, slow_unit="sdeg", 
              timelabel="s", plot_ZeroX=False,
              plot_save=False, save_option="png", save_dpi=120,
              plot_3c=True, plot_rotate=True):
    
    """
    Plot single timestep of GMV, reference seismograms, and FK diagram
    
    :param GMV: processed data dictionary
    :param event_dic: event dictionary
    :param stream_info: stream info dictionary
    :param thechosenone: the reference station info dictionary
    :param subarray_dict: subarray info dictionary
    :param FK_dict: FK analysis streams and info dictionary
    :param start_movie: starting time of the movie (in s)
    :param end_movie: ending time of the movie (in s)
    :param interval: movie interval (index)
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param arr_img: AlpArray logo
    :param movie_directory: movie directory path
    :param slow_unit: slowness unit "sdeg" or "skm", default: "sdeg"
    :param plot_save: If True, save figure
    :param plot_3c: If True, plot in 3 component on GMV
    :param plot_rotate: If True, seismograms plotted in RT instead of NE
    """
    
    # Variables
    start    = stream_info["start"]
    end      = stream_info["end"]
    timestep = stream_info["timestep"]
    time_st  = stream_info["time_st"]
    lat_sta_new  = GMV["lat_sta"]
    lon_sta_new  = GMV["lon_sta"]
    name_sta_new = GMV["name_sta"]
    st_fk1   = subarray_dict["st_fk1"]
    starttime_fk = subarray_dict["starttime_fk"]
    sx, sy, sx_m, sl_s, freq, win_frac, interval_win, win_len, minval, maxval = FK_dict
    f11, f22 = freq
    
    start_movie_index = int((start_movie-start)/timestep)
    end_movie_index   = int((end_movie-start)/timestep)

    if timelabel == "hr":
        d_time        = 60*60
        seismo_labels = np.arange(round(start/d_time *2 +0.5)/2, round(end/d_time *2 +0.5)/2, 0.5)
        print("Seismograms will be plotted in HOUSE.")
    elif timelabel == "min":
        d_time        = 60
        seismo_labels = np.arange(round(round(start/d_time*2 +1)/2, -1), round(round(end/d_time*2 +1)/2,-1), 15)
        print("Seismograms will be plotted in MINUTES.")
    else:
        d_time = 1
        seismo_labels = np.arange(round(start+1, -3), round(end, -3)+1, 1000)
        print("Seismograms will be plotted in SECONDS.")
    
    if timeframes is not None:
        print("The following time frames in seconds will be plotted:")
        print(*timeframes)
        timeframes = ((np.array(timeframes)-start)/timestep).astype(int)
    else:
        timeframes = range(start_movie_index, end_movie_index+1, interval)
        # Start and end of movies in seconds
        if timelabel == "hr":
            print("Movie will start at %02.1f"%(start_movie/d_time)+" hr ("+str(start_movie)+"s) after OT and end at %02.1f"%(end_movie/d_time)+" hr ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        elif timelabel == "min":
            print("Movie will start at %04.1f"%(start_movie/d_time)+" min ("+str(start_movie)+"s) after OT and end at %04.1f"%(end_movie/d_time)+" min ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        else:
            print("Movie will start at "+str(start_movie)+"s after OT and end at "+str(end_movie)+"s after OT with interval "+str(interval*timestep)+"s.")
        
    if plot_3c:
        print("Ready to plot 3D wave motion and FK analysis!")
    else:
        print("Ready to plot vertical seismogram and FK analysis (vertical component only)!")  
        
    if plot_ZeroX:
        print("Plotting zero crossings instead of full ground motion.")
        
    if plot_save:
        print("Plots will be saved in " + movie_directory)
    else:
        print("Plots will not be saved.")
                
    for it in timeframes: # from start time to end time 
        if it == start_movie_index or (start+it*timestep) % 500 == 0:
            print("Plotting %06.1f s..."%(start+it*timestep))
        # Setting up the plot
        fig = plt.figure(figsize=(16, 9)) 
        gs=GridSpec(4,4, height_ratios=[0.45,3.4,0.4,0.7], width_ratios=[2.1,0.03,1,0.05])
        gs.update(hspace=0.1)
        ax1 = plt.subplot(gs[:3,0])
        ax2 = plt.subplot(gs[-1,0])
        ax3 = plt.subplot(gs[1,2], polar=True) # FK plot
        ax4 = plt.subplot(gs[1,-1])# FK colorbar 
        
        # Array processing interval
        t_interval = it*timestep
        end_fk     = end # 3600 # ending of seismogram
        
        # Setting up the map
        m = Basemap(projection='mill',llcrnrlat=AA_lat1,urcrnrlat=AA_lat2,llcrnrlon=AA_lon1,urcrnrlon=AA_lon2,resolution="i",ax=ax1)
        m.drawcoastlines()
        m.drawmapboundary(fill_color='lightblue')
        m.fillcontinents(color='lightyellow',lake_color='lightblue')
        m.drawcountries(color="lightgrey")
        
        # Draw parallels and meridians.
        parallels = np.arange(AA_lat1,AA_lat2+dlat,dlat)
        # Label the meridians and parallels
        m.drawparallels(parallels,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        # Draw Meridians and Labels
        meridians = np.arange(AA_lon1,AA_lon2+dlon,dlon)
        m.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        
        # Draw subarray box
        draw_screen_poly(subarray_dict["lats_rec"], subarray_dict["lons_rec"], m, ax1)
        
        # Plot the stations
        # Plot 3C motion
        if plot_3c:
            x_sta, y_sta = m(lon_sta_new+GMV["GMV_E"][:,it], lat_sta_new+GMV["GMV_N"][:,it])
            if plot_ZeroX:
                alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_ZeroX"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
            else:
                alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot vertical motion only
        else:
            x_sta, y_sta = m(lon_sta_new, lat_sta_new)
            if plot_ZeroX:
                alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_ZeroX"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
            else:
                alpmap = m.scatter(x_sta, y_sta, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1)
        # Plot reference station (Plotting it again to avoid being covered by other dots)
        if plot_ZeroX:
            alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_ZeroX"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
        else:
            alpmap_n1= m.scatter(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], c=GMV["GMV_Z"][thechosenone["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4, ax=ax1)
        alpmap_n = m.plot(x_sta[thechosenone["sta_index"]], y_sta[thechosenone["sta_index"]], fillstyle='none', markeredgecolor="lime", markeredgewidth=3, marker="o", markersize=8, zorder=5, ax=ax1)
        mcolorbar(alpmap, vmin, vmax) # Plot map colorbar
        
        ax1.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                      "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                      ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                      '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                      ', Distance '+ "%.1f"% np.median(GMV["dist_sta"])+'\N{DEGREE SIGN}, '+str(len(name_sta_new))+' STA', fontsize=14)
    
        # Add AA logo on plot
        imagebox = OffsetImage(arr_img, zoom=0.45)
        imagebox.image.axes = ax1
        ab = AnnotationBbox(imagebox, (1, 1),
                            xybox=(40., 307.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5, frameon=False)
        
        ax1.add_artist(ab)
        
        # Plot reference seismograms
        # HHZ
        ax2.plot( (time_st+start)/d_time, GMV["GMV_Z"][thechosenone["sta_index"]], color="k", linewidth=1.5, label=thechosenone["sta_name"]+".Z")
        ax2.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        # Plot phase marker for channel Z
        phase_marker(thechosenone["arr"], ax2, "Z", start/d_time, end/d_time, timelabel=timelabel)
        # Box showing the window for array processing
        ax2.add_patch(plt.Rectangle(( (t_interval+start-(interval_win/2))/d_time, -1.1), interval_win/d_time, 2.2, fc="c", alpha=0.6))
        
        ax2.grid(b=True, which='major')
        ax2.set_ylabel('Normalized'+'\n'+'Displacement', fontsize=11)
        if timelabel == "hr":
            ax2.set_xlabel('Time after origin [hr]', fontsize=14)
            ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
        elif timelabel == "min":
            ax2.set_xlabel('Time after origin [min]', fontsize=14)
        else:
            ax2.set_xlabel('Time after origin [s]', fontsize=14)
        ax2.set_xlim([start/d_time,end/d_time])
        ax2.set_ylim([-1.1,1.1]) 
        ax2.set_xticks(seismo_labels)
        ax2.set_xticklabels(seismo_labels) 
        ax2.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.06))
        
        ## Array Processing

        # Plotting             
        ax3.set_theta_direction(-1)
        ax3.set_theta_zero_location("N")
        
        # First resolution
        kwargs1 = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=-sx, slm_x=sx, sll_y=-sy, slm_y=sy, sl_s=sl_s,
            # sliding window properties
            win_len=win_len, win_frac=win_frac,
            # frequency properties
            frqlow=f11, frqhigh=f22, prewhiten=0,
            # restrict output
            semb_thres=-1e9, vel_thres=-1e9,
            stime=starttime_fk+t_interval-(interval_win/2),
            etime=starttime_fk+t_interval+(interval_win/2)
        )
        
        out1 = array_processing(st_fk1.copy(), **kwargs1) # output of array_processing
        
        # Make output human readable, adjust backazimuth to values between 0 and 360
        t1, rel_power1, abs_power1, baz1, slow1 = out1.T
        baz1[baz1 < 0.0] += 360
        
        N1 = 36
        N2 = int(sx_m/sl_s)
        abins1 = np.arange(N1 + 1) * 360. / N1   # angle resolution
        if slow_unit == "skm":
            sbins1 = np.linspace(0, sx_m, N2 + 1)      # slowness resolution s/km
        else:
            sbins1 = np.linspace(0, sx_m*100, N2 + 1)  # slowness resolution s/deg
        
        ## Sum rel power in bins given by abins and sbins
        # s/km
        if slow_unit == "skm":
            hist1, baz_edges1, sl_edges1 = \
                np.histogram2d(baz1, slow1, bins=[abins1, sbins1], weights=rel_power1)
        # s/deg
        else: 
            hist1, baz_edges1, sl_edges1 = \
                np.histogram2d(baz1, slow1*KM_PER_DEG, bins=[abins1, sbins1], weights=rel_power1)
            
        # Transform to radian
        baz_edges1 = np.radians(baz_edges1)
        dh1 = abs(sl_edges1[1] - sl_edges1[0])
        dw1 = abs(baz_edges1[1] - baz_edges1[0])
        
        # circle through backazimuth
        for i, row in enumerate(hist1):
            row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
            row_norm_log = np.log10(row_norm)
            row_norm_log /= np.log10(maxval) - np.log10(minval)
            # s/km or s/deg
            bars1 = ax3.bar(x=(i * dw1) * np.ones(N2),                
                            height=dh1 * np.ones(N2),
                            width=dw1, bottom=dh1 * np.arange(N2),
                            color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
        
        # Second resolution 
        N1_m = 72
        N2_m = int((sx-sx_m)/sl_s)
        abins2 = np.arange(N1_m + 1) * 360. / N1_m   # angle resolution
        if slow_unit == "skm":
            sbins2 = np.linspace(sx_m, sx, N2_m + 1)   # slowness resolution s/km
        else:
            sbins2 = np.linspace(sx_m*100, sx*100, N2_m + 1)   # slowness resolution s/deg
        
        # Sum rel power in bins given by abins and sbins
        # s/km
        if slow_unit == "skm":
            hist2, baz_edges2, sl_edges2 = \
                np.histogram2d(baz1, slow1, bins=[abins2, sbins2], weights=rel_power1)
        # s/deg
        else:
            hist2, baz_edges2, sl_edges2 = \
                np.histogram2d(baz1, slow1*KM_PER_DEG, bins=[abins2, sbins2], weights=rel_power1)
        
        # Transform to radian
        baz_edges2 = np.radians(baz_edges2)
        dh2 = abs(sl_edges2[1] - sl_edges2[0])
        dw2 = abs(baz_edges2[1] - baz_edges2[0])
                   
        for i, row in enumerate(hist2):
            row_norm = row/minval #row / hist.max() #(np.log10(row / hist.max())+3)/3.
            row_norm_log = np.log10(row_norm)
            row_norm_log /= np.log10(maxval) - np.log10(minval)
            # s/km
            if slow_unit == "skm":
                bars2 = ax3.bar(x=(i * dw2) * np.ones(N2_m),                
                                height=dh2 * np.ones(N2_m) + sx_m,
                                width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m,
                                color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
            # s/deg
            else:
                bars2 = ax3.bar(x=(i * dw2) * np.ones(N2_m),                
                               height=dh2 * np.ones(N2_m) + sx_m*100,
                               width=dw2, bottom=dh2 * np.arange(N2_m) + sx_m*100,
                               color=cmap_fk(row_norm_log), edgecolor="lightgrey", linewidth=0.1, align="edge")
                
        ax3.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax3.set_xticklabels(['N', 'E', 'S', 'W'])
        ax3.set_title(thechosenone["sta_name"]+", "+str(len(st_fk1))+" STA \n"+" dist = %0.1f" % thechosenone["sta_dist"]+"\N{DEGREE SIGN}, baz = %0.1f" % thechosenone["sta_bazi"]+"\N{DEGREE SIGN}", y=1.1, fontsize=11)
                    
        # Set slowness limits
        if slow_unit == "skm":
            ax3.set_ylim(0, sx) # s/km
        else:
            ax3.set_ylim(0, sx*100) # s/deg
        [i.set_color('grey') for i in ax3.get_yticklabels()]
        
        # add a line to split two resolutions
        rads = np.arange(0, (2*np.pi), 0.01)
        if slow_unit == "skm":
            ax3.plot(rads,[sx_m]*len(rads), color="k",zorder=5, lw=0.9)      # s/km
        else:
            ax3.plot(rads,[sx_m*100]*len(rads), color="k",zorder=5, lw=0.8)  # s/deg
        
        # Plot backzimuth on FK plot
        # s/km
        if slow_unit == "skm":
            arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
            arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
            ax3.annotate('', xy=(arrow_baz, 0.5), xytext=(arrow_baz, 0.62),
                              arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
            ax3.annotate('', xy=(arrow_baz_r, 0.5), xytext=(arrow_baz_r, 0.62),
                              arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
        # s/deg
        else:
            arrow_baz   = np.radians(thechosenone["sta_bazi"]) 
            arrow_baz_r = np.radians(thechosenone["sta_bazi"]+180.) if thechosenone["sta_bazi"] < 180. else np.radians(thechosenone["sta_bazi"]-180)
            ax3.annotate('', xy=(arrow_baz, 50), xytext=(arrow_baz, 62),
                              arrowprops=dict(facecolor='green', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)
            ax3.annotate('', xy=(arrow_baz_r, 50), xytext=(arrow_baz_r, 62),
                              arrowprops=dict(facecolor='orange', edgecolor='none', width=2.5, headwidth=7), annotation_clip=False)

        # set colorbar
        cbar = ColorbarBase(ax4, cmap=cmap_fk) 
        cbar.set_ticks([])
        cbar.set_label("Relative Power", rotation=270, labelpad=14, fontsize=11)
        
        plt.tight_layout(h_pad=1)
        
        if plot_save:
            if plot_3c:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_3C_FKanalysis_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
            else:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_FKanalysis_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
                
            plt.clf()
            plt.close()
            
        else:
            plt.show()
            
    if plot_save:
        print("Plots are saved. End of plotting.")
    else:
        print("Plots are not saved. End of plotting.")


# ====================
# Cross-section plot
    
def GMV_Xsec(GMV, event_dic, stream_info, Xsec_dict, thechosenone_Xsec, 
             start_movie, end_movie, interval,
             vmin, vmax, arr_img, 
             movie_directory, timeframes=None, timelabel="s",
             plot_save=False, save_option="png", save_dpi=120,
             plot_3c=True):
    
    """
    Plot single timestep of GMV, reference vertical seismogram, and ZNE/ZRT cross-sections
    
    :param GMV: processed data dictionary
    :param event_dic: event dictionary
    :param stream_info: stream info dictionary
    :param Xsec_dict: cross-section info dictionary
    :param thechosenone_Xsec: the center station info dictionary
    :param start_movie: starting time of the movie (in s)
    :param end_movie: ending time of the movie (in s)
    :param interval: movie interval (index)
    :param vmin: colorbar min
    :param vmax: colorbar max
    :param arr_img: AlpArray logo
    :param movie_directory: movie directory path
    :param plot_save: If True, save figure
    :param plot_3c: If True, plot in 3 component on GMV
    """
        
    # Variables
    start    = stream_info["start"]
    end      = stream_info["end"]
    timestep = stream_info["timestep"]
    time_st  = stream_info["time_st"]
    lat_sta_new  = GMV["lat_sta"]
    lon_sta_new  = GMV["lon_sta"]
    name_sta_new = GMV["name_sta"]
    
    cross_sec = Xsec_dict["cross_sec"]
    dist_sta_Xsec = Xsec_dict["dist_sta_Xsec"]
    
    start_movie_index = int((start_movie-start)/timestep)
    end_movie_index   = int((end_movie-start)/timestep)

    if timelabel == "hr":
        d_time        = 60*60
        seismo_labels = np.arange(round(start/d_time *2 +0.5)/2, round(end/d_time *2 +0.5)/2, 0.5)
        print("Seismograms will be plotted in HOURS.")
    elif timelabel == "min":
        d_time        = 60
        seismo_labels = np.arange(round(round(start/d_time*2 +1)/2, -1), round(round(end/d_time*2 +1)/2,-1), 15)
        print("Seismograms will be plotted in MINUTES.")
    else:
        d_time = 1
        seismo_labels = np.arange(round(start+1, -3), round(end, -3)+1, 1000)
        print("Seismograms will be plotted in SECONDS.")
    
    if timeframes is not None:
        print("The following time frames in seconds will be plotted:")
        print(*timeframes)
        timeframes = ((np.array(timeframes)-start)/timestep).astype(int)
    else:
        timeframes = range(start_movie_index, end_movie_index+1, interval)
        # Start and end of movies in seconds
        if timelabel == "hr":
            print("Movie will start at %02.1f"%(start_movie/d_time)+" hr ("+str(start_movie)+"s) after OT and end at %02.1f"%(end_movie/d_time)+" hr ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        elif timelabel == "min":
            print("Movie will start at %04.1f"%(start_movie/d_time)+" min ("+str(start_movie)+"s) after OT and end at %04.1f"%(end_movie/d_time)+" min ("+str(end_movie)+"s) after OT with interval "+str(interval*timestep)+"s.")
        else:
            print("Movie will start at "+str(start_movie)+"s after OT and end at "+str(end_movie)+"s after OT with interval "+str(interval*timestep)+"s.")
        
    if plot_3c:
        print("Ready to plot 3C figures!")
    else:
        print("Ready to plot (vertical component only)!")  
        
    if plot_save:
        print("Plots will be saved in " + movie_directory)
    else:
        print("Plots will not be saved.")
        
    print("Plotting GMV and crosssections instead of regular seismograms...")
    
    for it in timeframes: # from start time to end time 
        if it == start_movie_index or (start+it*timestep) % 500 == 0:
            print("Plotting %06.1f s..."%(start+it*timestep))
        # Setting up the plot
        fig = plt.figure(figsize=(10, 9)) 
        gs_Xsec=GridSpec(5,1, height_ratios=[4,0.2,0.55,0.55,0.55])
        gs_Xsec.update(hspace=0.1)
        ax1_Xsec = plt.subplot(gs_Xsec[0])
        axs1_Xsec = plt.subplot(gs_Xsec[1])
        axs1_Xsec.set_visible(False)
        ax2_Xsec = plt.subplot(gs_Xsec[2])
        ax3_Xsec = plt.subplot(gs_Xsec[3])
        ax4_Xsec = plt.subplot(gs_Xsec[4])
        
        # Setting up the map
        m_Xsec = Basemap(projection='mill',llcrnrlat=AA_lat1,urcrnrlat=AA_lat2,llcrnrlon=AA_lon1,urcrnrlon=AA_lon2,resolution="i",ax=ax1_Xsec)
        m_Xsec.drawcoastlines()
        m_Xsec.drawmapboundary(fill_color='lightblue')
        m_Xsec.fillcontinents(color='lightyellow',lake_color='lightblue')
        m_Xsec.drawcountries(color="lightgrey")
        
        # Draw parallels and meridians.
        parallels = np.arange(AA_lat1,AA_lat2+dlat,dlat)
        # Label the meridians and parallels
        m_Xsec.drawparallels(parallels,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        # Draw Meridians and Labels
        meridians = np.arange(AA_lon1,AA_lon2+dlon,dlon)
        m_Xsec.drawmeridians(meridians,labels=[True,False,False,True], linewidth=1.0, fontsize=12)
        
        # Plot the stations
        # Plot 3C motion
        if plot_3c:
            x_sta_Xsec, y_sta_Xsec = m_Xsec(lon_sta_new+GMV["GMV_E"][:,it], lat_sta_new+GMV["GMV_N"][:,it])
            alpmap_Xsec = m_Xsec.scatter(x_sta_Xsec, y_sta_Xsec, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1_Xsec)
        # Plot vertical motion only
        else:
            x_sta_Xsec, y_sta_Xsec = m_Xsec(lon_sta_new, lat_sta_new)
            alpmap_Xsec  = m_Xsec.scatter(x_sta_Xsec, y_sta_Xsec, c=GMV["GMV_Z"][:,it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=3, ax=ax1_Xsec)
        
        # Mark the cross-section stations
        alpmap_sta_Xsec  = m_Xsec.plot(x_sta_Xsec[cross_sec], y_sta_Xsec[cross_sec], fillstyle='none', markeredgecolor="lime", markeredgewidth=2.5, marker="o", markersize=8, zorder=4, ax=ax1_Xsec, linestyle="None")
        # Mark the reference station
        alpmap_sta_Xsec2  = m_Xsec.scatter(x_sta_Xsec[thechosenone_Xsec["sta_index"]], y_sta_Xsec[thechosenone_Xsec["sta_index"]], c=GMV["GMV_Z"][thechosenone_Xsec["sta_index"],it], edgecolors='k', marker='o', s=45, cmap='bwr', vmin=vmin, vmax=vmax, zorder=5, ax=ax1_Xsec)
        alpmap_sta_Xsec2  = m_Xsec.plot(x_sta_Xsec[thechosenone_Xsec["sta_index"]], y_sta_Xsec[thechosenone_Xsec["sta_index"]]+1, fillstyle='none', markeredgecolor="magenta", markeredgewidth=3, marker="o", markersize=8, zorder=6, ax=ax1_Xsec, linestyle="None")
    
        mcolorbar(alpmap_Xsec, vmin, vmax) # Plot map colorbar
        
        ax1_Xsec.set_title("%04d"% event_dic["year"]+'/'+"%02d"% event_dic["month"] +'/'+"%02d"% event_dic["day"]+' '+
                            "%02d"% event_dic["hour"] +':'+"%02d"% event_dic["minute"] +':'+"%02d"% event_dic["second"]+
                            ' '+event_dic["mag_type"].capitalize()+' '+"%.1f"% event_dic["mag"]+' '+string.capwords(event_dic["region"])+'\n'+
                            '  Lat '+"%.2f"% event_dic["lat"] +' Lon '+"%.2f" % event_dic["lon"]+', Depth '+ "%.1f"% event_dic["depth"]+'km'+
                            ', Distance '+ "%.1f"% np.median(GMV["dist_sta"])+'\N{DEGREE SIGN}, '+str(len(name_sta_new))+' STA', fontsize=14)
        
        # Add AA logo on plot
        imagebox_Xsec = OffsetImage(arr_img, zoom=0.45)
        imagebox_Xsec.image.axes = ax1_Xsec
        ab_Xsec = AnnotationBbox(imagebox_Xsec, (1, 1),
                            xybox=(40., 290.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5, frameon=False)
        
        ax1_Xsec.add_artist(ab_Xsec)
        
        # Plot cross-sections        
        if Xsec_dict["facenorth"]: # If North is on the left hand side
            area1  = (45-(GMV["GMV_E"][:,it][cross_sec]*250))
            area1[area1 < 15.] = 15.
            area11 = (45-(GMV["GMV_E"][:,it][thechosenone_Xsec["sta_index"]]*250))
            area11 = 15. if area11 < 15. else area11
            area2  = (45-(GMV["GMV_T"][:,it][cross_sec]*250))
            area2[area2 < 15.] = 15.
            area22 = (45-(GMV["GMV_T"][:,it][thechosenone_Xsec["sta_index"]]*250))
            area22 = 15. if area22 < 15. else area22
            # ZNE
            ax2_Xsec.scatter(dist_sta_Xsec-GMV["GMV_N"][:,it][cross_sec], GMV["GMV_Z"][:,it][cross_sec], c=GMV["GMV_Z"][:,it][cross_sec], edgecolors='k', marker='o', s=area1, cmap='bwr', vmin=vmin, vmax=vmax)
            # Mark reference station
            ax2_Xsec.scatter(GMV["dist_sta"][thechosenone_Xsec["sta_index"]]-GMV["GMV_N"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], c=GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], edgecolors="k", linewidths=3, marker="o", s=area11, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4)
            ax2_Xsec.scatter(GMV["dist_sta"][thechosenone_Xsec["sta_index"]]-GMV["GMV_N"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], facecolors='none', edgecolors="magenta", linewidths=3, marker="o", s=area11, zorder=5, linestyle="None")
            # ZRT
            ax3_Xsec.scatter(dist_sta_Xsec+GMV["GMV_R"][:,it][cross_sec], GMV["GMV_Z"][:,it][cross_sec], c=GMV["GMV_Z"][:,it][cross_sec], edgecolors='k', marker='o', s=area2, cmap='bwr', vmin=vmin, vmax=vmax)
            # Mark reference station
            ax3_Xsec.scatter(GMV["dist_sta"][thechosenone_Xsec["sta_index"]]+GMV["GMV_R"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], c=GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], edgecolors="k", linewidths=3, marker="o", s=area22, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4)
            ax3_Xsec.scatter(GMV["dist_sta"][thechosenone_Xsec["sta_index"]]+GMV["GMV_R"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], facecolors='none', edgecolors="magenta", linewidths=3, marker="o", s=area22, zorder=5, linestyle="None")
        else: # If North is on the right hand side
            area1  = (45+(GMV["GMV_E"][:,it][cross_sec]*250))
            area1[area1 < 15.] = 15.
            area11 = (45+(GMV["GMV_E"][:,it][thechosenone_Xsec["sta_index"]]*250))
            area11 = 15. if area11 < 15. else area11
            area2  = (45+(GMV["GMV_T"][:,it][cross_sec]*250))
            area2[area2 < 15.] = 15.
            area22 = (45+(GMV["GMV_T"][:,it][thechosenone_Xsec["sta_index"]]*250))
            area22 = 15. if area22 < 15. else area22
            # ZNE
            ax2_Xsec.scatter(dist_sta_Xsec-GMV["GMV_N"][:,it][cross_sec], GMV["GMV_Z"][:,it][cross_sec], c=GMV["GMV_Z"][:,it][cross_sec], edgecolors='k', marker='o', s=area1, cmap='bwr', vmin=vmin, vmax=vmax)
            # Mark reference station
            ax2_Xsec.scatter(thechosenone_Xsec["sta_dist"]+GMV["GMV_N"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], c=GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], edgecolors="k", linewidths=3, marker="o", s=area11, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4)
            ax2_Xsec.scatter(thechosenone_Xsec["sta_dist"]+GMV["GMV_N"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], facecolors='none', edgecolors="magenta", linewidths=3, marker="o", s=area11, zorder=5,linestyle="None")
            # ZRT
            ax3_Xsec.scatter(dist_sta_Xsec+GMV["GMV_R"][:,it][cross_sec], GMV["GMV_Z"][:,it][cross_sec], c=GMV["GMV_Z"][:,it][cross_sec], edgecolors='k', marker='o', s=area2, cmap='bwr', vmin=vmin, vmax=vmax)
            # Mark reference station
            ax3_Xsec.scatter(thechosenone_Xsec["sta_dist"]-GMV["GMV_R"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], c=GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], edgecolors="k", linewidths=3, marker="o", s=area22, cmap='bwr', vmin=vmin, vmax=vmax, zorder=4)
            ax3_Xsec.scatter(thechosenone_Xsec["sta_dist"]-GMV["GMV_R"][:,it][thechosenone_Xsec["sta_index"]], GMV["GMV_Z"][:,it][thechosenone_Xsec["sta_index"]], facecolors='none', edgecolors="magenta", linewidths=3, marker="o", s=area22, zorder=4,linestyle="None")
            
        ax2_Xsec.set_xlim([int(min(dist_sta_Xsec)-1.5), int(max(dist_sta_Xsec)+1.5)])
        ax2_Xsec.set_xticklabels(Xsec_dict["dist_label_Xsec"])
        ax2_Xsec.tick_params(axis="x",labelsize=11)
        ax2_Xsec.xaxis.tick_top()
        ax2_Xsec.xaxis.set_label_position('top') 
        ax2_Xsec.set_ylim([-1.1, 1.1])
        ax2_Xsec.grid(b=True, which='major')
        
        ax3_Xsec.xaxis.tick_top()
        ax3_Xsec.set_xticklabels([])
        ax3_Xsec.set_xlim([int(min(dist_sta_Xsec)-1.5), int(max(dist_sta_Xsec)+1.5)])
        ax3_Xsec.set_ylim([-1.1, 1.1])
        ax3_Xsec.grid(b=True, which='major')
        ax3_Xsec.set_ylabel('Normalized Displacement', fontsize=11)
        
        if not Xsec_dict["facenorth"]:
            ax2_Xsec.invert_xaxis()
            ax3_Xsec.invert_xaxis()
        
        # Annotation of cross section direction
        x, y, arrow_length = 0.07, 0.2, 0.04   
        ax3_Xsec.annotate('', xy=(x+arrow_length, y), xytext=(x, y),
                          arrowprops=dict(facecolor='black', width=1, headwidth=4),
                          ha='center', va='center', fontsize=9,
                          xycoords=ax3_Xsec.transAxes)
        ax2_Xsec.text(0.02, 0.2, "N", transform=ax2_Xsec.transAxes, fontsize=11, ha='center', va='center')     # left direction ZNE
        ax2_Xsec.text(0.98, 0.2, "S", transform=ax2_Xsec.transAxes, fontsize=11, ha='center', va='center')     # right direction ZNE
        ax3_Xsec.text(0.035, 0.2, Xsec_dict["leftDir"], transform=ax3_Xsec.transAxes, fontsize=11, ha='center', va='center') # left direction ZRT
        ax3_Xsec.text(0.975, 0.2, Xsec_dict["rightDir"],transform=ax3_Xsec.transAxes, fontsize=11, ha='center', va='center') # right direction ZRT
        
        # Cross section label
        ax2_Xsec.text(0.03, 0.8, "ZNE",transform=ax2_Xsec.transAxes, fontsize=11, ha='center', va='center')
        ax3_Xsec.text(0.03, 0.8, "ZRT",transform=ax3_Xsec.transAxes, fontsize=11, ha='center', va='center')
        
        # Plot selected reference vertical seismogram
        # HHZ
        ax4_Xsec.plot((time_st+start)/d_time, GMV["GMV_Z"][thechosenone_Xsec["sta_index"]]/maxabs(GMV["GMV_Z"][thechosenone_Xsec["sta_index"]]), color="k", linewidth=1.5, label=thechosenone_Xsec["sta_name"]+".Z")
        ax4_Xsec.axvline(x=(start+it*timestep)/d_time, color="r", linewidth=1.2) # Time marker
        # Plot phase marker for channel Z
        phase_marker(thechosenone_Xsec["arr"], ax4_Xsec, "Z", start, end)
        
        ax4_Xsec.grid(b=True, which='major')
        ax4_Xsec.tick_params(axis="x",labelsize=11)
        if timelabel == "hr":
            ax4_Xsec.set_xlabel('Time after origin [hr]', fontsize=14)
            ax4_Xsec.xaxis.set_minor_locator(MultipleLocator(0.1))
        elif timelabel == "min":
            ax4_Xsec.set_xlabel('Time after origin [min]', fontsize=14)
        else:
            ax4_Xsec.set_xlabel('Time after origin [s]', fontsize=14)
        ax4_Xsec.set_xlim([start/d_time,end/d_time])
        ax4_Xsec.set_ylim([-1.1,1.1]) 
        ax4_Xsec.set_xticks(seismo_labels)
        ax4_Xsec.set_xticklabels(seismo_labels) 
        ax4_Xsec.legend(loc='lower right', fontsize=8, bbox_to_anchor=(0.99, -0.07)) 
            
        plt.tight_layout(h_pad=2)
    
        if plot_save:
            if plot_3c:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_3C_Xsec_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
            else:
                plt.savefig(movie_directory+ event_dic['event_name'] +"_Xsec_"+ "%07.1f"%(start+it*timestep)+"s."+save_option, dpi=save_dpi)
            plt.clf()
            plt.close()       
        else:
            plt.show()
    if plot_save:
        print("Plots are saved. End of plotting.")
    else:
        print("Plots are not saved. End of plotting.")
    