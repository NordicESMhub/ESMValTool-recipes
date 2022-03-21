#! /usr/bin/env python3
# This script is for extracting specific transcom (TC) region data from several sources
import os
import numpy as np
import datetime as dt
from glob import glob
from calendar import monthrange as mr
from netCDF4 import Dataset
from scipy.interpolate import griddata

# Directories
base_dir  = '/home/makelaj/data/'
outputdir = base_dir + 'TC_extracts/generated/'

# Regions files
regions_file = base_dir + 'TC_extracts/regions_esticc2_201801_wcoords.nc'
regions = Dataset(regions_file,'r')
tc_regions = regions['transcom_regions'][:]
tc_lat     = regions['latitude'][:]
tc_lon     = regions['longitude'][:]
print(tc_lat[0],tc_lat[-1])
print(tc_lon[0],tc_lon[-1])
tc_extract = [16,29]


# Data sources to be extracted
#datadirs = ['CH4','CMPI6','JASMIN']
datadirs = ['CH4']
datafiles = []
for ddir in datadirs:
    datafiles += glob(base_dir + ddir + '/*wetlandCH4*.nc')

    
def interpolate_2d_data(data, lats, lons, target_lats, target_lons):
    """Interpolates the data values to a specific lat/lon grid.
    This function should only be used for 2D arrays (no time indeces etc.)"""

    # First check if the coordinates are the same, otherwise interpolate
    if (np.array_equal(lats, target_lats) and np.array_equal(lons, target_lons)):
        return data

    else:
        grid_lats, grid_lons = np.meshgrid(target_lats, target_lons)
        datapoints = np.zeros((len(lats) * len(lons), 2))
        datavalues = data.reshape(-1)
        i = 0
        for lat in lats:
            datapoints[i:i + len(lons), 0] = lat
            datapoints[i:i + len(lons), 1] = lons
            i += len(lons)

        # Calculate the values on the new grid
        gridvalues = griddata(datapoints, datavalues, (grid_lats, grid_lons),
                              method='linear')
        nearest = griddata(datapoints, datavalues, (grid_lats, grid_lons),
                           method='nearest')

        # In some cases griddata cannot export values on the border of the grid
        for i in range(gridvalues.shape[0]):
            for j in range(gridvalues.shape[1]):
                if (np.isnan(gridvalues[i, j])):
                    gridvalues[i, j] = nearest[i, j]
        return gridvalues.T

    
def get_model_info(dfile):
    filename = dfile.split('/')[-1]
    if filename == 'created_wetlandCH4-RF-DYPTOP.nc':
        mname = 'RF-DYPTOP'
        year  = 2013
        month = 1
        sind  = 0
    elif filename == 'created_wetlandCH4-RF-GLWD.nc':
        mname = 'RF-GLWD'
        year  = 2013
        month = 1
        sind  = 0
    elif filename == 'created_wetlandCH4-RF-mean.nc':
        mname = 'RF-mean'
        year  = 2013
        month = 1
        sind  = 0
    elif filename == 'created_wetlandCH4-RF-PEATMAP.nc':
        mname = 'RF-PEATMAP'
        year  = 2013
        month = 1
        sind  = 0
    elif filename == 'Inversion-wet-mean_wetlandCH4_2000-2015.nc':
        mname = 'Inversion'
        year  = 2000
        month = 1
        sind  = 0
    elif filename == 'Inversion-wet-soil-mean_wetlandCH4_2000-2015.nc':
        mname = 'Inversion-with-soil'
        year  = 2000
        month = 1
        sind  = 0
    elif filename == 'wetlandCH4_Emon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc':
        mname = 'CESM2'
        year  = 2000
        month = 1
        sind  = 1680
    elif filename == 'wetlandCH4_Emon_CESM2-WACCM_historical_r1i1p1f1_gn_185001-201412.nc':
        mname = 'CESM2-WACCM'
        year  = 2000
        month = 1
        sind  = 1680
    elif filename == 'wetlandCH4_Emon_NorESM2-LM_historical_r1i1p1f1_gn_200001-201412.nc':
        mname = 'NorESM2-LM'
        year  = 2000
        month = 1
        sind  = 0
    elif filename == 'wetlandCH4_Emon_NorESM2-MM_historical_r1i1p1f1_gn_200001-201412.nc':
        mname = 'NorESM2-MM'
        year  = 2000
        month = 1
        sind  = 0
    elif filename == 'wetlandCH4_Emon_UKESM1-0-LL_historical_r2i1p1f2_gn_195001-201412.nc':
        mname = 'UKESM1-0-LL'
        year  = 2000
        month = 1
        sind  = 600
    elif filename == 'CLM45.CMCC_S3_wetlandCH4.nc':
        mname = 'CLM45'
        year  = 2000
        month = 1
        sind  = 1200
    elif filename == 'CLM5_S3_wetlandCH4.nc':
        mname = 'CLM5'
        year  = 2000
        month = 1
        sind  = 1200
    elif filename == 'JULES_S3_wetlandCH4.nc':
        mname = 'JULES'
        year  = 2000
        month = 1
        sind  = 1680
    elif filename == 'wetlandCH4_Emon_LPX_c4mip_r1i1p1_200001-201212.nc':
        mname = 'LPX'
        year  = 2000
        month = 1
        sind  = 0
    elif filename == 'wetlandCH4_Emon_LPX-dyptop_c4mip_r1i1p1_199901-201412.nc':
        mname = 'LPX-DYPTOP'
        year  = 2000
        month = 1
        sind  = 12
    elif filename == 'created_LPJ-GUESS_wetlandCH4_1901-2015.nc':
        mname = 'LPJ-GUESS'
        year  = 2000
        month = 1
        sind  = 1188
    else:
        print("unknown",filename)
        mname = 'XXX'
        year  = 2000
        month = 1
        sind  = 0
    return mname, year, month, sind


def calculate_grid_cell_areas(lats, lons):
    """Calculates the area of each gridcell in m2"""
    R = 6371
    grid = np.zeros((len(lats),len(lons)))
    res = np.abs(lats[0] - lats[1])
    dlat = res*0.5
    for i in range(len(lats)):
        grid[i,:] = np.abs( np.sin(np.pi*(lats[i] + dlat)/180.) -
                            np.sin(np.pi*(lats[i] - dlat)/180.)) * res
    grid = 1E6*R**2 * np.pi * grid / 180 
    return grid


for dfile in datafiles:
    data = Dataset(dfile,'r')
    mname, year, month, sind = get_model_info(dfile)
    print("Starting for",mname)
    # Extract variables
    time = data['time'][:]
    #tunits = data.variables['time'].units
    #get_start_time(time,tunits)
    
    try:
        dlat = data['lat'][:]
    except:
        dlat = data['latitude'][:]
    try:
        dlon = data['lon'][:]
    except:
        dlon = data['longitude'][:]
    try:
        wch4 = data['wetlandCH4'][sind:]
        units = data.variables['wetlandCH4'].units
    except:
        wch4 = data['fch4_wetl_npp'][sind:]
        units = data.variables['fch4_wetl_npp'].units
    print(mname,"; data length:",len(wch4),"; original units:",units)
    wch4 = np.ma.masked_invalid(wch4)
    np.ma.set_fill_value(wch4,0)

    if mname in ['UKESM1-0-LL','CESM2','CESM2-WACCM']:
        fracfile = dfile.replace("wetlandCH4","wetlandFrac")
        fracdata = Dataset(fracfile,'r')
        frac = fracdata['wetlandFrac'][sind:]
        fracdata.close()
    else:
        frac = np.ones(wch4.shape)

    # Reorient TC map to data so basically chech lat orientation and lon base
    # More efficient to modify the mask than the whole data cube
    mlat = tc_lat.copy()
    mlon = tc_lon.copy()
    tc_reg = tc_regions.copy()
    if (dlat[0] > dlat[-1]):
        # reverse lat orientation
        mlat = mlat[::-1]
        tc_reg = tc_reg[::-1,:]
    if (dlon[0] > 0):
        # convert [-180,180] to [0,360]
        i = np.int(len(mlon)/2)
        # first the coordinates
        temp = mlon[:i].copy()
        mlon[:i] = mlon[i:]
        mlon[i:] = 360+temp[:]
        # then the tc_reg
        temp = tc_reg[:,:i].copy()
        tc_reg[:,:i] = tc_reg[:,i:]
        tc_reg[:,i:] = temp[:,:]
    # print("\n",dfile,len(dlat),len(dlon))
    # print(dlat[0],dlat[-1],dlon[0],dlon[-1])
    # print(mlat[0],mlat[-1],mlon[0],mlon[-1])

    for reg in tc_extract:
        # Generate mask per required area
        mask = tc_reg.copy() - reg
        mask[(mask < -0.5)] = 1
        mask[(mask >  0.5)] = 1
        # use defined interpolation function and mask ambiguous points
        nmask = interpolate_2d_data(mask,mlat,mlon,dlat,dlon)
        nmask[(nmask > 0.1)] = 1
        area = calculate_grid_cell_areas(dlat,dlon)
        scaled_area = (1-nmask)*area
        # And initialize out data
        outd = np.zeros((wch4.shape[0],3))
        for m in range(len(outd)):
            outd[m,0] = np.int(year + np.int(m/12))
            outd[m,1] = np.int(month + m%12)
            if (outd[m,1] in [1,3,5,7,8,10,12]):
                mlength = 31
            elif (outd[m,1] in [4,6,9,11]):
                mlength = 30
            elif (outd[m,0]%4 == 0):
                mlength = 29
            else:
                mlength = 28
            # kg /m2 /s -> Tg / area /month
            scale = 1E-9 * mlength * 24*60*60
            #outd[m,2] = np.sum(np.ma.masked_array(wch4[m,:,:],nmask)*area) * scale
            outd[m,2] = np.sum(wch4[m,:,:]*frac[m,:,:]*scaled_area) * scale
        np.savetxt(base_dir + '/generated/' + mname + '_TC-'+ str(reg) + '.csv',outd,delimiter=',')
    data.close()


# def get_start_time(time,tstr):
#     # Extract base year 
#     inc = tstr.split()[0]
#     base = tstr.split()[2]
#     y0,m0,d0 = base.split('-')
#     y0 = np.int(y0)
#     m0 = np.int(m0)
#     d0 = np.int(d0)
#     ### Then to check first 
#     basedate = dt.date(y0,m0,d0)
#     if inc == 'seconds':
#         d1 = basedate + dt.timedelta(seconds=np.int(time[0]))
#     elif inc == 'days':
#         d1 = basedate + dt.timedelta(days=np.int(time[0]))
#     else:
#         print("XXXXXX","\n",inc)
#     if (d1.year < 2000):
#         delta = dt.date(2000,1,1) - basedate        
#     #print(d1.year, d1.month)
#     print(len(time))
