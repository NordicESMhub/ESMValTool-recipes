#! /usr/bin/env python
# This script is for plotting inversion model timeseries
import os
import numpy as np
from glob import glob
from calendar import monthrange as mr
from netCDF4 import Dataset

# Where everything is
base_dir  = '/home/makelaj/data/ch4_olli/'
outputdir = base_dir

infiles = glob(base_dir + 'RF-*.nc')
infiles.sort()
for f in infiles:
    if 'RF-mean' in f:
        infiles.remove(f)

def generate_months(byear,syear,years):
    """Generates the desired months with respect to starting year"""
    time = np.zeros(12*years)
    for y in range(years):
        time[12*y:12*(y+1)] = 365*y + np.array([12, 43, 71, 102, 132, 163, 193, 224, 255, 285, 316, 346])
    time = time + (syear-byear)*365
    return time

def scale_data(data, syear):
    """Scales the input array by month lenghs starting from january of start year"""
    scale = np.zeros(data.shape[0])
    for m in range(len(scale)):
        mo = np.int(m%12+1)
        scale[m] = 1E-3/(mr(np.int(syear+np.floor(m/12)),mo)[1]*24*60*60)
    # input is g(CH4) m-2 month-1
    # output is kg(CH4) m-2 s-1
    for m in range(len(scale)):
        data[m,:,:] = data[m,:,:]*scale[m]
    return data

# Create known helper variables
byear = 1900
syear = 2013
years = 2
ivar  = 'FCH4'
var   = 'wetlandCH4'
mtime = generate_months(byear,syear,years)
mtime = mtime.astype(np.int32)


for model in infiles:
    print("Reformatting", model)
    # Read in the values and format
    name = model.split('/')[-1]
    data = Dataset(model,'r')
    fch4 = data.variables[ivar][:]
    ilat = data.variables['lat'][:]
    ilon = data.variables['lon'][:]
    data.close()
    res  = 360/fch4.shape[2]
    lats = np.linspace(-90+res/2, 90-res/2, np.int(180./res), endpoint=True)
    lons = np.linspace(-180+res/2, 180-res/2, np.int(360./res), endpoint=True)

    # Creating output
    outfile = outputdir + 'created_' + var + '-' + name
    out = Dataset(outfile,'w')
    setattr(out,'title',var)
    setattr(out,'model',name.split('.')[0])
    setattr(out,'simulation','CRESCENDO')
    setattr(out,'variable',var)
    setattr(out,'versionNumber',1)

    # Set file dimensions and attributes
    out.createDimension('longitude', len(lons))
    out.createDimension('latitude', len(lats))
    out.createDimension('time', len(mtime))
    out.createVariable('longitude','d',('longitude',))
    out.createVariable('latitude','d',('latitude',))
    out.createVariable('time','i',('time',))

    lon     = out.variables['longitude']
    lat     = out.variables['latitude']
    time    = out.variables['time']
    lon[:]  = lons[:]
    lat[:]  = lats[:]
    time[:] = mtime[:]
    
    sdays = 'days since ' + str(np.int(byear))  + '-01-01'
    setattr(lon, 'units', 'degrees_east')
    setattr(lon, 'long_name', 'longitude')
    setattr(lat, 'units', 'degrees_north')
    setattr(lat, 'long_name', 'latitude')
    setattr(time, 'units', sdays)
    setattr(time, 'calendar', '365_day')
    setattr(time, 'long_name', sdays)

    # Create the desired variable
    out.createVariable(var,'d',('time', 'latitude', 'longitude'))
    values = out.variables[var]
    setattr(values, 'units', 'kg m-2 s-1')
    setattr(values, 'long_name', 'wetland_methane_emissions')

    # Check if the order is normal
    if ilat[-1]-ilat[0] > 0:
        values[:,-fch4.shape[1]:,:] = scale_data(fch4, syear)
    else:
        values[:,-fch4.shape[1]:,:] = scale_data(fch4[:,::-1,:], syear)
    
    out.close()
    print("Created file:", outfile)
