#! /usr/bin/env python
# This script is for plotting ch4 maps and timeseries
import os
import sys
import numpy as np
from glob import glob
from datetime import date
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from numpy import tile
from calendar import monthrange as mr
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline as inter

# Where everything is
base_dir     = '/home/makelaj/data/ESGF/'
output_dir   = '/home/makelaj/data/images/'

# general settings
lat_bound = 45
lat_min   = 50
ifmt      = '.png'
# Map settings
mmin      = -1+1
mmax      = 11+1
mlevs     = np.linspace(mmin,mmax,7)
dmin      = -4.5
dmax      = 4.5
dlevs     = np.linspace(dmin,dmax,7)
near0     = 0.1

# kg m-2 s-1 -> g m-2 year-1
scale = 1E3*(365*24*60*60)

all_models = ['CLM45','CLM5','JULES','LPJ-GUESS','InvEnsMean', 'RF-DYPTOP', 'RF-GLWD', 'RF-PEATMAP']
ok_models  = ['CLM45','CLM5','JULES','LPJ-GUESS']
obs_models = ['InvEnsMean', 'RF-DYPTOP', 'RF-GLWD', 'RF-PEATMAP']
map_obs    = ['InvEnsMean', 'RF-mean']


def calculate_map_data(lats,lons,data,blat):
    """returns the averaged data and correct lats and lons"""
    if lats[-1] < lats[0]:
        data = data[:,::-1,:]
        lats = lats[::-1]
    lat0 = np.where(lats>blat)[0][0]
    data = data[:,lat0:,:]
    lats = lats[lat0:]
    #Correct the lons at end if needed
    ndata = np.zeros(data.shape[1:])
    for i in range(ndata.shape[0]):
        for j in range(ndata.shape[1]):
            ndata[i,j] = data[:,i,j].mean()
    if lons[-1] > 180:
        lon180 = np.where(lons>180)[0][0]
        lon0   = np.where(lons>=0)[0][0]
        west = np.copy(ndata[:,lon180:])
        east = np.copy(ndata[:,:lon180])
        ndata[:,lon180:] = east
        ndata[:,:lon180] = west
        lons = lons - 180
    return lats, lons, ndata

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

def get_colour(model):
    if model == 'CLM45':
        colour = 'green'
    elif model == 'CLM5':
        colour = 'lime'
    elif model == 'JULES':
        colour = 'blue'
    elif model == 'LPJ-GUESS':
        colour = 'gold'
    elif model == 'InvEnsMean':
        colour = 'purple'
    elif model == 'RF-mean':
        colour = 'red'
    elif model == 'RF-DYPTOP':
        colour = 'brown'
    elif model == 'RF-GLWD':
        colour = 'salmon'
    elif model == 'RF-PEATMAP':
        colour = 'maroon'
    else:
        colour = 'black'
    return colour


def grid_area(lats):
    """Calculates the different lat/lon areas in m2 based on lats.
    Also scales the input from kg -> Tg. """
    R    = 6371*1E3
    res  = lats[1]-lats[0]
    grid = np.zeros((len(lats),np.int(360/res)))
    dlon = res
    dlat = 0.5*res
    for i in range(len(lats)):
        grid[i,:] = np.abs( np.sin(np.pi*(lats[i] + dlat)/180.) -
                            np.sin(np.pi*(lats[i] - dlat)/180.)) * dlon
    grid = R**2 * np.pi * grid / 180 *1E-9
    return grid


if plot_times == True:
    # Timeseries of yearly values
    plt.clf()
    fig = plt.figure(figsize=(10,8))
    models = ok_models + obs_models
    ye = 2013
    for model in models:
        dfile = glob(base_dir + 'wetlandCH4*' + model + '*.nc')
        data  = Dataset(dfile[0],'r')
        try:
            lats  = data.variables['latitude'][:]
        except:
            lats  = data.variables['lat'][:]
        try:
            lons  = data.variables['longitude'][:]
        except:
            lons  = data.variables['lon'][:]
        time  = data.variables['time'][:]
        tu    = data.variables['time'].units.split(' ')[2]
        tu    = tu.split('-')
        d0    = date(np.int(tu[0]),np.int(tu[1]),np.int(tu[2]))
        d1    = date(1999,12,31)
        delta = d1-d0
        # Get data from wrong names and order
        if model == 'JULES':
            time = time / (24*60*60)
            vname = 'fch4_wetl_npp'
            start = np.int(12*(2000-np.int(tu[0])))
        else:
            vname = 'wetlandCH4'
            start = np.where(time > delta.days)[0][0]
        stop  = 12*np.int(len(time[start:])/12)+1
        yend  = np.int(2000+stop/12)
        ye    = np.max([ye,yend])
        if lats[-1]>lats[0]:
            lat0 = np.where(lats>lat_min)[0][0]
            var  = data.variables[vname][start:start+stop,lat0:,:]
            lats = lats[lat0:]
        else:
            lat0 = np.where(lats>lat_min)[0][-1]
            var  = data.variables[vname][start:start+stop,lat0::-1,:]
            lats = lats[lat0::-1]
        # Rearrange lons if we get some errors - now not needed
        data.close()
        
        # For annual plots we want to average the months but sum the longitudes
        narr = np.zeros(var.shape[:2])
        var  = np.ma.masked_array(var)
        var  = np.ma.masked_invalid(var)
        np.ma.set_fill_value(var, 0)
        var = var.filled()
        for m in range(narr.shape[0]):
            for i in range(narr.shape[1]):
                narr[m,i] = np.sum(var[m,i,:])
        # We take only one slice of grid (areas are the same) to get annual values
        ann  = np.zeros(np.int(len(narr)/12))
        grid = grid_area(lats)[:,0]
        for m in range(narr.shape[0]):
            narr[m,:] = narr[m,:]*grid 
        for y in range(len(ann)):
            ann[y] = np.sum(narr[12*y:12*(y+1),:]) 

        # Area slicing etc converts kg m-2 s-1 -> Tg area s-1
        ann = ann * (60*60*24*365)/12
        # Plotting
        colour = get_colour(model)
        if 'RF' in model:
            x = np.linspace(2013,2014,2)
            plt.plot(x,ann,'ro',color=colour,linewidth=2,label=model)
        else:
            x = np.linspace(2000,yend,yend-2000,endpoint=False)
            plt.plot(x,ann,color=colour,linewidth=2,label=model)

        if model == 'InvEnsMean':
            limits = np.load('/home/makelaj/data/ch4_inversions/wetlandCH4-relative_yearly_minmax.npy')
            plt.plot(x,ann*limits[:len(ann),0],'r--',color=colour,linewidth=1)
            plt.plot(x,ann*limits[:len(ann),1],'r--',color=colour,linewidth=1)

    plt.legend(loc=1)
    plt.title('Annual CH4 fluxes from [' + str(lat_min) +'N:90N]')
    plt.ylabel('Tg', rotation=0)
    ax = plt.gca()
    ax.yaxis.set_label_coords(-0.02,1)
    ax.set_xlim([2000,2015])
    plt.xlabel('year')
    plt.grid(linestyle='dotted')
    outf = output_dir + 'CH4_timeseries_2000-' + str(ye) + ifmt
    plt.savefig(outf)
    print("Created image:",outf)


if plot_annual == True:
    # Monthly averages (calculate month length)
    mscale = np.zeros(12)
    for m in range(12):
        mscale[m] = mr(2001,m+1)[1]
    x = np.linspace(1,12,12)
    obsmodels = map_obs
    fig,axs = plt.subplots(1,2, figsize=(15,8))
    for omodel in obsmodels:
        col = obsmodels.index(omodel)
        ofile = glob(base_dir + 'wetlandCH4*' + omodel + '*.nc')
        odata  = Dataset(ofile[0],'r')
        try:
            olats  = odata.variables['latitude'][:]
        except:
            olats  = odata.variables['lat'][:]
        try:
            olons  = odata.variables['longitude'][:]
        except:
            olons  = odata.variables['lon'][:]
        
        # First calculate needed
        eyear = 2014
        if omodel == 'InvEnsMean':
            syear = 2000
        else:
            syear = 2013

        if olats[-1]>olats[0]:
            olat0 = np.where(olats>lat_min)[0][0]
            obs   = odata.variables['wetlandCH4'][:12*(eyear+1-syear),olat0:,:]
            olats = olats[olat0:]
        else:
            olat0 = np.where(olats>lat_min)[0][-1]
            obs   = odata.variables['wetlandCH4'][:12*(eyear+1-syear),olat0::-1,:]
            olats = lats[olat0::-1]
        odata.close()

        obs = np.ma.masked_array(obs)
        obs = np.ma.masked_invalid(obs)
        np.ma.set_fill_value(obs, 0)
        obs = obs.filled()

        # We want monthly values over the whole grid (remember grid cell areas)
        nobs = np.zeros(obs.shape[:2])
        for m in range(nobs.shape[0]):
            for i in range(nobs.shape[1]):
                nobs[m,i] = np.sum(obs[m,i,:])
        # We take only one slice of grid (areas are the same) to get annual values
        oann  = np.zeros(12)
        ogrid = grid_area(olats)[:,0]
        for m in range(nobs.shape[0]):
            nobs[m,:] = nobs[m,:]*ogrid 
        for y in range(len(oann)):
            oann[y] = np.sum(nobs[y::12,:])
        # Area slicing etc converts kg m-2 s-1 -> Tg area s-1
        # Above we sum over all grid cells but would average the months
        oann = oann * (60*60*24) * mscale / len(nobs[::12])

        # Obsplotting
        ocolour = get_colour(omodel)
        axs[col].plot(x,oann,color=ocolour,linewidth=2,label=omodel)
        axs[col].set_title('Monthly CH4 averages from [' +
                           str(lat_min) + 'N:90N] for ' +
                           str(syear) + '-' + str(eyear))
        axs[col].set_xlabel('month')
        axs[col].grid(linestyle='dotted')
        if omodel == 'InvEnsMean':
            limits = np.load('/home/makelaj/data/ch4_inversions/wetlandCH4-relative_annual_minmax.npy')
            axs[col].plot(x,oann*limits[:,0],'r--',color=colour,linewidth=1)
            axs[col].plot(x,oann*limits[:,1],'r--',color=colour,linewidth=1)
        
        models = ok_models
        for model in models:
            dfile = glob(base_dir + 'wetlandCH4*' + model + '*.nc')
            data  = Dataset(dfile[0],'r')
            try:
                lats  = data.variables['latitude'][:]
            except:
                lats  = data.variables['lat'][:]
            try:
                lons  = data.variables['longitude'][:]
            except:
                lons  = data.variables['lon'][:]
            time  = data.variables['time'][:]
            tu    = data.variables['time'].units.split(' ')[2]
            tu    = tu.split('-')

            d0    = date(np.int(tu[0]),np.int(tu[1]),np.int(tu[2]))
            d1    = date(syear,12,31)
            delta = d1-d0
        
            if model == 'JULES':
                time = time / (24*60*60)
                vname = 'fch4_wetl_npp'
                if omodel == 'InvEnsMean':
                    start = np.int(12*(2012-np.int(tu[0])))
                else:
                    start = np.int(12*(2012-np.int(tu[0])))
            else:
                vname = 'wetlandCH4'
                start = np.where(time > delta.days)[0][0]
            stop = np.int(1+12*(eyear+1-syear))

            if lats[-1]>lats[0]:
                lat0 = np.where(lats>lat_min)[0][0]
                var  = data.variables[vname][start:start+stop,lat0:,:]
                lats = lats[lat0:]
            else:
                lat0 = np.where(lats>lat_min)[0][-1]
                var  = data.variables[vname][start:start+stop,lat0::-1,:]
                lats = lats[lat0::-1]
            data.close()
            
            var = np.ma.masked_array(var)
            var = np.ma.masked_invalid(var)
            np.ma.set_fill_value(var, 0)
            var = var.filled()

            # We want monthly values over the whole grid (remember grid cell areas)
            narr = np.zeros(var.shape[:2])
            for m in range(narr.shape[0]):
                for i in range(narr.shape[1]):
                    narr[m,i] = np.sum(var[m,i,:])
            # We take only one slice of grid (areas are the same) to get annual values
            ann  = np.zeros(12)
            grid = grid_area(lats)[:,0]
            for m in range(narr.shape[0]):
                narr[m,:] = narr[m,:]*grid 
            for y in range(len(ann)):
                ann[y] = np.sum(narr[y::12,:])
            
            # Area slicing etc converts kg m-2 s-1 -> Tg area s-1
            ann = ann * (60*60*24) * mscale / len(narr[::12])
            
            # Plotting
            colour = get_colour(model)
            axs[col].plot(x,ann,color=colour,linewidth=2,label=model)
            axs[col].set_ylim([-0.25,16.0])
            axs[col].set_xlim([1,12])
            axs[col].legend()
        
    outf = output_dir + 'CH4_monthly_average' + ifmt
    plt.savefig(outf)
    print("Created image:",outf)

    
# Then map plots
if plot_maps:
    obsmodels = map_obs
    for omodel in obsmodels:
        ofile = glob(base_dir + 'wetlandCH4*' + omodel + '*.nc')
        odata  = Dataset(ofile[0],'r')
        try:
            olats  = odata.variables['latitude'][:]
        except:
            olats  = odata.variables['lat'][:]
        try:
            olons  = odata.variables['longitude'][:]
        except:
            olons  = odata.variables['lon'][:]
        
        # First calculate needed
        eyear = 2014
        if omodel == 'InvEnsMean':
            syear = 2000
        else:
            syear = 2013
        obs = odata.variables['wetlandCH4'][:12*(eyear+1-syear),:,:]
        odata.close()
        
        obs = np.ma.masked_array(obs)
        obs = np.ma.masked_invalid(obs)
        np.ma.set_fill_value(obs, 0)
        obs = obs.filled()
        olats,olons,omap = calculate_map_data(olats,olons,obs,lat_bound)
        omap = omap*scale
        #print(omodel,olats[0],olats[-1],olons[0],olons[-1])

        #models = ok_models
        models = ok_models + obsmodels
        models.remove(omodel)
        for model in models:
            dfile = glob(base_dir + 'wetlandCH4*' + model + '*.nc')
            data  = Dataset(dfile[0],'r')
            try:
                lats  = data.variables['latitude'][:]
            except:
                lats  = data.variables['lat'][:]
            try:
                lons  = data.variables['longitude'][:]
            except:
                lons  = data.variables['lon'][:]
            time  = data.variables['time'][:]
            tu    = data.variables['time'].units.split(' ')[2]
            tu    = tu.split('-')

            d0    = date(np.int(tu[0]),np.int(tu[1]),np.int(tu[2]))
            d1    = date(syear,12,31)
            delta = d1-d0
        
            if model == 'JULES':
                time = time / (24*60*60)
                vname = 'fch4_wetl_npp'
                start = np.int(12*(2012-np.int(tu[0])))
            else:
                vname = 'wetlandCH4'
                start = np.where(time > delta.days)[0][0]
            stop = np.int(1+12*(eyear+1-syear))

            var = data.variables[vname][start:start+stop,:,:]
            data.close()
            var = np.ma.masked_array(var)
            var = np.ma.masked_invalid(var)
            np.ma.set_fill_value(var, 0)
            var = var.filled()
            lats, lons, vmap = calculate_map_data(lats,lons,var,lat_bound)
            vmap = vmap*scale

            if vmap.shape == omap.shape:
                diff = vmap - omap
            else:
                imap = inter(lats,lons,vmap)
                diff = imap(olats,olons) - omap

            if mask_near0:
                m0   = (vmap>near0*(-1))*(vmap<near0)
                vmap = np.ma.masked_array(vmap,m0)
                m1   = (diff>near0*(-1))*(diff<near0)
                diff = np.ma.masked_array(diff,m1)
            # Map plots
            plt.clf()
            
            fig,axs = plt.subplots(1,2, figsize=(15,8))
            #fig.subplots_adjust(hspace=0.3)
            #fig.subplots_adjust(wspace=0.3)


            i = 0
            for ax in axs.flat:
                map_ax = Basemap(ax=ax, projection='npstere',
                                 boundinglat=lat_min,lon_0=0.,resolution='l')
                map_ax.drawcoastlines()
                map_ax.drawparallels(np.arange(lat_min,89.,10.))
                map_ax.drawmeridians(np.arange(0.,360.,45.))

                if i==0:
                    x, y = np.meshgrid(lons,lats)
                    con0 = map_ax.contourf(x,y, vmap,
                                           cmap='hot_r',
                                           #cmap='summer_r',
                                           vmin=mlevs[0], vmax=mlevs[-1],
                                           levels=mlevs,
                                           extend='both',
                                           latlon=True)

                    cax0 = fig.add_axes([0.15, 0.1, 0.3, 0.05])
                    cbar = plt.colorbar(con0, cax=cax0, label='g m-2 year-1',
                                        orientation='horizontal') 
                else:
                    x, y = np.meshgrid(olons,olats)
                    con1 = map_ax.contourf(x,y, diff,
                                           cmap='bwr',
                                           #cmap='RdBu',
                                           vmin=dlevs[0], vmax=dlevs[-1],
                                           levels=dlevs,
                                           extend='both',
                                           latlon=True)
                    cax1 = fig.add_axes([0.58, 0.1, 0.3, 0.05])
                    cbar = plt.colorbar(con1, cax=cax1, label='g m-2 year-1',
                                        orientation='horizontal')                    
                i+=1


            axs[0].set_title(model + ' averaged CH4 flux from ' + str(syear) + '-' + str(eyear))
            axs[1].set_title(model + ' - ' + omodel)
            out = 'CH4-' + model + '_' + omodel + '_' + str(syear) + '-' + str(eyear) + ifmt
            plt.savefig(output_dir + out)
            print("Created image:", output_dir + out)



if plot_scatter == True:
    # Scatterplot monthly temperature and methane values
    models = ok_models + map_obs
    fig    = plt.figure(figsize=(10,10))
    for model in models:
        wfile = glob(base_dir + 'wetlandCH4*' + model + '*.nc')
        wdata = Dataset(wfile[0],'r')
        #if model == 'LPJ-GUESS':
            #tfile = glob(base_dir + 'tsoil*' + model + '*.nc')
        if 'RF-' in model:
            tfile = glob(base_dir + 'tas*RF-*.nc')
        else:
            tfile = glob(base_dir + 'tas*' + model + '*.nc')
        if not tfile:
            tfile = glob(base_dir + 'tas*WHOI*.nc')
            print("no temp for",model)
        tdata = Dataset(tfile[0],'r')

        
        # First deal with methane
        try:
            wlats  = wdata.variables['latitude'][:]
        except:
            wlats  = wdata.variables['lat'][:]
        try:
            wlons  = wdata.variables['longitude'][:]
        except:
            wlons  = wdata.variables['lon'][:]
        time  = wdata.variables['time'][:]
        tu    = wdata.variables['time'].units.split(' ')[2]
        tu    = tu.split('-')
        syear = 2000
        if 'RF-' in model:
            syear = 2013
        d0    = date(np.int(tu[0]),np.int(tu[1]),np.int(tu[2]))
        d1    = date(syear,12,31)
        delta = d1-d0
        if model == 'JULES':
            time  = time / (24*60*60)
            vname = 'fch4_wetl_npp'
            start = np.int(12*(2000-np.int(tu[0])))
            eyear = 2013
            stop  = np.int(1+12*(eyear+1-syear))
        else:
            vname = 'wetlandCH4'
            start = np.where(time > delta.days)[0][0]
            eyear = 2014
            stop  = np.int(1+12*(eyear+1-syear))
        if 'WHOI' in tfile[0]:
            eyear = np.min([eyear,2013])
        if wlats[-1]>wlats[0]:
            lat0  = np.where(wlats>lat_min)[0][0]
            var   = wdata.variables[vname][start:start+stop,lat0:,:]
            wlats = wlats[lat0:]
        else:
            lat0  = np.where(wlats>lat_min)[0][-1]
            var   = wdata.variables[vname][start:start+stop,lat0::-1,:]
            wlats = wlats[lat0::-1]
        wdata.close()

        
        # Then deal with temperature
        try:
            tlats  = tdata.variables['latitude'][:]
        except:
            tlats  = tdata.variables['lat'][:]
        try:
            tlons  = tdata.variables['longitude'][:]
        except:
            tlons  = tdata.variables['lon'][:]
        vname = os.path.basename(tfile[0]).split('_')[0]
        tl    = len(var)
        
        if tlats[-1]>tlats[0]:
            lat0 = np.where(tlats>lat_min)[0][0]
            temp = tdata.variables[vname][-tl:,lat0:,:]
            tlats = tlats[lat0:]
        else:
            lat0 = np.where(tlats>lat_min)[0][-1]
            temp = tdata.variables[vname][-tl:,lat0::-1,:]
            tlats = tlats[lat0::-1]
        units = tdata.variables[vname].units
        tdata.close()

        var  = np.ma.masked_array(var)
        var  = np.ma.masked_invalid(var)
        temp = np.ma.masked_less(temp,-99)
        temp = np.ma.masked_invalid(temp)
        if units == 'K' or units =='Kelvin':
            temp = temp - 273.15
        
        # Get monthly values
        mw = np.zeros(len(var))
        mt = np.zeros(len(mw))
        ga = grid_area(wlats)
        
        for i in range(len(mw)):
            mlength = mr(syear + np.int(i/12),1+i%12)[1]*24*60*60
            mw[i] = np.ma.sum(var[i,:,:]*ga)*mlength 
            mt[i] = np.ma.mean(temp[i,:,:])
            #tmap  = inter(tlats,tlons,temp[i,:,:])
            #mt[i] = np.mean(tmap(wlats,wlons)[var.mask[i,:,:]])
            
        #mt = mt - mt.mean()
        mw = mw/mw.max()
        
            
        # Plotting
        colour = get_colour(model)
        plt.scatter(mt,mw,color=colour,s=5,label=model)
        print(model,len(mt))

        
    plt.legend()
    plt.title('Monthly CH4 dependency on temperature for [' + str(lat_min) +'N:90N]')
    outf = output_dir + 'CH4-temp_scatter' + ifmt
    plt.savefig(outf)
    print("Created image:",outf)            
