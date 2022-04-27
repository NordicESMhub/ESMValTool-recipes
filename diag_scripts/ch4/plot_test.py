#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:46:08 2022

@author: bergmant
"""

import logging
import os
from pprint import pformat
from pathlib import Path
import cartopy.crs as ccrs
import iris
import iris.analysis.maths as imath
import iris.coord_categorisation
import iris.quickplot as qplt
import iris.plot as iplt
import matplotlib.pyplot as plt
import numpy as np
from esmvalcore.preprocessor import extract_region, extract_season

from esmvaltool.diag_scripts.shared import (
    apply_supermeans,
    save_data,
    save_figure,
    get_control_exper_obs,
    group_metadata,
    run_diagnostic,
)
from esmvaltool.diag_scripts.shared._base import ProvenanceLogger
from esmvaltool.diag_scripts.shared.plot import quickplot
from esmvaltool.diag_scripts.shared.plot import global_contourf

logger = logging.getLogger(os.path.basename(__file__))

def load_cube(filename):
    logger.debug("Loading %s", filename)
    cube = iris.load_cube(filename)

    #logger.debug("Running example computation")
    #cube = iris.util.squeeze(cube)
    return cube

def scaling():
    scale_kg_per_m2_per_s_to_g_per_m2_per_year = 1E3*(365*24*60*60)
    return     scale_kg_per_m2_per_s_to_g_per_m2_per_year

def plot_map(cube):
    fig=plt.figure(figsize=(8,8))
    #projection=ccrs.NorthPolarStereo())
    for index, yx_slice in enumerate(cube.slices(['latitude', 'longitude'])):
        logger.info('Plotting year: %s',cube.coord('time'))
        ax=fig.add_subplot(111,projection=ccrs.NorthPolarStereo())
        # Northern Hemisphere from 23 degrees north:
        ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
        #newcube, extent = iris.analysis.cartography.project(yx_slice,ccrs.NorthPolarStereo(), nx=400, ny=200)
        #ax.set_global()
        #plt.subplot(1,cube.coord('time').shape[0],index+1)
        print(repr(yx_slice))
        #logger.info('config quickplot %s',**cfg['quickplot'])
        #quickplot(newcube, **cfg['quickplot'])
        #global_contourf(yx_slice, cbar_range=[-1e-9,10e-9,7])
        scale=scaling()

        #con0=iplt.contourf(yx_slice*scale,cmap='hot_r',vmin=-1e-9, vmax=10e-9,levels=np.linspace(-1e-9,10e-9,11),extend='both')
        con0=iplt.contourf(yx_slice*scale,cmap='hot_r',vmin=-1, vmax=20,levels=np.linspace(-1,20,22),extend='both')
        ax.coastlines()
        #cax0 = fig.add_axes([0.15, 0.1, 0.3, 0.05])
        cbar = plt.colorbar(con0,  label='g m-2 year-1',
                            orientation='horizontal')

def plot_timeseries(cube):
    iplt.plot(cube)


def plot_diagnostic(cube, basename, provenance_record, cfg,variable_group):
    """Create diagnostic data and plot it."""

    # Save the data used for the plot
    save_data(basename, provenance_record, cfg, cube)
    time=cube.coord('time')
    logger.info('times %s',time)
    logger.info('times %s',time[0])
    logger.info(cfg.get('quickplot'))
    #logger.info(cfg['variable_group'])
    logger.info(cfg)
    #logger.info(cfg['variables'])
    if cfg.get('quickplot').get('plot_type')=='polar'and variable_group=='wetlandCH4_map':
        logger.info(cfg.get('quickplot'))
        logger.info(cfg)
        logger.info(str(cube.coord('time')))
        # Create the plot
        logger.info(cube.coord('time'))
        logger.info(cube.coord('time').shape)
        #plt.subplots(nrows=1,ncols=cube.coord('time').shape[0],projection=ccrs.NorthPolarStereo())
        plot_map(cube)
# =============================================================================
#         fig=plt.figure(figsize=(8,8))
#         #projection=ccrs.NorthPolarStereo())
#         for index, yx_slice in enumerate(cube.slices(['latitude', 'longitude'])):
#             logger.info('Plotting year: %s',cube.coord('time'))
#             ax=fig.add_subplot(111,projection=ccrs.NorthPolarStereo())
#             # Northern Hemisphere from 23 degrees north:
#             ax.set_extent([-180, 180, 45, 90], crs=ccrs.PlateCarree())
#             #newcube, extent = iris.analysis.cartography.project(yx_slice,ccrs.NorthPolarStereo(), nx=400, ny=200)
#             #ax.set_global()
#             #plt.subplot(1,cube.coord('time').shape[0],index+1)
#             print(repr(yx_slice))
#             #logger.info('config quickplot %s',**cfg['quickplot'])
#             #quickplot(newcube, **cfg['quickplot'])
#             #global_contourf(yx_slice, cbar_range=[-1e-9,10e-9,7])
#             scale=scaling()
#
#             #con0=iplt.contourf(yx_slice*scale,cmap='hot_r',vmin=-1e-9, vmax=10e-9,levels=np.linspace(-1e-9,10e-9,11),extend='both')
#             con0=iplt.contourf(yx_slice*scale,cmap='hot_r',vmin=-1, vmax=20,levels=np.linspace(-1,20,22),extend='both')
#             ax.coastlines()
#             #cax0 = fig.add_axes([0.15, 0.1, 0.3, 0.05])
#             cbar = plt.colorbar(con0,  label='g m-2 year-1',
#                                 orientation='horizontal')
# =============================================================================
    elif cfg.get('quickplot').get('plot_type')=='times' and variable_group=='wetlandCH4_timeseries':
        logger.info('timeseries plot')
        plot_timeseries(cube)
        #iplt.plot(cube)
        # And save the plot
    else:
        logger.info('no plot')
    save_figure(basename, provenance_record, cfg)

def get_provenance_record(attributes, ancestor_files):
    """Create a provenance record describing the diagnostic data and plot."""
    caption = ("Average {long_name} between {start_year} and {end_year} "
               "according to {dataset}.".format(**attributes))

    record = {
        'caption': caption,
        'statistics': ['mean'],
        'domains': ['global'],
        'plot_types': ['zonal'],
        'authors': [
            'andela_bouwe',
            'righi_mattia',
        ],
        'references': [
            'acknow_project',
        ],
        'ancestors': ancestor_files,
    }
    return record
def main(cfg):
    """Execute validation analysis and plotting."""
    logger.setLevel(cfg['log_level'].upper())
    #input_data, grouped_input_data = do_preamble(cfg)
    input_data=cfg['input_data'].values()
    my_files_dict=group_metadata(input_data,'dataset')
    logger.info('Testing: %s' ,pformat(my_files_dict))
    groups=group_metadata(input_data,'variable_group',sort='dataset')
    for group_name in groups:
        logger.info('Processing variable %s',group_name)
        for attributes in groups[group_name]:
            logger.info('processing dataset %s',attributes['dataset'])
            input_file=attributes['filename']
            data_cube=load_cube(input_file)

            output_basename = Path(input_file).stem
            if group_name != attributes['short_name']:
                output_basename = group_name + '_' + output_basename
            provenance_record = get_provenance_record(
                attributes, ancestor_files=[input_file])
            plot_diagnostic(data_cube, output_basename, provenance_record, cfg,group_name)
if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)