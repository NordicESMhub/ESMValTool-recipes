'''
diag_cbfs.py

Calls the cbfs module with the approriate inputs.
Used for testing the functions in the module.
This code requires iris, scipy, and numpy.

Module assumes the NetCDF files given are the correct season and region.
Example file are for ERA20th Century reanalysis and the CMIP6 Historical runs for three variants of the NorESM model.
The field in the example files is mean sea level pressure and the domain covers 20N-90N for all longitudes.
Given these files, the modules calculates the NAM (AO) pattern.

There is a flag for single model runs.
If loading a single model, this test all the sub-functions in the module directly.
If loading multiple models, this uses the primary cbfs function.
In both cases, this code plots the EOF from ERA20C and the CBF for NorESM2-LM.

There is a flag toturn off the plotting example.
Plotting requires matplotlib and cartopy.

Written by Stephen Outten October 2021
Transformed to ESMValTool diagnostic by Yanchun He May 2023.
'''

# ************************* Do Not Modify Below This Line **********************

import numpy as np
import sys
import warnings
import cbfs 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib

import logging
from pathlib import Path
from pprint import pformat
import pprint

import iris
import iris.cube 
#
from esmvaltool.diag_scripts.shared import (
    group_metadata,
    get_diagnostic_filename,
    run_diagnostic,
    save_data,
    save_figure,
    select_metadata,
    sorted_metadata,
    ProvenanceLogger,
)
from esmvaltool.diag_scripts.shared.plot import quickplot

logger = logging.getLogger(Path(__file__).stem)


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
            'unmaintained'
        ],
        'references': [
            'NICEST-2',
        ],
        'ancestors': ancestor_files,
    }
    return record

def compute_diagnostic(cfg):
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()

    selection = select_metadata(input_data, short_name='psl', dataset='ERA-20C')
    logger.info("Select observational dataset:\n%s",
                pformat(selection))
    obs_fn=selection[0]['filename']
    obs_vn=selection[0]['short_name']

    # Demonstrate use of metadata access convenience functions.
    selections = select_metadata(input_data, short_name='psl',activity='CMIP')
    logger.info("Select model dataset:\n%s",
                pformat(selection))

    model_vn=selection[0]['short_name']

    model_fn=[]
    datasets=[]
    for selection in selections:
        model_fn.append(selection['filename'])
        datasets.append(selection['alias'])

    cbf,cbf_perc,eof,eof_perc = cbfs.cbfs(model_fn, model_vn, obs_fn, obs_vn, 1)
    odat,ocoord = cbfs.load_data(obs_fn,obs_vn)

    #print('Percentage variance explained for EOF is {0:5.3}%, and for CBF is {1}%.'.format(eof_perc.data,
        #[int(i*100)/100 for i in cbf_perc.data]))

    input_file=model_fn
    attrs={'long_name':'cbf of psl','start_year':1950,'end_year':2010,'dataset':'CMIP models'}
    provenance_record = get_provenance_record(
        attrs, ancestor_files=model_fn)

    basename = 'diag_cbfs'
    #basename = 'diag_cbfs.'+cfg['output_file_type']
    cbf_dict={'datasets':datasets,
            'cbf':cbf,'cbf_perc':cbf_perc,'eof':eof,'eof_perc':eof_perc,
            'odat':odat,'ocoord':ocoord}

    filename = get_diagnostic_filename(basename, cfg)
    logger.info("Saving analysis results to %s", filename)
    #iris.save(cube, target=filename, **kwargs)
    #with ProvenanceLogger(cfg) as provenance_logger:
        #provenance_logger.log(filename, provenance_record)

    #cubes = iris.cube.CubeList([cbf,cbf_perc,eof,eof_perc])
    #save_data(basename, provenance_record, cfg, cubes)

    #iris.save(cubes,'/projects/NS2345K/www/diagnostics/esmvaltool/yanchun/tmp/recipe_cbf_20230519_092454/work/map/script1/cbf_data.nc')
    #output_file = '/path/to/result.nc'
    #with ProvenanceLogger(cfg) as provenance_logger:
        #provenance_logger.log(output_file, provenance_record)
    #"""Compute an example diagnostic."""
    #logger.debug("Loading %s", filename)
    #cube = iris.load_cube(filename)
    return provenance_record, cbf_dict


#def plot_diagnostic(cube, basename, provenance_record, cfg):
    #"""Create diagnostic data and plot it."""

    # Save the data used for the plot
    #save_data(basename, provenance_record, cfg, cube)
#
    #if cfg.get('quickplot'):
        # Create the plot
        #quickplot(cube, **cfg['quickplot'])
        # And save the plot
        #save_figure(basename, provenance_record, cfg)


def plot_diagnostic(cbf_dict, basename, provenance_record, cfg):
    """ Create plot """

    proj = ccrs.PlateCarree(central_longitude=0.0)

    eof = np.asarray(cbf_dict['eof'].data)
    clevs = np.linspace(-10, 10, 21)
    extents = [-120, 120, 20, 90]
    lon = cbf_dict['odat'].coord('longitude').points
    lat = cbf_dict['odat'].coord('latitude').points

    cbf=cbf_dict['cbf']
    ncbf = np.shape(cbf)[0]

    input_data = cfg['input_data'].values()
    selection = select_metadata(input_data, short_name='psl', dataset='ERA-20C')

    ax1 = plt.subplot(1+ncbf, 1, 1, projection=proj)
    ax1.coastlines()
    ax1.set_global()
    ax1.contourf(lon, lat, eof[:,:]/100, levels=clevs, cmap=plt.cm.RdBu_r, transform=proj)
    ax1.axis(extents)
    ax1.set_title(selection[0]['dataset'])

    # Demonstrate use of metadata access convenience functions.
    selections = select_metadata(input_data, short_name='psl',activity='CMIP')
    #print(selections)

    for n in range(0,ncbf):
        cbf_plot = cbf_dict['cbf'][n,:,:].copy()

        ax = plt.subplot(1+ncbf, 1, 2+n, projection=proj)
        ax.coastlines()
        ax.set_global()
        cs = ax.contourf(lon, lat, cbf_plot[:,:]/100, levels=clevs, cmap=plt.cm.RdBu_r, transform=proj)
        ax.axis(extents)
        ax.set_title(selections[n]['dataset']+'_'+selections[n]['ensemble'])

    f1 = plt.gcf()
    #f1.subplots_adjust(right=0.85)
    f1.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=1.0, wspace=0.3)
    cbar_ax = f1.add_axes([0.7, 0.18, 0.02, 0.65])
    f1.colorbar(cs, cax=cbar_ax)

    save_figure(basename, provenance_record, cfg)

def main(cfg):

    """ Compute EOFs of observation and CBFs of models """
    provenance_record, cbf_dict = compute_diagnostic(cfg)

    basename = 'diag_cbfs'
    #filename = get_diagnostic_filename(basename, cfg)
    filename = Path(cfg['plot_dir']) / basename
    plot_diagnostic(cbf_dict, basename, provenance_record, cfg)

    return

if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
