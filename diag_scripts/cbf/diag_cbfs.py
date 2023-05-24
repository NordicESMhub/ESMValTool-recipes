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

def compute_eof(cfg):
    """ compute_eof """
    obs_fn=cfg['filename']
    obs_vn=cfg['short_name']
    odat, ocoord = cbfs.load_data(obs_fn, obs_vn)
    eof,eof_perc = cbfs.calc_eof(odat, ocoord, nmode=1)

    return odat, ocoord, eof, eof_perc

def compute_cbf(odat, ocoord, eof, cfg):
    """ compute_cbf """
    model_fn = cfg['filename']
    model_vn = cfg['short_name']
    mdat, mcoord = cbfs.load_data(model_fn, model_vn)

    if (mdat.coord(mcoord[1]).shape[0], mdat.coord(mcoord[2]).shape[0]) != (odat.coord(ocoord[1]).shape[0], odat.coord(ocoord[2]).shape[0]):
        print('Observations and model have different latitude and longitudes for model_fn {}. Models must use same spatial grid as observations. Recommend re-gridding model data with CDO.'.format(model_fn[ii]))
        print('Skipping this model.')
        return

    cbf_temp,perc_temp = cbfs.calc_cbf(mdat, eof, mcoord)
    cbf = cbf_temp.copy()
    cbf_perc = perc_temp
    return mdat, mcoord, cbf, cbf_perc

def save_text(basename, provenance_record, cfg, data):
    """Save the scalar data to a text file.

    Parameters
    ----------
    basename: str
        Base name for saving the data into text file
    provenance_record: dict
        Provenance record for the data
    cfg: dict
        Dictionary with diagnostic configuration.
    data: numpy arrary
        Scalar to save.

    See Also
    --------
    ProvenanceLogger: For an example provenance record that can be used
        with this function.
    """

    print('Percentage variance explained by EOF/CBF for {0} is {1:5.3}%.'.format(basename, data))

    filename = get_diagnostic_filename(basename, cfg, extension='txt')
    logger.info("Saving variance contribution to %s", filename)

    # save data to text file
    with open(filename, 'w') as file:
        file.write(str(data))
        
    with ProvenanceLogger(cfg) as provenance_logger:
        provenance_logger.log(filename, provenance_record)


def plot_diagnostic(basename, provenance_record, cfg, cube):
    """Create diagnostic data and plot it."""

    if cfg.get('quickplot'):
        # Create the plot
        quickplot(cube, **cfg['quickplot'])
        plt.gca().coastlines()
        # And save the plot
        save_figure(basename, provenance_record, cfg)


def plot_diagnostic2(basename, provenance_record, cfg, cube):
    """
    Create diagnostic data and plot it.
    Similar as plot_diagnostic, but use direclty matplotlib pyplot
    """

    proj = ccrs.PlateCarree(central_longitude=0.0)

    data = np.asarray(cube.data)
    clevs = np.linspace(-10, 10, 21)
    extents = [-120, 120, 20, 90]
    lat = cube.coord('latitude').points
    lon = cube.coord('longitude').points

    ax = plt.subplot(projection=proj)
    ax.coastlines()
    ax.set_global()
    plot=ax.contourf(lon, lat, data[:,:]/100, levels=clevs, cmap=plt.cm.RdBu_r, transform=proj)
    ax.axis(extents)
    ax.set_title(basename)

    cbar = plt.colorbar(plot,orientation='horizontal',label='EOF spatial correlation')

    save_figure(basename, provenance_record, cfg)

def main(cfg):
    """
    Compute EOF or CBF for each input dataset.
    Save the data and plots.
    """

    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()

    # Compute EOFs of observation
    datasets = select_metadata(input_data, short_name='psl', dataset='ERA-20C')
    for dataset in datasets:
        logger.info("Select only observational data:\n%s",
                    pformat(dataset))
        odat, ocoord, eof, eof_perc = compute_eof(dataset)

        basename = 'eof_' + Path(dataset['filename']).stem
        provenance_record = get_provenance_record(
            dataset, ancestor_files=[dataset['filename']])
        save_data(basename, provenance_record, cfg, eof)
        plot_diagnostic2(basename, provenance_record, cfg, eof)

        basename = 'eof_perc_' + Path(dataset['filename']).stem
        save_text(basename, provenance_record, cfg, eof_perc)

    # Compute CBFs of CMIP model data
    datasets = select_metadata(input_data, short_name='psl', activity='CMIP')
    for dataset in datasets:
        logger.info("Select only CMIP5/6 model data:\n%s",
		    pformat(dataset))
        mdat,mcoord, cbf, cbf_perc = compute_cbf(odat, ocoord, eof, dataset)

        basename = 'cbf_' + Path(dataset['filename']).stem
        provenance_record = get_provenance_record(
            dataset, ancestor_files=[dataset['filename']])
        save_data(basename, provenance_record, cfg, cbf)
        plot_diagnostic2(basename, provenance_record, cfg, cbf)

        basename = 'cbf_perc_' + Path(dataset['filename']).stem
        save_text(basename, provenance_record, cfg, cbf_perc)


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
