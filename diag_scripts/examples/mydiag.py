# import libraries for logging
import logging
from pathlib import Path
from pprint import pformat

# import libraries for logging
from esmvaltool.diag_scripts.shared import run_diagnostic, group_metadata

import iris
import iris.quickplot
import matplotlib.pyplot as plt

#from esmvaltool.diag_scripts.shared.plot import quickplot

logger = logging.getLogger(Path(__file__).stem)

def main(cfg):
    """Compute the time average for each input dataset."""
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()

    #selections = select_metadata(input_data, short_name='tas', project='CMIP6')
    #logger.info("Example of how to select only CMIP6 temperature data:\n%s",
                #pformat(selections))

    # Example of how to loop over dervied variables
    groups = group_metadata(input_data, 'variable_group', sort='dataset')
    for group_name in groups:
        logger.info("Processing variable %s", group_name)
        for attributes in groups[group_name]:
            logger.info("Processing dataset %s", attributes['dataset'])
            input_file = attributes['filename']
            var_name   = attributes['short_name']
            cube = iris.load_cube(input_file)

            basename = Path(input_file).stem
            plot_dir = cfg['plot_dir']
            file_type = cfg['output_file_type']

            plt.figure()
            if "tas_map" in group_name:
                iris.quickplot.pcolormesh(cube,cmap='RdBu')
                plt.gca().coastlines()
                plt.colorbar()

            if "tas_ts" in group_name:
                plot = iris.quickplot.plot(cube)

            plt.savefig(plot_dir+'/'+group_name+'_'+basename+'.'+file_type)

            """
            if cfg.get('quickplot'):
                #Create the plot
                quickplot(cube, **cfg['quickplot'])
                #And save the plot
                save_figure(basename, provenance_record, cfg)
            """


if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
