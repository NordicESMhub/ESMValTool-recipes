# import libraries for logging

from esmvaltool.diag_scripts.shared import run_diagnostic

def main(cfg):
    # Get a description of the preprocessed data that we will use as input.
    input_data = cfg['input_data'].values()
    print(input_data)

if __name__ == '__main__':

    with run_diagnostic() as config:
        main(config)
