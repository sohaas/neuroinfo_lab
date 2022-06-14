import numpy as np


def load_data(config, type):
    """
    Load dataset
        Args:
            config <dict> config file containing the input file paths
            type: <str> 'train' for training data, 'test' for test data,
                        'validate' for validation data
        Returns:
            data: <list> list containing meteorological data (x), radar 
                         precipitation (y), cosmo predictions (c), and timepoints (t)
    """
    
    if type != 'train' and type != 'test' and type != 'validate':
        print("Data type must be 'train', 'test', or 'validate'.")
        return

    # load meteorological data
    x = np.load(config['data'][type]['x'])
    # load radar precipitation
    y = np.load(config['data'][type]['y'])
    # load cosmo predictions
    c = np.load(config['data'][type]['c'])
    # load timepoints
    t = np.load(config['data'][type]['t'])

    return (x, y, c, t)