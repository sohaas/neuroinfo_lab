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

    x = np.load(config['data'][type]['x'])
    #print("x:", x.shape)
    #print("x max value: ", np.amax(x[0, :, :, :]))
    #print("x min value: ", np.amin(x[0, :, :, :]))
    y = np.load(config['data'][type]['y'])
    #print("y:", y.shape)
    #print("y max value: ", np.amax(y[0, :, :]))
    #print("y min value: ", np.amin(y[0, :, :]))
    c = np.load(config['data'][type]['c'])
    #print("c:", c.shape)
    t = np.load(config['data'][type]['t'])
    #print("t:", t.shape)

    data = [x, y, c, t]

    return data


#TODO y > 0 (precipitation in mm?) but x also negative values (deviation from mean?)
def preprocessing(data):
    """
    Preprocess data

    TODO already done for train dataset:
         deviation of the mean instead of amount of rain (normalized)
    """
    return