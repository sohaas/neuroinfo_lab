import numpy as np


def load_data(config, type):
    
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

    return [x, y, c, t]


#TODO y > 0 (precipitation in mm?) but x also negative values (deviation from mean?)
def preprocessing(data):
    """
    # already done for train dataset:
    # deviation of the mean instead of amount of rain (normalized)
    """
    return