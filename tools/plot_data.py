import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_cube(data, ts, timepoints, config):
    """
    Plot 3D cube of meteorological data (deviation from mean precipitation),
    scaled to color extreme values
        Args:
            data: <np.ndarray> input data to plot (4D)
            ts: <int> timestamp, i.e. which observation to plot
            timepoints: <np.ndarray> timestamp information (date, time)
            config <dict> config file containing the output directory path
    """

    if data.ndim != 4:
        print("Data should have 4 dimensions but has", data.ndim)
        return
    elif ts > timepoints.shape[0]:
        print("Timepoint out of bounds, must be smaller than", timepoints.shape[0])
        return

    plot_data = data[ts, :, :, :]
    date = str(np.datetime64(timepoints[ts])).partition('T')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.mgrid[:plot_data.shape[0], :plot_data.shape[1], :plot_data.shape[2]]
    img = ax.scatter(x, y, z, c=plot_data.ravel(), norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=np.amin(plot_data), vmax=np.amax(plot_data), base=10), cmap='RdBu_r')
    cbar = fig.colorbar(img, orientation='horizontal')
    cbar.set_label('Deviation from mean')
    #TODO meaning of values ?
    ax.set_xlabel("Area x")
    ax.set_ylabel("Area y")
    ax.set_zlabel("Variables")
    plt.title("Meteorological Data \n {} at {}".format(date[0], date[2][:-10]))
    plt.savefig(config['data']['output_path'] + 'meteorological_data.png')
    plt.show()


def plot_map(data, ts, timepoints, config):
    """
    Plot 2D map of radar precipitation
        Args:
            data: <np.ndarray> input data to plot (3D)
            ts: <int> timestamp, i.e. which observation to plot
            timepoints: <np.ndarray> timestamp information (date, time)
            config <dict> config file containing the output directory path
    """

    if data.ndim != 3:
        print("Data should have 3 dimensions but has", data.ndim)
        return
    elif ts > timepoints.shape[0]:
        print("Timepoint out of bounds, must be smaller than", timepoints.shape[0])
        return
    
    plot_data = data[ts, :, :]
    date = str(np.datetime64(timepoints[ts])).partition('T')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = np.mgrid[:plot_data.shape[0], :plot_data.shape[1]]
    img = ax.contourf(x, y, plot_data, cmap='jet')
    cbar = fig.colorbar(img, orientation='horizontal')
    #TODO unit ?
    cbar.set_label('Precipitation in some unit')
    #TODO meaning of values ?
    ax.set_xlabel("Area x")
    ax.set_ylabel("Area y")
    plt.title("Radar Precipitation \n {} at {}".format(date[0], date[2][:-10]))
    plt.savefig(config['data']['output_path'] + 'radar_precipitation.png')
    plt.show()