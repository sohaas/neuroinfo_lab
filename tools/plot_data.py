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
            config: <dict> config file containing the output directory path
    """

    if data.ndim != 4:
        print("Data should have 4 dimensions but has", data.ndim)
        return
    elif ts > timepoints.size:
        print("Timepoint out of bounds, must be smaller than", timepoints.size)
        return

    plot_data = data[ts, :, :, :]
    date = str(np.datetime64(timepoints[ts])).partition('T')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.mgrid[:plot_data.shape[0], :plot_data.shape[1], :plot_data.shape[2]]
    # scale to color extremes
    img = ax.scatter(x, y, z, c=plot_data.ravel(), norm=colors.SymLogNorm(linthresh=1, linscale=1, vmin=np.amin(plot_data), vmax=np.amax(plot_data), base=10), cmap='RdBu_r')
    cbar = fig.colorbar(img, orientation='horizontal')
    cbar.set_label('Deviation from mean')
    ax.set_xlabel("Area in km")
    ax.set_ylabel("Area in km")
    ax.set_zlabel("Variables")
    plt.title("Meteorological Data \n {} at {}".format(date[0], date[2][:-10]))
    plt.savefig(config['data']['output_path'] + 'input.png')
    plt.close()


def plot_map(data, ts, timepoints, path):
    """
    Plot 2D map of radar precipitation
        Args:
            data: <np.ndarray> input data to plot (3D)
            ts: <int> timestamp, i.e. which observation to plot
            timepoints: <np.ndarray> timestamp information (date, time)
            path <str> output directory path and image name
    """

    if data.ndim != 3:
        print("Data should have 3 dimensions but has", data.ndim)
        return
    elif ts > timepoints.size:
        print("Timepoint out of bounds, must be smaller than", timepoints.size)
        return
    
    plot_data = data[ts, :, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = np.mgrid[:plot_data.shape[0], :plot_data.shape[1]]
    img = ax.contourf(x, y, plot_data, cmap='jet')
    cbar = fig.colorbar(img, orientation='horizontal')
    cbar.set_label('Precipitation in mm/h')
    ax.set_xlabel("Area in km")
    ax.set_ylabel("Area in km")
    if timepoints.size > 1:
        date = str(np.datetime64(timepoints[ts])).partition('T')
        plt.title("Radar Precipitation \n {} at {}".format(date[0], date[2][:-10]))
    elif timepoints.size == 1:
        date = str(np.datetime64(timepoints)).partition('T')
        plt.title("Radar Precipitation \n {} at {}".format(date[0], date[2][:-10]))
    else:
        plt.title("Radar Precipitation")
    plt.savefig(path)
    plt.close()


def plot_train_process(train_losses, test_losses, test_accuracies, config):
    """
    Plot model training process
        Args:
            train_losses: <list> losses of training dataset
            test_losses: <list> losses of test dataset
            test_accuracies: <list> accuracies of test dataset 
            config: <dict> config file containing the output directory path
    """

    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    line3, = plt.plot(test_accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/accuracy")
    plt.legend((line1,line2, line3),("Training", "Test", "Test accuracy"))
    plt.title("Training process")
    plt.savefig(config['data']['output_path'] + 'training_process')
    plt.close()